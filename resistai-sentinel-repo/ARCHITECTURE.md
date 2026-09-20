# Architecture

## 1. Layering

Four layers, one direction of dependency. Nothing below the UI line imports
Streamlit.

```
┌─────────────────────────────────────────────────────────────┐
│  PRESENTATION           app.py  ·  ui/theme  ·  ui/state     │
│                         ui/views/{overview, patient,          │
│                         hospital, explainability,             │
│                         simulator, actions}                   │
└───────────────────────────────┬─────────────────────────────┘
                                │ reads only
┌───────────────────────────────▼─────────────────────────────┐
│  DOMAIN         models/inference   scoring/rules             │
│                 models/train       scoring/notes             │
│                 features/hospital  reporting/pdf             │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│  DATA           data/generator  ·  data/loader  ·  schema    │
└───────────────────────────────┬─────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────┐
│  CONFIG         config.py  (paths, thresholds, feature list, │
│                 hyperparameters, hospital topology)          │
└─────────────────────────────────────────────────────────────┘
```

`config.py` is imported by every layer and imports nothing from the project.
It is the single place a threshold or a path is defined.

## 2. Data flow

### Offline (run once, or whenever the cohort changes)

```
GeneratorSettings
     │
     ▼
data/generator.generate_cohort()
     │   logistic model → latent probability → Bernoulli outcome
     ▼
schema.validate_dataset()  →  data/amr_dataset.csv
     │
     ▼
models/train.train_model()
     │   stratified split → RandomForest → held-out metrics
     ▼
ModelBundle  →  models/amr_model.joblib
     (estimator + feature_names + positive_class_index + metrics + versions)
```

### Online (every Streamlit run)

```
app.main()
     │
     ├─ ui.state.bootstrap()
     │      ├─ get_predictor()      @cache_resource  → load_predictor()
     │      ├─ get_dataset()        @cache_data      → load + validate
     │      └─ get_scored_cohort()  @cache_data      → score all 4,000 once
     │
     ├─ _sidebar()  → department filter → risk filter → patient_uid
     │
     ├─ selected_patient()  → exact-match lookup, never an eval'd filter
     │
     └─ tabs
          Overview        ← scored cohort (department_summary)
          Patient         ← record + rule_score + note_flags
          Hospital map    ← scored cohort (department_summary, room_summary)
          Explainability  ← feature_importance + ablation counterfactuals
          Simulator       ← predictor.counterfactual / risk_trajectory
          Actions         ← recommendation + room hotspots + PDF bytes
```

Cohort scoring happens once per process, not once per interaction. Only the
simulator and the ablation chart call the model per-render, and both operate
on a single row.

## 3. Key interfaces

| Component | Entry point | Contract |
|---|---|---|
| Cohort generation | `generate_cohort(settings) -> DataFrame` | Conforms to `DATASET_COLUMNS` |
| Validation | `validate_dataset(df) -> DataFrame` | Raises `DatasetValidationError` |
| Training | `train_model(df, settings) -> ModelBundle` | Bundle carries feature order + metrics |
| Inference | `AMRPredictor.predict_one(row) -> (float, str)` | Reindexes to bundle feature order |
| Counterfactual | `AMRPredictor.counterfactual(row, overrides)` | Rejects unknown feature names |
| Rule engine | `rule_score(row) -> int` in `[0, 10]` | Pure function, no state |
| Note screening | `note_flags(text) -> list[str]` | Pure function, no state |
| Report | `build_patient_report(...) -> bytes` | In-memory, no filesystem writes |

## 4. The feature contract

`config.FEATURE_COLUMNS` is the single definition of what the model consumes.
Three components read it and none of them redefine it:

- `generator.py` — produces exactly these columns
- `train.py` — selects exactly these columns, stores the order in the bundle
- `inference.py` — reindexes every input to the stored order

That chain is what prevents the class of failure where the UI passes columns
in a different order than training used. Adding a feature means editing one
list and re-running the two scripts.

## 5. Threat model (local deployment)

| Asset | Threat | Control |
|---|---|---|
| Cohort CSV | Tampering → poisoned predictions | Schema validation on load |
| Model artefact | Malicious pickle → RCE on load | Type check on unpickle; artefact is gitignored and locally produced |
| Secrets | Credential leak via source control | `.env` only, `.gitignore`, `.env.example` template |
| Dashboard | Unauthenticated network exposure | Loopback bind, XSRF on, no public tunnel |
| Rendered HTML | Injection via `unsafe_allow_html` | Whitelist + `html.escape` on every interpolated value |
| PDF export | Path traversal, residual PHI on disk | In-memory `BytesIO`, no filesystem write |

`joblib.load` executes pickled code by construction. The bundle here is
produced locally by `scripts/train_model.py` and never downloaded, which is
the only safe posture. If a model artefact ever arrives from outside, it must
be signed and verified before loading — a type check is a guard rail, not a
security boundary.

## 6. Where this goes next

Ordered by value, not by effort.

1. **Tests** — `scoring/rules.py`, `data/schema.py` and
   `models/inference._as_frame` are pure and cover the highest-risk logic.
2. **Calibration** — wrap the forest in `CalibratedClassifierCV`. The Brier
   score of 0.21 says the displayed probabilities are usable but not tight.
3. **SHAP** — replace the ablation estimate in the Explainability tab.
4. **FastAPI service** — expose `predict_one` and `score_cohort`; the domain
   layer already has no UI dependency.
5. **Real data adapter** — a second loader behind the same schema contract.
6. **Auth + audit** — required before any real patient record enters the system.
