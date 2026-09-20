# ResistAI Sentinel

[![CI](https://github.com/USERNAME/resistai-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/resistai-sentinel/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Live demo:** _add the Streamlit Cloud URL here after the first deploy_

Explainable antimicrobial-resistance (AMR) risk intelligence for hospitals.
A Streamlit clinical decision-support dashboard backed by a scikit-learn
classifier, a transparent rule engine, and a keyword screener for clinician
notes.

> Synthetic data, decision support only. Not a medical device, not validated
> for clinical use.

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the cohort and train the model
python scripts/generate_data.py
python scripts/train_model.py

# 4. Run the dashboard
streamlit run app.py
```

The app opens at <http://127.0.0.1:8501>.

Step 3 is optional — if the artefacts are missing, the app builds them on
first run (see [Deployment](#deployment)). Running the scripts explicitly is
still the better habit locally: you see the training metrics, and the first
page load is instant.

Requires **Python 3.11 or newer**; 3.13 is what CI and the hosted deployment
use. On Windows, if `python` opens the Microsoft Store, use `py` or call the
virtual environment's interpreter directly:
`.\.venv\Scripts\python.exe`.

In VS Code, `F5` runs any of the three launch configurations in
`.vscode/launch.json` (dashboard, data generation, training).

---

## Project structure

```
resistai-sentinel/
├── app.py                       # Streamlit entry point — layout and wiring only
├── requirements.txt             # Pinned dependencies
├── .python-version              # Interpreter pin for Streamlit Cloud and CI
├── LICENSE                      # MIT
├── .github/workflows/           # CI smoke test + secret scan
├── notebooks/                   # Archived original exploration (not imported)
├── .env.example                 # Secret template (.env is gitignored)
├── .streamlit/config.toml       # Theme + hardened server settings
├── .vscode/                     # Editor settings, launch configs, extensions
├── data/                        # Generated cohort (gitignored)
├── models/                      # Trained model bundle (gitignored)
├── scripts/
│   ├── generate_data.py         # CLI: build the synthetic cohort
│   ├── train_model.py           # CLI: train and persist the model
│   └── smoke_test.py            # CLI: headless render check (used by CI)
└── src/resistai/
    ├── config.py                # All paths, thresholds, hyperparameters
    ├── bootstrap.py             # First-run artefact generation
    ├── data/
    │   ├── generator.py         # Logistic-model synthetic cohort
    │   ├── loader.py            # Load + validate
    │   └── schema.py            # Schema contract and validation
    ├── features/
    │   └── hospital.py          # MRNs, departments, rooms, aggregates
    ├── scoring/
    │   ├── rules.py             # Transparent points-based risk score
    │   └── notes.py             # Keyword screening of clinician notes
    ├── models/
    │   ├── train.py             # Training + ModelBundle artefact
    │   └── inference.py         # AMRPredictor facade
    ├── reporting/
    │   └── pdf.py               # In-memory PDF report
    └── ui/
        ├── theme.py             # Design tokens, CSS, Plotly template
        ├── state.py             # Cached loaders + bootstrap
        └── views/               # One module per dashboard tab
```

Domain logic never imports Streamlit. That boundary is what makes the model,
the rule engine and the report generator usable from a script, a test, or a
future FastAPI service without touching the UI.

---

## What changed from the notebook

The original `digital_hakthone.ipynb` was 75 cells of iterative Colab work
with roughly a dozen competing versions of `app.py`. It is archived at
`notebooks/01_original_exploration.ipynb` with outputs stripped and the
credential redacted; nothing imports it. These are the substantive fixes, not
just the reorganisation.

### 1. Target leakage — the big one

`amr_risk_prob` was passed to the model as a feature *and* was the source of
the `amr_risk_level` label (`"High" if prob > 0.45`). The classifier only had
to learn one threshold on one column, which is why it looked near-perfect and
why `feature_importance` was meaningless.

Now: the cohort is simulated from an explicit logistic model, the observed
outcome (`amr_confirmed`) is a Bernoulli draw from the latent probability,
and `amr_risk_prob` is excluded from `FEATURE_COLUMNS`. Held-out ROC AUC is
**0.73** — a real number on a real problem.

### 2. Non-deterministic identifiers

`uuid.uuid4()` generated a fresh medical record number on every Streamlit
rerun, so the patient selector pointed at a different patient after each
click. `hash()` for room assignment is salted per process, so rooms shuffled
between runs. Both now derive from SHA-256 (`features/hospital.py`).

### 3. Fake motion in the trend chart

The risk trend added `(time.time() % 5) * 0.01`, so the chart moved on its
own with no clinical meaning. Replaced with `risk_trajectory()`, which asks
the model how risk responds as the admission lengthens.

### 4. Model artefact carried no contract

`joblib.dump(model, "amr_model.pkl")` saved a bare estimator. Callers passed
columns in whatever order they liked; sklearn either warned or silently
scored nonsense. Now a `ModelBundle` carries the feature order, the sklearn
version, the training timestamp and the held-out metrics, and
`AMRPredictor._as_frame` reindexes every input to that order.

### 5. Performance

No caching meant re-reading a 4,000-row CSV and re-loading the model on every
widget interaction. `st.cache_data` / `st.cache_resource` in `ui/state.py`
reduce that to once per process, and the whole cohort is scored once.

### 6. Security

See the section below.

---

## Security

The notebook had one hard finding and several soft ones.

| Issue | Where it was | Fix |
|---|---|---|
| **ngrok auth token committed in plaintext** | `ngrok.set_auth_token("38CRk…")` | Removed. Read from `NGROK_AUTHTOKEN` in `.env`, which is gitignored. **Revoke that token** — see below. |
| Server bound to `0.0.0.0` behind a public tunnel with no auth | `--server.address 0.0.0.0` + `ngrok.connect` | `.streamlit/config.toml` binds to `127.0.0.1`, XSRF protection on, upload cap 5 MB. |
| PDF written to a filesystem path built from UI input | `f"/tmp/{patient_uid}_AMR_Report.pdf"` | `reporting/pdf.py` renders to an in-memory `BytesIO`. Nothing is written to disk, so no path traversal and no clinical data left behind. |
| Unescaped interpolation into `unsafe_allow_html` | floor-plan SVG, department names | Values are whitelisted against `HOSPITAL_MAP` and escaped (`html.escape`) before reaching the markup. `risk_badge()` validates the band against known values. |
| Unpinned `!pip install` at runtime | cells 5, 10, 16 | Pinned `requirements.txt`, no installs at runtime. |
| Missing input validation on data load | direct `pd.read_csv` | `data/schema.py` validates columns, types, ranges and ID uniqueness at the boundary. |

### Action required

An ngrok auth token was hardcoded in the original notebook
(`ngrok.set_auth_token("38CRk…")`). It has been redacted from the archived
copy and must be treated as compromised: revoke it at
<https://dashboard.ngrok.com/get-started/your-authtoken> and issue a new one.

The full value is deliberately **not** reproduced anywhere in this repository
— writing a leaked credential into the README would republish it to every
clone and every scanner, which is the problem, not the fix.

### Before this touches real patient data

Everything above assumes a synthetic cohort on a local machine. Real records
would additionally require: authentication and role-based access in front of
the app, TLS termination, an audit log of every prediction and export,
encryption at rest for the cohort and model, a data processing agreement, and
a retention policy. Streamlit has no built-in authentication — put it behind
a reverse proxy that does.

---

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **Create app** → pick the repo, branch `main`, main file `app.py`.
4. Deploy.

Nothing else to configure. `data/` and `models/` are gitignored, so on a fresh
checkout `src/resistai/bootstrap.py` generates the cohort and trains the model
on first boot — roughly 20–30 seconds, once per container. Both steps are
seeded from `config.RANDOM_SEED`, so every deployment of a given commit
produces byte-identical artefacts.

Set `RESISTAI_AUTO_BOOTSTRAP=0` to turn that off and require pre-built
artefacts instead.

The interpreter comes from `.python-version` (3.13). Dependencies are pinned
in `requirements.txt` — if the build fails, that file is the first place to
look.

**Once deployed, put the URL in the "Live demo" line at the top of this file.**

### Continuous integration

Two workflows run on every push and pull request:

| Workflow | What it does |
|---|---|
| `ci.yml` | Installs dependencies, generates an 800-patient cohort, trains the model, and renders every dashboard tab headlessly via `scripts/smoke_test.py`. Fails if any tab raises. |
| `secret-scan.yml` | Runs gitleaks across the full history. Catches a credential before it reaches a public repo — which is the failure mode that started this refactor. |

### Before you make the repository public

Work through this once:

- [ ] The ngrok token from the original notebook is **revoked** at
      <https://dashboard.ngrok.com/get-started/your-authtoken>.
- [ ] `git log -p | grep -i "authtoken\|api_key\|secret"` returns nothing.
      Rewriting history after the fact is painful; checking first is not.
- [ ] `.env` is not tracked: `git ls-files | grep "\.env$"` is empty.
- [ ] `data/` and `models/` contain only `.gitkeep`:
      `git ls-files data models`.
- [ ] The archived notebook has no outputs and no credentials —
      `notebooks/01_original_exploration.ipynb` was stripped, but verify.
- [ ] Enable **Secret scanning** and **Push protection** in the repository's
      Settings → Code security. Both are free on public repositories and stop
      the next leak at `git push` rather than after it.

---

## Design notes

**Two scoring tracks, deliberately.** The random forest predicts, and the
rule engine in `scoring/rules.py` scores the same patient with a six-line
points table. Clinicians can audit a points table; they cannot audit 300
trees. When the two disagree, the Patient tab says so — that disagreement is
a triage signal, not a bug.

**Ablation, not SHAP.** The Explainability tab estimates each feature's local
effect by re-scoring the patient with that feature zeroed. It is cheap and
approximate, and the UI says so. Adding `shap` is a one-file change in
`ui/views/explainability.py` if you want Shapley values.

**Calibration is reported.** ROC AUC tells you the ranking is good; the Brier
score tells you whether a "70% risk" actually means 70%. For a triage tool
that shows a probability to a clinician, the second number matters more.

---

## Extending it

- **Real data** — implement a new loader in `data/`, keep `schema.py` as the
  contract, and nothing downstream changes.
- **A different model** — `models/train.py` returns a `ModelBundle`; swap the
  estimator, keep the interface.
- **An API** — `AMRPredictor` has no Streamlit dependency. A FastAPI service
  wrapping `predict_one` and `score_cohort` is roughly 40 lines.
- **Tests** — the domain modules are pure functions over dataframes. Start
  with `scoring/rules.py` and `data/schema.py`.
