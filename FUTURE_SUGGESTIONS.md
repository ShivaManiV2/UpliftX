# 🔭 UpliftX — Future Suggestions & Roadmap

This document tracks ideas for extending UpliftX beyond its current state. It's organized by effort/impact so you can pick items opportunistically.

---

## Quick wins (low effort, high value)

- **Load pre-trained models instead of retraining every session.** `app/app.py` currently retrains the T-Learner and churn model on every cold cache. Since `src/utils.py` already has `load_model`, wire up: if `models/model_t.pkl` / `model_c.pkl` exist, load them; otherwise train and save. This makes cold starts much faster after the first run.
- **Add a "Retrain model" button** in the sidebar that clears `st.cache_resource` and re-runs training — useful after changing hyperparameters or data.
- **Persist the trained model version/timestamp** somewhere visible in the UI (e.g. sidebar caption) so users know how fresh the current model is.
- **Add unit tests** for `src/evaluator.py` (Qini math), `src/business_sim.py` (ROI math), and `src/feature_eng.py` (pipeline shape) using `pytest`. These are pure functions and easy to test with small synthetic DataFrames.
- **Add a `Makefile` or `justfile`** with `make run`, `make train`, `make test` shortcuts.

---

## Modeling improvements

- **Add more uplift meta-learners for comparison**: S-Learner, X-Learner, and R-Learner alongside the current T-Learner, with a toggle in the dashboard to compare their Qini curves side by side.
- **Hyperparameter tuning**: wrap `TLearner` training in `GridSearchCV`/`Optuna` and expose best params. Currently `n_estimators=100, max_depth=4, learning_rate=0.05` are hardcoded in `src/uplift_model.py`.
- **Cross-validation for Qini estimates**: right now the Qini curve/coefficient is computed on a single train/test split (`test_size=0.2`). K-fold cross-validated Qini estimates would be more robust and let you report a confidence interval.
- **Confidence intervals on uplift scores** (e.g. via bootstrapping) so the dashboard can show uncertainty, not just point estimates.
- **Segment-level uplift** — break down Qini/uplift by `channel`, `zip_code`, or `newbie` status to find which sub-populations respond best.
- **Two-treatment-arm modeling** — the current pipeline collapses "Men's email" and "Women's email" into one `is_treated` flag. A proper 3-arm (or 2-treatment) uplift model could recommend *which* email to send, not just whether to send one.
- **Calibration check** — plot predicted vs. observed uplift by decile to verify the model isn't systematically over/under-confident.

## Data & Feature Engineering

- **Feature engineering ideas**: interaction terms (`recency × history`), log-transform `history` (likely right-skewed), and a customer-lifetime-value proxy feature.
- **Data validation layer** — use `pandera` or `great_expectations` to validate the schema/ranges of the Hillstrom CSV before training, so a corrupted download fails loudly instead of silently producing a bad model.
- **Support multiple datasets** — abstract `data_loader.py` behind a small interface so UpliftX could plug in a different (e.g. proprietary) treatment/control dataset without touching the modeling code.

## Business Simulation

- **Cost curve realism**: `simulate_business_roi` uses a flat per-customer cost. Real campaigns often have fixed + variable costs, or per-channel costs (SMS vs. email vs. print) — model that.
- **Multi-scenario comparison**: let users save 2–3 cost/revenue scenarios and compare profit curves overlaid on one chart.
- **Budget-constrained optimization**: instead of "target top N%", solve "given a fixed budget $B, what's the profit-maximizing targeting depth?"

## Dashboard / UX

- **Authentication / multi-user support** if this is ever deployed for a team, not just local use (Streamlit supports this via `streamlit-authenticator` or a reverse proxy).
- **CSV upload** — let a user upload their own treatment/control dataset (with a required schema) instead of only using the bundled Hillstrom data.
- **Model comparison page** — once multiple learners exist, a page to compare Qini coefficients, decile lift, and ROI curves across models.
- **Export a full PDF report** (not just CSV) summarizing the current model's Qini coefficient, top features, and optimal targeting depth — good for sharing with non-technical stakeholders.
- **Light theme toggle** — the current UI is a fixed premium dark theme. See [`UI_CUSTOMIZATION_GUIDE.md`](UI_CUSTOMIZATION_GUIDE.md) for where a light theme variant would go.
- **Mobile responsiveness pass** — the multi-column layouts (`st.columns`) collapse reasonably in Streamlit already, but the hero banner and flow-diagram CSS could use explicit narrow-viewport rules.

## Engineering / Ops

- **CI pipeline** (GitHub Actions) that runs `pip install`, `pytest`, and a `streamlit run --headless` smoke test on every push.
- **Containerize** with a `Dockerfile` so the dashboard can be deployed consistently (Streamlit Community Cloud, a VM, or a container platform).
- **Config file** (`config.yaml` or `.env`) for paths, hyperparameters, and default business-sim assumptions instead of hardcoded literals scattered across `src/`.
- **Logging** — `src/utils.py` already has `setup_logger`; wire it into `data_loader.py`, `uplift_model.py`, and `churn_model.py` (currently they just use `print`).
- **Type hints** across `src/` for better editor support and to catch integration bugs earlier (e.g. `mypy` in CI).

---

## How to prioritize

If you only do three things next:
1. **Load pre-trained models on cold start** (biggest UX win, smallest effort).
2. **Add pytest coverage for the math-heavy modules** (`evaluator.py`, `business_sim.py`) — these are the modules most likely to silently break if edited.
3. **Add an S-Learner or X-Learner comparison** — the single biggest modeling credibility upgrade for a portfolio/demo project.
