# 🚀 UpliftX

UpliftX is a machine learning project focused on **Uplift Modeling** using the Hillstrom MineThatData E-Mail Analytics challenge dataset. Uplift modeling identifies the *persuadable* customer segment — the people who will only convert **because** they received a specific treatment (e.g. a marketing email) — so businesses can target spend where it actually creates incremental value, instead of on customers who would have converted (or never will) regardless.

Unlike a plain response/propensity model, UpliftX separates customers into four latent groups (persuadables, sure things, lost causes, sleeping dogs) by modeling treatment and control populations independently and comparing predicted outcomes.

---

## ✨ Features

- **Data Ingestion & Preprocessing** — automated download and cleaning of the public Hillstrom dataset.
- **Uplift Modeling** — a T-Learner built from two independent XGBoost classifiers (`Model_T`, `Model_C`).
- **Churn Risk Modeling** — a standalone XGBoost classifier that flags customers unlikely to convert regardless of treatment, useful for suppression/win-back lists.
- **Evaluation** — Qini curves, the Qini coefficient (normalized AUUC), and decile-based uplift metrics.
- **Business Simulation** — ROI, incremental margin, and profit-optimal targeting depth, driven by adjustable cost/revenue assumptions.
- **Interactive Dashboard** — a 5-page Streamlit app (Overview, Data Explorer, Model Evaluation, Churn Risk, Business Simulation) with a custom glassmorphism dark theme.

---

## 📁 Project Structure

```
UpliftX/
├── data/
│   ├── raw/                    # Original Hillstrom CSV (downloaded on first run)
│   └── processed/              # Cleaned data for modeling
├── notebooks/
│   └── 01_eda_and_uplift.ipynb # Exploratory analysis & Qini curves
├── models/                     # Saved .pkl / .joblib model artifacts (created at runtime)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Fetch and clean the Hillstrom data
│   ├── feature_eng.py          # Feature scaling and encoding (sklearn ColumnTransformer)
│   ├── churn_model.py          # Classifier to find at-risk (non-converting) users
│   ├── uplift_model.py         # T-Learner implementation (Model_T & Model_C)
│   ├── evaluator.py            # Qini curve, Qini coefficient, decile & feature-importance logic
│   ├── business_sim.py         # ROI, margin, and spend calculations
│   └── utils.py                # Model save/load and logging helpers
├── app/
│   ├── app.py                  # Streamlit dashboard
│   └── style.css               # Custom UI theme (glassmorphism, dark)
├── requirements.txt
├── README.md
├── FUTURE_SUGGESTIONS.md        # Roadmap / ideas for the next iteration
├── UI_CUSTOMIZATION_GUIDE.md    # Map of "what to edit, and where" for UI changes
├── PROJECT_KNOWLEDGE.html       # Full project knowledge doc (printable to PDF)
└── .gitignore
```

---

## ⚙️ Setup & Installation

1. **Clone the repository (or navigate to the folder)**
   ```bash
   cd UpliftX
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### 1. Exploratory analysis (optional)
```bash
jupyter notebook notebooks/01_eda_and_uplift.ipynb
```

### 2. Launch the dashboard
```bash
streamlit run app/app.py
```

On first launch, the app will automatically:
1. Download the Hillstrom dataset into `data/raw/` (requires internet access on first run only).
2. Engineer features and cache the fitted preprocessor to `models/preprocessor.joblib`.
3. Train the T-Learner (`model_t.pkl`, `model_c.pkl`) and the churn model (`churn_model.pkl`).
4. Cache all of the above via `st.cache_data` / `st.cache_resource` so subsequent interactions (slider changes, page switches) are instant.

### Dashboard pages

| Page | What it shows |
|---|---|
| 🏠 **Overview** | Project summary, methodology, pipeline diagram |
| 📊 **Data Explorer** | Dataset stats, conversion rates by group/channel, distributions |
| 🎯 **Model Evaluation** | Qini curve, Qini coefficient, decile uplift, feature importance |
| ⚠️ **Churn Risk** | Churn model accuracy/AUC, risk distribution, top at-risk customers |
| 💰 **Business Simulation** | Interactive cost/revenue sliders, profit curve, CSV export |

---

## 🧠 Methodology in brief

- **Outcome (`y`)**: `visit` — did the customer visit the site.
- **Treatment (`t`)**: `is_treated` — did the customer receive *any* e-mail (Men's or Women's) vs. no e-mail.
- **T-Learner**: two independent XGBoost classifiers are trained — `Model_T` on treated customers, `Model_C` on control customers. The uplift score for a customer is `P(visit=1 | X, treated) − P(visit=1 | X, control)`.
- **Qini coefficient**: the area between the model's cumulative-uplift (Qini) curve and the random-targeting baseline, normalized by population size — a single number summarizing how much better than random the model's ranking is.
- **Business simulation**: for each targeting depth (10%–100% of the population, ranked by uplift score), it computes true incremental conversions, cost, revenue, profit and ROI, and reports the profit-maximizing depth.

---

## 🛠️ Technologies Used

- Python, Pandas, NumPy, scikit-learn
- XGBoost for predictive modeling
- Plotly for interactive charts
- Streamlit for the frontend application

---

## 📚 Further reading

- [`FUTURE_SUGGESTIONS.md`](FUTURE_SUGGESTIONS.md) — roadmap and ideas for extending the project.
- [`UI_CUSTOMIZATION_GUIDE.md`](UI_CUSTOMIZATION_GUIDE.md) — where to look when you want to change the dashboard's look, layout, or add a new page.
- [`PROJECT_KNOWLEDGE.html`](PROJECT_KNOWLEDGE.html) — a comprehensive, print-to-PDF-friendly knowledge document covering everything about this project.
