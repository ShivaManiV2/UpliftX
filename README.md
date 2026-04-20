# UpliftX

UpliftX is an advanced machine learning project focused on **Uplift Modeling** using the Hillstrom MineThatData E-Mail Analytics challenge dataset. Uplift modeling helps identify the *persuadable* customer segment—those who will only convert if they receive a specific treatment (e.g., an email campaign), allowing businesses to optimize their marketing spend.

## Features
- **Data Ingestion & Preprocessing**: Automated download and cleaning of the Hillstrom dataset.
- **Uplift Modeling**: Implementation of the T-Learner approach using powerful XGBoost classifiers.
- **Evaluation**: Qini curves and decile-based uplift metrics for comprehensive model assessment.
- **Business Simulation**: Built-in logic to estimate Return on Investment (ROI) and incremental margins.
- **Interactive Dashboard**: A sleek, aesthetically pleasing Streamlit app to visualize EDA, Qini curves, and simulate business outcomes in real time.

## Project Structure
```
UpliftX/
├── data/
│   ├── raw/                    # Original Hillstrom CSV
│   └── processed/              # Cleaned data for modeling
├── notebooks/
│   └── 01_eda_and_uplift.ipynb # Exploratory analysis & Qini curves
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Script to fetch and clean Hillstrom data
│   ├── feature_eng.py          # Feature scaling and encoding logic
│   ├── churn_model.py          # Classifier to find at-risk users
│   ├── uplift_model.py         # The T-Learner implementation (Model_T & Model_C)
│   ├── evaluator.py            # Qini Curve and Uplift Decile logic
│   ├── business_sim.py         # ROI, Margin, and Spend calculations
│   └── utils.py                # Model saving/loading and logging
├── models/                     # Saved .pkl or .joblib model files
├── app/
│   ├── app.py                  # Streamlit Dashboard code
│   └── style.css               # Custom UI styling
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md                   # Project documentation
```

## Setup & Installation

1. **Clone the repository (or navigate to the folder)**:
   ```bash
   cd UpliftX
   ```

2. **Create a virtual environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run Data Loader and Model Training
You can run the data loading and training pipeline by using the provided notebook or directly running the modules.
For exploratory analysis, start up Jupyter:
```bash
jupyter notebook notebooks/01_eda_and_uplift.ipynb
```

### 2. Launch the Streamlit Dashboard
To interact with the visualizations and business simulations:
```bash
streamlit run app/app.py
```

## Technologies Used
- Python, Pandas, Scikit-Learn
- XGBoost for predictive modeling
- Streamlit for the frontend application
