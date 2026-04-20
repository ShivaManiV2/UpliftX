import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import save_model, load_model

class TLearner:
    """
    T-Learner approach for uplift modeling.
    Trains two separate models:
    - Model T: Trained on treatment group
    - Model C: Trained on control group
    Uplift = P(Outcome=1 | Treatment) - P(Outcome=1 | Control)
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_t = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=random_state, eval_metric='logloss')
        self.model_c = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=random_state, eval_metric='logloss')
        
    def fit(self, X, y, treatment):
        """
        Fits the two models.
        X: features
        y: target variable (e.g., visit or conversion)
        treatment: binary indicator (1 for treatment, 0 for control)
        """
        # Split data by treatment
        X_t = X[treatment == 1]
        y_t = y[treatment == 1]
        
        X_c = X[treatment == 0]
        y_c = y[treatment == 0]
        
        print(f"Training Model T on {len(X_t)} samples...")
        self.model_t.fit(X_t, y_t)
        
        print(f"Training Model C on {len(X_c)} samples...")
        self.model_c.fit(X_c, y_c)
        
        return self
        
    def predict_uplift(self, X):
        """
        Calculates the uplift score (ITE - Individual Treatment Effect).
        """
        # Probability of outcome if treated
        prob_t = self.model_t.predict_proba(X)[:, 1]
        
        # Probability of outcome if not treated
        prob_c = self.model_c.predict_proba(X)[:, 1]
        
        # Uplift score
        uplift = prob_t - prob_c
        return uplift

def train_uplift_model(X, y, treatment, test_size=0.2, random_state=42, save_dir="models/"):
    """
    Trains the T-Learner and saves the models.
    """
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, treatment, test_size=test_size, random_state=random_state
    )
    
    tlearner = TLearner(random_state=random_state)
    tlearner.fit(X_train, y_train, t_train)
    
    # Save the individual models
    save_model(tlearner.model_t, f"{save_dir}model_t.pkl")
    save_model(tlearner.model_c, f"{save_dir}model_c.pkl")
    
    # Calculate uplift on test set for evaluation
    uplift_scores = tlearner.predict_uplift(X_test)
    
    results = pd.DataFrame({
        'y_true': y_test,
        'treatment': t_test,
        'uplift_score': uplift_scores
    })
    
    return tlearner, results

if __name__ == "__main__":
    from data_loader import load_data, preprocess_basic
    from feature_eng import engineer_features
    
    df = load_data()
    df = preprocess_basic(df)
    X = engineer_features(df, is_training=True)
    
    # Target: visit (1 if visited, 0 otherwise)
    y = df['visit']
    t = df['is_treated']
    
    model, results = train_uplift_model(X, y, t)
    print(results.head())
