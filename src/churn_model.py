import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
import os
from .utils import save_model

def train_churn_model(X, y_churn, test_size=0.2, random_state=42, save_path="models/churn_model.pkl"):
    """
    Trains a baseline model to predict 'churn' (or no visit/conversion).
    We define churn = 1 if user did NOT visit.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=test_size, random_state=random_state)
    
    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=4, 
        eval_metric='logloss',
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("--- Churn Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if save_path:
        save_model(model, save_path)
        
    return model

if __name__ == "__main__":
    from data_loader import load_data, preprocess_basic
    from feature_eng import engineer_features
    
    df = load_data()
    df = preprocess_basic(df)
    X = engineer_features(df, is_training=True)
    
    # Define churn as 1 if visit == 0 (did not visit)
    y_churn = (df['visit'] == 0).astype(int)
    
    train_churn_model(X, y_churn)
