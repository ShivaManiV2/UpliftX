import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib

def get_feature_pipeline():
    """
    Creates a scikit-learn pipeline for feature transformation.
    Numerical features: standard scaling.
    Categorical features: one-hot encoding.
    """
    numerical_features = ['recency', 'history']
    categorical_features = ['zip_code', 'channel']
    
    # We will pass 'mens', 'womens', 'newbie' as they are already 0/1 indicator variables
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Leave 'mens', 'womens', 'newbie' untouched
    )
    
    return preprocessor, numerical_features, categorical_features

def engineer_features(df, is_training=True, save_path="models/preprocessor.joblib"):
    """
    Applies the feature pipeline to the data.
    """
    # Features to use
    features = ['recency', 'history', 'mens', 'womens', 'zip_code', 'newbie', 'channel']
    
    X = df[features]
    
    if is_training:
        preprocessor, _, _ = get_feature_pipeline()
        X_processed = preprocessor.fit_transform(X)
        
        # Save preprocessor
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(preprocessor, save_path)
    else:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Preprocessor not found at {save_path}. Must train first.")
        preprocessor = joblib.load(save_path)
        X_processed = preprocessor.transform(X)
        
    # Get feature names after transformation to return a dataframe
    try:
        # For newer sklearn versions
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older versions or robust handling
        feature_names = [f"f_{i}" for i in range(X_processed.shape[1])]
        
    # Remove prefixes if any
    feature_names = [name.replace("num__", "").replace("cat__", "").replace("remainder__", "") for name in feature_names]
    
    # Convert to dense if sparse
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
        
    X_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    
    return X_df

if __name__ == "__main__":
    from data_loader import load_data, preprocess_basic
    df = load_data()
    df = preprocess_basic(df)
    X = engineer_features(df, is_training=True)
    print(X.head())
