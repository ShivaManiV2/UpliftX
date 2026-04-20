import os
import pandas as pd
import requests

def download_hillstrom_data(raw_data_dir="data/raw", filename="hillstrom.csv"):
    """
    Downloads the Hillstrom dataset if it does not already exist.
    """
    url = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    os.makedirs(raw_data_dir, exist_ok=True)
    filepath = os.path.join(raw_data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Dataset downloaded and saved to {filepath}")
    else:
        print(f"Dataset already exists at {filepath}")
        
    return filepath

def load_data(raw_path="data/raw/hillstrom.csv", processed_path="data/processed/hillstrom_cleaned.csv"):
    """
    Loads the Hillstrom dataset. 
    Prioritizes the processed version if it exists, otherwise loads raw.
    Downloads raw if neither exists.
    """
    # 1. Try loading processed data first
    if os.path.exists(processed_path):
        print(f"Loading PROCESSED data from {processed_path}...")
        return pd.read_csv(processed_path)
    
    # 2. If no processed data, check for raw data
    if not os.path.exists(raw_path):
        print(f"Raw data not found at {raw_path}. Initiating download...")
        download_hillstrom_data(raw_data_dir=os.path.dirname(raw_path), filename=os.path.basename(raw_path))
    
    print(f"Loading RAW data from {raw_path}...")
    df = pd.read_csv(raw_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def preprocess_basic(df):
    """
    Performs basic initial preprocessing on the raw data.
    """
    # Rename columns for easier access (lower case, replace space with underscore)
    df.columns = [col.lower().replace('-', '_') for col in df.columns]
    
    # We map segments to simple names for the treatment
    # 'Mens E-Mail', 'Womens E-Mail', 'No E-Mail'
    df['treatment_group'] = df['segment'].map({
        'Mens E-Mail': 'treatment_mens',
        'Womens E-Mail': 'treatment_womens',
        'No E-Mail': 'control'
    })
    
    # For binary treatment effect, often we look at just one vs control.
    # Let's create a binary treatment indicator for any email vs no email
    df['is_treated'] = df['segment'].apply(lambda x: 0 if x == 'No E-Mail' else 1)
    
    return df

def save_processed_data(df, processed_data_dir="data/processed", filename="hillstrom_cleaned.csv"):
    """
    Saves the processed dataframe to the specified directory.
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    filepath = os.path.join(processed_data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")
    return filepath

if __name__ == "__main__":
    df = load_data()
    df = preprocess_basic(df)
    save_processed_data(df)
    print(df.head())
