import pandas as pd
import numpy as np

def calculate_qini(df, target_col='y_true', treatment_col='treatment', uplift_col='uplift_score'):
    """
    Calculates the Qini curve values.
    Returns a dataframe with population cumulative counts and cumulative uplift.
    """
    # Sort by uplift score descending
    df_sorted = df.sort_values(by=uplift_col, ascending=False).reset_index(drop=True)
    
    # Calculate cumulative metrics
    df_sorted['n_t'] = df_sorted[treatment_col].cumsum()
    df_sorted['n_c'] = (1 - df_sorted[treatment_col]).cumsum()
    
    df_sorted['y_t'] = (df_sorted[target_col] * df_sorted[treatment_col]).cumsum()
    df_sorted['y_c'] = (df_sorted[target_col] * (1 - df_sorted[treatment_col])).cumsum()
    
    # Qini curve: Cumulative uplift
    # Q(x) = y_t(x) - y_c(x) * (n_t(x) / n_c(x))
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    df_sorted['uplift_cumulative'] = df_sorted['y_t'] - df_sorted['y_c'] * (df_sorted['n_t'] / (df_sorted['n_c'] + epsilon))
    
    # Random curve (baseline)
    total_y_t = df_sorted['y_t'].iloc[-1]
    total_y_c = df_sorted['y_c'].iloc[-1]
    total_n_t = df_sorted['n_t'].iloc[-1]
    total_n_c = df_sorted['n_c'].iloc[-1]
    
    total_uplift = total_y_t - total_y_c * (total_n_t / (total_n_c + epsilon))
    
    df_sorted['n_pop'] = np.arange(1, len(df_sorted) + 1)
    df_sorted['random_cumulative'] = df_sorted['n_pop'] * (total_uplift / len(df_sorted))
    
    return df_sorted

def get_uplift_by_decile(df, target_col='y_true', treatment_col='treatment', uplift_col='uplift_score'):
    """
    Calculates average uplift by deciles.
    """
    df['decile'] = pd.qcut(df[uplift_col], 10, labels=False, duplicates='drop')
    # Reverse so decile 1 is highest uplift
    df['decile'] = 9 - df['decile'] + 1
    
    decile_stats = []
    
    for dec in sorted(df['decile'].unique()):
        group = df[df['decile'] == dec]
        
        n_t = group[group[treatment_col] == 1].shape[0]
        n_c = group[group[treatment_col] == 0].shape[0]
        
        y_t = group[(group[treatment_col] == 1) & (group[target_col] == 1)].shape[0]
        y_c = group[(group[treatment_col] == 0) & (group[target_col] == 1)].shape[0]
        
        conv_t = y_t / n_t if n_t > 0 else 0
        conv_c = y_c / n_c if n_c > 0 else 0
        
        uplift = conv_t - conv_c
        
        decile_stats.append({
            'decile': dec,
            'n_treatment': n_t,
            'n_control': n_c,
            'conversion_treatment': conv_t,
            'conversion_control': conv_c,
            'uplift': uplift
        })
        
    return pd.DataFrame(decile_stats)
