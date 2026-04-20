import pandas as pd
import numpy as np

def simulate_business_roi(df_uplift, treatment_cost=0.5, revenue_per_conversion=50.0):
    """
    Simulates the business value of targeting based on the uplift model.
    df_uplift needs: 'y_true', 'treatment', 'uplift_score'
    
    treatment_cost: The cost to send the treatment (e.g., mail/discount)
    revenue_per_conversion: Expected revenue if the target variable=1
    
    Returns a dataframe that evaluates ROI by targeting top N% of users based on uplift score.
    """
    df = df_uplift.sort_values(by='uplift_score', ascending=False).reset_index(drop=True)
    
    # We evaluate how we would perform if we target the top x% of the population
    total_population = len(df)
    
    results = []
    
    # Test targeting different deciles (10%, 20%, ..., 100% of population)
    for i in range(1, 11):
        cutoff_index = int(total_population * (i / 10.0))
        targeted_group = df.iloc[:cutoff_index]
        
        # Incremental conversions logic
        # For simplicity, we estimate incremental conversions as the predicted uplift sum 
        # (if uplift_score is probability difference) or we use actuals if this is a holdout set
        
        # Here we use the actuals in the targeted group to estimate true incremental lift
        n_t = targeted_group[targeted_group['treatment'] == 1].shape[0]
        n_c = targeted_group[targeted_group['treatment'] == 0].shape[0]
        
        y_t = targeted_group[(targeted_group['treatment'] == 1) & (targeted_group['y_true'] == 1)].shape[0]
        y_c = targeted_group[(targeted_group['treatment'] == 0) & (targeted_group['y_true'] == 1)].shape[0]
        
        # True Uplift Conversion Rate
        conv_t = y_t / n_t if n_t > 0 else 0
        conv_c = y_c / n_c if n_c > 0 else 0
        incremental_conv_rate = conv_t - conv_c
        
        # Number of people targeted
        n_targeted = len(targeted_group)
        
        # Incremental Conversions = Rate * Targeted
        incremental_conversions = incremental_conv_rate * n_targeted
        
        # Costs and Revenues
        total_cost = n_targeted * treatment_cost
        incremental_revenue = incremental_conversions * revenue_per_conversion
        profit = incremental_revenue - total_cost
        roi = (profit / total_cost) if total_cost > 0 else 0
        
        results.append({
            'Targeted_Percentage': i * 10,
            'N_Targeted': n_targeted,
            'Total_Cost': total_cost,
            'Incremental_Conversions': incremental_conversions,
            'Incremental_Revenue': incremental_revenue,
            'Profit': profit,
            'ROI': roi
        })
        
    return pd.DataFrame(results)
