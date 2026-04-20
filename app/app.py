import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure src is in path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data, preprocess_basic
from src.feature_eng import engineer_features
from src.uplift_model import train_uplift_model
from src.evaluator import calculate_qini, get_uplift_by_decile
from src.business_sim import simulate_business_roi
from src.utils import load_model

# Page config
st.set_page_config(page_title="UpliftX Dashboard", page_icon="🚀", layout="wide")

# Load Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("app/style.css")
except:
    pass

# -----------------
# App State & Data
# -----------------
@st.cache_data
def get_data():
    # load_data handles both raw and processed paths
    df = load_data()
    if 'is_treated' not in df.columns:
        df = preprocess_basic(df)
    return df

@st.cache_data
def get_features_and_results(df):
    X = engineer_features(df, is_training=True, save_path='models/preprocessor.joblib')
    y = df['visit']
    t = df['is_treated']
    model, results = train_uplift_model(X, y, t, save_dir='models/')
    return X, results

st.title("🚀 UpliftX")
st.markdown("##### Strategic Customer Targeting & ROI Simulation")

try:
    df = get_data()
    X, results = get_features_and_results(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview & EDA", "Model Evaluation (Qini)", "Business Simulation"])

# -----------------
# 1. Overview & EDA
# -----------------
if page == "Overview & EDA":
    st.header("Dataset Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    
    treatment_cr = df[df['is_treated'] == 1]['visit'].mean() * 100
    control_cr = df[df['is_treated'] == 0]['visit'].mean() * 100
    avg_uplift = treatment_cr - control_cr
    
    col2.metric("Treatment CR", f"{treatment_cr:.2f}%")
    col3.metric("Control CR", f"{control_cr:.2f}%")
    col4.metric("Avg Uplift", f"{avg_uplift:.2f}%", delta=f"{avg_uplift:.2f}%")
    
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("Visit Conversion by Group")
        # Interactive Plotly count plot
        fig = px.histogram(df, x="treatment_group", color="visit", 
                           barmode="group", color_discrete_sequence=px.colors.qualitative.Pastel,
                           template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Sample Records")
        st.dataframe(df.head(10), use_container_width=True)

# -----------------
# 2. Model Evaluation
# -----------------
elif page == "Model Evaluation (Qini)":
    st.header("Performance Analytics")
    
    qini_df = calculate_qini(results)
    
    st.subheader("Interactive Qini Curve")
    st.markdown("Hover over the lines to see exact incremental conversion values.")
    
    fig = go.Figure()
    
    # Model Curve
    fig.add_trace(go.Scatter(
        x=qini_df['n_pop'] / len(qini_df), 
        y=qini_df['uplift_cumulative'],
        mode='lines',
        name='T-Learner Strategy',
        line=dict(color='#60a5fa', width=4),
        hovertemplate='<b>%{x:.1%} Targeted</b><br>Incremental Conversions: %{y:.0f}<extra></extra>'
    ))
    
    # Baseline
    fig.add_trace(go.Scatter(
        x=qini_df['n_pop'] / len(qini_df), 
        y=qini_df['random_cumulative'],
        mode='lines',
        name='Random Baseline',
        line=dict(color='#94a3b8', width=2, dash='dash'),
        hovertemplate='<b>%{x:.1%} Targeted</b><br>Random Incremental: %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Proportion of Population Targeted",
        yaxis_title="Cumulative Uplift",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Incremental Lift by Decile")
    decile_df = get_uplift_by_decile(results)
    
    fig2 = px.bar(decile_df, x='decile', y='uplift', 
                  color='uplift', color_continuous_scale='Viridis',
                  labels={'uplift': 'Incremental Conversion Rate'},
                  template="plotly_dark")
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

# -----------------
# 3. Business Simulation
# -----------------
elif page == "Business Simulation":
    st.header("Strategic ROI Simulation")
    
    st.sidebar.subheader("Campaign Parameters")
    cost_per_treatment = st.sidebar.slider("Cost per Treatment ($)", 0.0, 5.0, 0.5, 0.1)
    revenue_per_conversion = st.sidebar.slider("Revenue per Visit/Conversion ($)", 0.0, 200.0, 50.0, 5.0)
    
    roi_df = simulate_business_roi(results, treatment_cost=cost_per_treatment, revenue_per_conversion=revenue_per_conversion)
    
    optimal_row = roi_df.loc[roi_df['Profit'].idxmax()]
    
    # Modern highlight card
    st.info(f"💡 **Optimal Strategy Found:** By targeting the top **{optimal_row['Targeted_Percentage']}%** of customers, you maximize profit at **${optimal_row['Profit']:,.2f}**.")
    
    st.subheader("Profit Optimization Curve")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roi_df['Targeted_Percentage'], 
        y=roi_df['Profit'],
        mode='lines+markers',
        line=dict(color='#a78bfa', width=3),
        marker=dict(size=8, color='#60a5fa'),
        name='Profit ($)',
        hovertemplate='Targeted: %{x}%<br>Profit: $%{y:,.2f}<extra></extra>'
    ))
    
    # Mark optimal point
    fig.add_annotation(
        x=optimal_row['Targeted_Percentage'],
        y=optimal_row['Profit'],
        text="Max Profit",
        showarrow=True,
        arrowhead=2,
        bgcolor="#1e1b4b",
        bordercolor="#60a5fa"
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Targeting Depth (%)",
        yaxis_title="Incremental Profit ($)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Simulation Breakdown")
    st.dataframe(roi_df.style.format({
        'Total_Cost': '${:,.2f}',
        'Incremental_Revenue': '${:,.2f}',
        'Profit': '${:,.2f}',
        'ROI': '{:.2%}'
    }), use_container_width=True)
