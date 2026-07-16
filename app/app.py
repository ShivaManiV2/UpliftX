import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Ensure src is in path to import modules regardless of the working directory
# the app is launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from src.data_loader import load_data, preprocess_basic
from src.feature_eng import engineer_features
from src.uplift_model import train_uplift_model
from src.churn_model import train_churn_model
from src.evaluator import calculate_qini, get_uplift_by_decile, calculate_qini_coefficient, get_top_feature_importance
from src.business_sim import simulate_business_roi

# -----------------
# Page Config
# -----------------
st.set_page_config(
    page_title="UpliftX Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


def local_css(file_name):
    """Loads a CSS file relative to this script's directory, regardless of cwd."""
    css_path = os.path.join(APP_DIR, file_name)
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


try:
    local_css("style.css")
except FileNotFoundError:
    st.warning("Custom styling (style.css) not found — using default Streamlit theme.")

PAGES = [
    "🏠 Overview",
    "📊 Data Explorer",
    "🎯 Model Evaluation",
    "⚠️ Churn Risk",
    "💰 Business Simulation",
]

# -----------------
# Cached data / model pipeline
# -----------------
@st.cache_data(show_spinner=False)
def get_data():
    df = load_data()
    if 'is_treated' not in df.columns:
        df = preprocess_basic(df)
    return df


@st.cache_data(show_spinner=False)
def get_features(df):
    return engineer_features(df, is_training=True, save_path=os.path.join(PROJECT_ROOT, 'models', 'preprocessor.joblib'))


@st.cache_resource(show_spinner=False)
def get_uplift_pipeline(_df, _X):
    y = _df['visit']
    t = _df['is_treated']
    model, results = train_uplift_model(_X, y, t, save_dir=os.path.join(PROJECT_ROOT, 'models') + os.sep)
    return model, results


@st.cache_resource(show_spinner=False)
def get_churn_pipeline(_df, _X):
    y_churn = (_df['visit'] == 0).astype(int)
    model, X_test, y_test, y_prob, metrics = train_churn_model(
        _X, y_churn, save_path=os.path.join(PROJECT_ROOT, 'models', 'churn_model.pkl')
    )
    return model, X_test, y_test, y_prob, metrics


def card_open(title=None, subtitle=None):
    header = ""
    if title:
        header += f'<div class="uplift-card-title">{title}</div>'
    if subtitle:
        header += f'<div class="uplift-card-subtitle">{subtitle}</div>'
    st.markdown(f'<div class="uplift-card">{header}', unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------
# Header
# -----------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🚀 UpliftX</div>
        <div class="hero-subtitle">Strategic Customer Targeting, Churn Intelligence &amp; ROI Simulation</div>
        <div class="hero-badges">
            <span class="badge badge-blue">T-Learner</span>
            <span class="badge badge-purple">XGBoost</span>
            <span class="badge badge-pink">Qini Evaluation</span>
            <span class="badge badge-green">ROI Simulation</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------
# Load data & train models (with visible progress)
# -----------------
try:
    with st.spinner("Loading dataset..."):
        df = get_data()
    with st.spinner("Engineering features..."):
        X = get_features(df)
    with st.spinner("Training T-Learner uplift model (Model_T & Model_C)..."):
        uplift_model, results = get_uplift_pipeline(df, X)
except Exception as e:
    st.error(f"Error loading data or training the uplift model: {e}")
    st.stop()

# -----------------
# Sidebar
# -----------------
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ About this dataset", expanded=False):
    st.markdown(
        """
        **Hillstrom MineThatData E-Mail Analytics** challenge dataset
        (~64k customers). Each customer was randomly assigned to receive a
        Men's e-mail campaign, a Women's e-mail campaign, or no e-mail
        (control). We model **visit** as the outcome and treat any e-mail
        as the treatment.
        """
    )

st.sidebar.markdown(
    '<div class="sidebar-footer">Built with Streamlit • XGBoost T-Learner<br>UpliftX v2.0</div>',
    unsafe_allow_html=True,
)

# =========================================================
# 1. Overview
# =========================================================
if page == "🏠 Overview":
    st.header("Welcome to UpliftX")
    st.markdown(
        "UpliftX helps you move beyond simple churn/response prediction and answer the "
        "question that actually drives marketing ROI: **who should we target, and who "
        "should we leave alone?**"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        card_open("📦 What it does")
        st.markdown(
            "- Trains a **T-Learner uplift model** (two XGBoost classifiers) on the "
            "Hillstrom e-mail dataset\n"
            "- Estimates the **incremental** effect of treatment per customer, not "
            "just their likelihood to convert\n"
            "- Flags customers likely to churn regardless of treatment"
        )
        card_close()
    with c2:
        card_open("🧠 Why uplift, not response")
        st.markdown(
            "A response model tells you *who will convert* — many of them would have "
            "converted anyway. An uplift model tells you *who converts **because** of "
            "the treatment* (the persuadables), so spend isn't wasted on sure things "
            "or lost causes."
        )
        card_close()
    with c3:
        card_open("📈 What you can explore")
        st.markdown(
            "- **Data Explorer** — dataset shape & distributions\n"
            "- **Model Evaluation** — Qini curve, deciles, feature importance\n"
            "- **Churn Risk** — at-risk customer identification\n"
            "- **Business Simulation** — profit-optimal targeting depth"
        )
        card_close()

    st.markdown("---")
    st.subheader("How the pipeline works")
    st.markdown(
        """
        <div class="flow-row">
            <div class="flow-step">1. Raw Data<br><span>Hillstrom CSV</span></div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">2. Feature Engineering<br><span>scale + one-hot encode</span></div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">3. T-Learner<br><span>Model_T &amp; Model_C</span></div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">4. Uplift Score<br><span>P(T) − P(C)</span></div>
            <div class="flow-arrow">→</div>
            <div class="flow-step">5. Qini + ROI<br><span>evaluate &amp; simulate</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 2. Data Explorer
# =========================================================
elif page == "📊 Data Explorer":
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
        fig = px.histogram(df, x="treatment_group", color="visit",
                            barmode="group", color_discrete_sequence=px.colors.qualitative.Pastel,
                            template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Sample Records")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    c3, c4 = st.columns([1, 1])
    with c3:
        st.subheader("Recency Distribution")
        fig_rec = px.histogram(df, x="recency", nbins=12, color_discrete_sequence=['#60a5fa'],
                                template="plotly_dark")
        fig_rec.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rec, use_container_width=True)

    with c4:
        st.subheader("Historical Spend Distribution")
        fig_hist = px.histogram(df, x="history", nbins=30, color_discrete_sequence=['#a78bfa'],
                                 template="plotly_dark")
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Conversion Rate by Acquisition Channel")
    channel_conv = df.groupby('channel')['visit'].mean().reset_index()
    fig_chan = px.bar(channel_conv, x='channel', y='visit', color='channel',
                       color_discrete_sequence=px.colors.qualitative.Pastel,
                       labels={'visit': 'Conversion Rate'}, template="plotly_dark")
    fig_chan.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    st.plotly_chart(fig_chan, use_container_width=True)

# =========================================================
# 3. Model Evaluation
# =========================================================
elif page == "🎯 Model Evaluation":
    st.header("Performance Analytics")

    qini_df = calculate_qini(results)
    qini_coef = calculate_qini_coefficient(qini_df)

    m1, m2, m3 = st.columns(3)
    m1.metric("Qini Coefficient", f"{qini_coef:,.2f}", help="Area between the model's Qini curve and the random baseline. Higher is better.")
    m2.metric("Test-set Size", f"{len(results):,}")
    m3.metric("Observed Avg. Uplift", f"{(results[results.treatment==1].y_true.mean() - results[results.treatment==0].y_true.mean())*100:.2f}%")

    st.markdown("---")

    st.subheader("Interactive Qini Curve")
    st.markdown("Hover over the lines to see exact incremental conversion values.")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=qini_df['n_pop'] / len(qini_df),
        y=qini_df['uplift_cumulative'],
        mode='lines',
        name='T-Learner Strategy',
        line=dict(color='#60a5fa', width=4),
        hovertemplate='<b>%{x:.1%} Targeted</b><br>Incremental Conversions: %{y:.0f}<extra></extra>'
    ))

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

    st.markdown("---")

    st.subheader("What drives the model's predictions?")
    fi1, fi2 = st.columns(2)

    imp_t = get_top_feature_importance(uplift_model.model_t, X.columns)
    imp_c = get_top_feature_importance(uplift_model.model_c, X.columns)

    with fi1:
        st.markdown("**Model_T** (trained on treated customers)")
        fig_imp_t = px.bar(imp_t.sort_values('importance'), x='importance', y='feature', orientation='h',
                            color_discrete_sequence=['#60a5fa'], template="plotly_dark")
        fig_imp_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp_t, use_container_width=True)

    with fi2:
        st.markdown("**Model_C** (trained on control customers)")
        fig_imp_c = px.bar(imp_c.sort_values('importance'), x='importance', y='feature', orientation='h',
                            color_discrete_sequence=['#a78bfa'], template="plotly_dark")
        fig_imp_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp_c, use_container_width=True)

# =========================================================
# 4. Churn Risk
# =========================================================
elif page == "⚠️ Churn Risk":
    st.header("Churn Risk Analysis")
    st.markdown(
        "This model predicts customers who are **unlikely to visit regardless of "
        "treatment** — useful for suppression lists or win-back campaigns that are "
        "separate from the uplift-driven targeting strategy."
    )

    with st.spinner("Training churn model..."):
        churn_model, X_test, y_test, y_prob, churn_metrics = get_churn_pipeline(df, X)

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{churn_metrics['accuracy']*100:.2f}%")
    m2.metric("ROC-AUC", f"{churn_metrics['roc_auc']:.3f}")
    m3.metric("At-risk Rate (test set)", f"{y_test.mean()*100:.2f}%")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn Risk Score Distribution")
        fig_dist = px.histogram(x=y_prob, nbins=30, color_discrete_sequence=['#f472b6'],
                                 labels={'x': 'Predicted Churn Probability'}, template="plotly_dark")
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.subheader("Top Churn Risk Drivers")
        imp_churn = get_top_feature_importance(churn_model, X.columns)
        fig_imp = px.bar(imp_churn.sort_values('importance'), x='importance', y='feature', orientation='h',
                          color_discrete_sequence=['#f472b6'], template="plotly_dark")
        fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("Highest-Risk Customers (test set)")

    risk_df = X_test.copy()
    risk_df['churn_probability'] = y_prob
    risk_df = risk_df.sort_values('churn_probability', ascending=False).head(25).reset_index(drop=True)
    st.dataframe(risk_df.style.format({'churn_probability': '{:.1%}'}), use_container_width=True)

# =========================================================
# 5. Business Simulation
# =========================================================
elif page == "💰 Business Simulation":
    st.header("Strategic ROI Simulation")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Campaign Parameters")
    cost_per_treatment = st.sidebar.slider("Cost per Treatment ($)", 0.0, 5.0, 0.5, 0.1)
    revenue_per_conversion = st.sidebar.slider("Revenue per Visit/Conversion ($)", 0.0, 200.0, 50.0, 5.0)

    roi_df = simulate_business_roi(results, treatment_cost=cost_per_treatment, revenue_per_conversion=revenue_per_conversion)

    optimal_row = roi_df.loc[roi_df['Profit'].idxmax()]

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

    st.download_button(
        label="⬇️ Download Simulation Results (CSV)",
        data=roi_df.to_csv(index=False).encode('utf-8'),
        file_name="uplift_roi_simulation.csv",
        mime="text/csv",
    )
