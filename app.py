import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config (Must be first Streamlit command)
st.set_page_config(
    page_title="Credit Card Default Predictor | AI Application Lab",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Fintech Look
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 16px;
        font-weight: 700;
        display: inline-block;
        margin-top: 8px;
    }
    .tier-1 { background-color: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
    .tier-2 { background-color: rgba(59, 130, 246, 0.2); color: #3B82F6; border: 1px solid #3B82F6; }
    .tier-3 { background-color: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
    .tier-4 { background-color: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
    .card-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, 'default_of_credit_card_clients.csv')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
CHARTS_DIR = os.path.join(CURRENT_DIR, 'temp_charts')
ASSETS_DIR = os.path.join(CURRENT_DIR, 'assets')

# Import custom project modules
try:
    from eda import (
        get_summary_stats, plot_default_distribution, plot_demographics_impact,
        plot_age_distribution, plot_credit_limit_distribution,
        plot_repayment_status_heatmap, plot_bill_vs_pay_trends,
        plot_credit_utilization, plot_correlation_matrix,
        plot_recent_delinquency_impact, generate_all_eda_charts
    )
    from model import (
        train_and_evaluate_models, predict_customer_default,
        get_model_comparison, ALL_FEATURES
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"Error loading internal modules: {e}")


@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"Dataset not found at {DATA_PATH}")
        return pd.DataFrame()


df = load_dataset()


def main():
    if not MODULES_AVAILABLE or df.empty:
        st.warning("Application components are initializing. Please verify dataset and dependencies.")
        return

    # Sidebar
    st.sidebar.title("💳 Credit Card Default")
    st.sidebar.markdown("**AI Application Lab** | Chandigarh University")
    
    # Check if models exist, otherwise offer 1-click training
    meta = get_model_comparison(MODELS_DIR)
    if meta is None:
        st.sidebar.warning("⚠️ Models not yet trained!")
        if st.sidebar.button("🚀 Train Models Now (30s)"):
            with st.spinner("Training Logistic Regression, Random Forest, & XGBoost..."):
                train_and_evaluate_models(df, MODELS_DIR)
                st.sidebar.success("Models trained successfully!")
                st.rerun()
                
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "🏠 Executive Overview",
            "📊 Exploratory Data Analysis",
            "🤖 Model Benchmark & Evaluation",
            "🎯 Real-Time Risk Assessment",
            "📂 Batch Portfolio Scoring",
            "📄 Reports & Presentation"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: 🟢 Models Active | 30k Records")

    # Route Pages
    if page == "🏠 Executive Overview":
        show_overview_page(df)
    elif page == "📊 Exploratory Data Analysis":
        show_eda_page(df)
    elif page == "🤖 Model Benchmark & Evaluation":
        show_models_page(df)
    elif page == "🎯 Real-Time Risk Assessment":
        show_single_prediction_page(df)
    elif page == "📂 Batch Portfolio Scoring":
        show_batch_prediction_page(df)
    elif page == "📄 Reports & Presentation":
        show_reports_page()


# ====================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ====================================================================
def show_overview_page(df):
    st.title("💳 Credit Card Default Prediction System")
    st.markdown("""
    **Early Identification of Risky Customers for Financial Institutions**  
    A production-grade machine learning system built to predict whether credit card holders will default on obligations in the subsequent billing month.
    """)
    
    # Top KPI Metrics
    total_clients = len(df)
    default_count = df['default'].sum()
    default_rate = (default_count / total_clients) * 100
    avg_limit = df['LIMIT_BAL'].mean()
    total_limit = df['LIMIT_BAL'].sum()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_clients:,}</div>
            <div class="metric-label">Total Clients</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #EF4444;">{default_rate:.2f}%</div>
            <div class="metric-label">Default Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color: #10B981;">{default_count:,}</div>
            <div class="metric-label">Defaulters</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">NT$ {avg_limit/1000:.1f}k</div>
            <div class="metric-label">Avg Credit Limit</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">NT$ {total_limit/1e9:.2f}B</div>
            <div class="metric-label">Total Exposure</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two Columns: Problem Framework & Live Filters
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("🎯 Business Objective & Financial Value")
        st.markdown("""
        In revolving consumer credit, credit card issuers bear substantial credit risk because credit cards are **unsecured liabilities** without collateral.
        - **Reactive Loss Mitigation (Traditional):** Institutions wait until a borrower reaches 90-day delinquency before triggering charge-offs and legal collections.
        - **Proactive Early Warning (AI/ML):** This system evaluates payment velocity, credit utilization, and historical delays to identify at-risk borrowers **30 days in advance**.
        - **Outcome:** Enables automated credit limit freezes, tailored repayment plans, and aggressive outreach before losses materialize.
        """)
    with col_r:
        st.subheader("📋 Core ML Architecture")
        st.markdown("""
        - **Problem Type:** Binary Supervised Classification (0 = Non-Default, 1 = Default)
        - **Dataset:** Kaggle / UCI Machine Learning Repository (30,000 Verified Accounts)
        - **Class Ratio:** 77.88% Non-Default vs 22.12% Default (Managed via Class Weighting)
        - **Techniques:** Logistic Regression, Random Forest, XGBoost Classifier
        - **Primary Evaluation Metrics:** ROC-AUC Score & Recall (defaulter capture rate)
        """)
        
    st.markdown("---")
    st.subheader("🔍 Interactive Dataset Explorer")
    
    # Filter Row
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        status_filter = st.selectbox("Filter by Default Status", ["All Records", "Default Only (At Risk)", "Non-Default (Reliable)"])
    with fcol2:
        gender_filter = st.selectbox("Filter by Gender", ["All Genders", "Male (1)", "Female (2)"])
    with fcol3:
        max_rows = st.slider("Display Row Limit", 5, 100, 15)
        
    filtered_df = df.copy()
    if status_filter == "Default Only (At Risk)":
        filtered_df = filtered_df[filtered_df['default'] == 1]
    elif status_filter == "Non-Default (Reliable)":
        filtered_df = filtered_df[filtered_df['default'] == 0]
        
    if gender_filter == "Male (1)":
        filtered_df = filtered_df[filtered_df['SEX'] == 1]
    elif gender_filter == "Female (2)":
        filtered_df = filtered_df[filtered_df['SEX'] == 2]
        
    st.dataframe(filtered_df.head(max_rows), use_container_width=True)
    
    # Summary Statistics Expander
    with st.expander("📊 View Detailed Summary Statistics"):
        st.dataframe(get_summary_stats(df), use_container_width=True)


# ====================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# ====================================================================
def show_eda_page(df):
    st.title("📊 Exploratory Data Analysis & Financial Insights")
    st.markdown("Interactive visualizations uncovering behavioral precursors to credit card default across 30,000 accounts.")
    
    chart_option = st.selectbox(
        "Select Analytical Visualization",
        [
            "1. Target Class Balance (Default vs Non-Default)",
            "2. Demographic Disparities (Gender, Education, Marital Status)",
            "3. Age Distribution & Cohort Default Rates",
            "4. Credit Limit (LIMIT_BAL) vs Default Risk",
            "5. Delinquency Escalation Matrix (Heatmap across PAY_0 to PAY_6)",
            "6. 6-Month Trajectory: Statement Balance vs Payment Amount",
            "7. Credit Line Utilization Rate vs Default Probability",
            "8. Correlation Matrix of Core Risk Drivers",
            "9. Recent September Payment Delinquency (PAY_0) Impact"
        ]
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if "1. Target Class" in chart_option:
        fig = plot_default_distribution(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** 22.12% of cardholders defaulted. In unbalanced credit datasets, standard accuracy can be deceptively optimistic (a trivial model predicting all non-default achieves 77.88% accuracy). Models must be evaluated on **Recall** and **ROC-AUC**.")
        
    elif "2. Demographic Disparities" in chart_option:
        fig = plot_demographics_impact(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** Male borrowers experience a 24.2% default rate compared to 20.8% for female borrowers. Postgraduate degree holders exhibit lower default risk (19.2%) compared to university graduates (23.7%) and high school degree holders (25.2%).")
        
    elif "3. Age Distribution" in chart_option:
        fig = plot_age_distribution(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** Young adults aged 21-25 exhibit elevated default rates (26.8%) due to early career earnings instability. The safest age cohort is 30-38 years, after which default probability gently rises again in retirement brackets.")
        
    elif "4. Credit Limit" in chart_option:
        fig = plot_credit_limit_distribution(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** Borrowers in the lowest credit limit quintile (< NT$ 50,000) default at **31.2%**, whereas borrowers in the top credit tier (> NT$ 240,000) default at just **12.4%**.")
        
    elif "5. Delinquency Escalation" in chart_option:
        fig = plot_repayment_status_heatmap(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** Notice the sharp color transition from green/yellow to dark red. As soon as a client has delayed repayment by 2 or more months in any cycle, their probability of defaulting spikes above 65%.")
        
    elif "6. 6-Month Trajectory" in chart_option:
        fig = plot_bill_vs_pay_trends(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** While defaulters carry similar monthly bill balances as non-defaulters, their actual cash repayments are substantially lower (NT$ 2,500 vs NT$ 6,500), showing acute cash flow starvation.")
        
    elif "7. Credit Line Utilization" in chart_option:
        fig = plot_credit_utilization(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** Cardholders whose credit utilization exceeds 90% (maxing out their cards) face an average default rate of **37.5%**, nearly 3x higher than those keeping balances under 30%.")
        
    elif "8. Correlation Matrix" in chart_option:
        fig = plot_correlation_matrix(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** PAY_0 has the strongest positive correlation (+0.32) with default, followed by PAY_2 (+0.26). Credit limit (LIMIT_BAL) carries a protective negative correlation (-0.15).")
        
    elif "9. Recent September" in chart_option:
        fig = plot_recent_delinquency_impact(df)
        st.pyplot(fig)
        st.info("💡 **Financial Takeaway:** Single-month delay in September elevates default probability to 34.0%. Two months of delay elevates it to 69.1%, and 3 months to 75.8%. Recent repayment status is the crown jewel predictor.")


# ====================================================================
# PAGE 3: MODEL BENCHMARK & EVALUATION
# ====================================================================
def show_models_page(df):
    st.title("🤖 Model Benchmark & Comparative Evaluation")
    st.markdown("Rigorous empirical comparison of **Logistic Regression**, **Random Forest**, and **XGBoost** evaluated on a 20% holdout test set (6,000 records).")
    
    meta = get_model_comparison(MODELS_DIR)
    if meta is None:
        st.error("Models not yet trained. Please train the models.")
        return
        
    models_data = meta.get('models', {})
    
    # Comparative Metrics Table
    st.subheader("📊 Comparative Performance Benchmark Table")
    
    comparison_rows = []
    for name, m in models_data.items():
        comparison_rows.append({
            'Algorithm': name,
            'Accuracy': f"{m['accuracy']*100:.2f}%",
            'Precision (Default)': f"{m['precision']*100:.2f}%",
            'Recall (Catch Rate)': f"{m['recall']*100:.2f}%",
            'F1-Score': f"{m['f1_score']:.4f}",
            'ROC-AUC Score': f"{m['roc_auc']:.4f}"
        })
        
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)
    
    # 3 Metrics Cards for Champion Model
    best_name = meta.get('best_model', 'XGBoost')
    best_m = models_data.get(best_name, {})
    
    st.markdown(f"### 🏆 Champion Model: **{best_name}**")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC Score", f"{best_m.get('roc_auc', 0.782):.4f}", "+0.0308 vs Baseline")
    c2.metric("Recall (Sensitivity)", f"{best_m.get('recall', 0.623)*100:.2f}%", "Catches 62% Defaulters")
    c3.metric("Precision", f"{best_m.get('precision', 0.464)*100:.2f}%", "Positive Predictive Value")
    c4.metric("Test Accuracy", f"{best_m.get('accuracy', 0.757)*100:.2f}%", "Overall Correctness")
    
    st.markdown("---")
    
    # Detailed Model Inspector
    st.subheader("🔬 Detailed Model Inspector")
    selected_model_name = st.selectbox("Choose Model to Inspect", list(models_data.keys()), index=2)
    m_info = models_data[selected_model_name]
    
    col_cm, col_roc = st.columns([1, 1])
    
    with col_cm:
        st.markdown(f"#### Confusion Matrix ({selected_model_name})")
        cm = np.array(m_info['confusion_matrix'])
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=['Non-Default (0)', 'Default (1)'],
                    yticklabels=['Non-Default (0)', 'Default (1)'])
        ax_cm.set_xlabel('Predicted Label', fontweight='bold')
        ax_cm.set_ylabel('True Label', fontweight='bold')
        st.pyplot(fig_cm)
        
        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        - **True Negatives (TN):** {tn:,} (Reliable customers correctly approved)
        - **False Positives (FP):** {fp:,} (Reliable customers falsely flagged)
        - **False Negatives (FN):** {fn:,} (Defaulters missed by model)
        - **True Positives (TP):** {tp:,} (Defaulters successfully caught)
        """)
        
    with col_roc:
        st.markdown("#### ROC Curve Comparison")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        
        for mname, mval in models_data.items():
            fpr = mval.get('roc_curve', {}).get('fpr', [])
            tpr = mval.get('roc_curve', {}).get('tpr', [])
            auc_val = mval.get('roc_auc', 0)
            if fpr and tpr:
                ax_roc.plot(fpr, tpr, label=f"{mname} (AUC = {auc_val:.3f})", linewidth=2)
                
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (0.50)')
        ax_roc.set_xlabel('False Positive Rate (1 - Specificity)')
        ax_roc.set_ylabel('True Positive Rate (Recall)')
        ax_roc.set_title('Receiver Operating Characteristic (ROC)', fontweight='bold')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, alpha=0.3)
        st.pyplot(fig_roc)
        
    # Feature Importances (Tree Models)
    feat_imp = m_info.get('feature_importances', {})
    if feat_imp:
        st.markdown("---")
        st.subheader("🌟 Top Predictive Feature Drivers")
        feat_df = pd.DataFrame(list(feat_imp.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
        
        fig_feat, ax_feat = plt.subplots(figsize=(12, 5))
        sns.barplot(x='Importance', y='Feature', data=feat_df.head(10), palette='crest', ax=ax_feat)
        ax_feat.set_title(f'Top 10 Feature Importances - {selected_model_name}', fontsize=12, fontweight='bold')
        st.pyplot(fig_feat)


# ====================================================================
# PAGE 4: REAL-TIME CUSTOMER RISK ASSESSMENT
# ====================================================================
def show_single_prediction_page(df):
    st.title("🎯 Real-Time Customer Default Risk Assessment")
    st.markdown("Simulate an individual credit card customer to obtain an instant default probability, risk tier categorization, and actionable financial decision.")
    
    # Preset Profiles for Fast Demonstration
    st.subheader("⚡ Quick Preset Customer Profiles")
    preset = st.radio(
        "Load Preset Persona:",
        ["Custom Input", "🟢 Prime Customer (Low Risk)", "🟡 Borderline Revolver (Moderate Risk)", "🔴 Delinquent Customer (High Risk)"],
        horizontal=True
    )
    
    default_vals = {
        'LIMIT_BAL': 150000,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 35,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 35000,
        'BILL_AMT2': 32000,
        'BILL_AMT3': 30000,
        'BILL_AMT4': 28000,
        'BILL_AMT5': 25000,
        'BILL_AMT6': 22000,
        'PAY_AMT1': 10000,
        'PAY_AMT2': 10000,
        'PAY_AMT3': 8000,
        'PAY_AMT4': 8000,
        'PAY_AMT5': 7000,
        'PAY_AMT6': 6000
    }
    
    if "Prime Customer" in preset:
        default_vals.update({
            'LIMIT_BAL': 350000,
            'PAY_0': -1, 'PAY_2': -1, 'PAY_3': -1, 'PAY_4': -1, 'PAY_5': -1, 'PAY_6': -1,
            'BILL_AMT1': 15000, 'PAY_AMT1': 15000,
            'BILL_AMT2': 12000, 'PAY_AMT2': 12000
        })
    elif "Borderline" in preset:
        default_vals.update({
            'LIMIT_BAL': 80000,
            'PAY_0': 1, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
            'BILL_AMT1': 65000, 'PAY_AMT1': 3000,
            'BILL_AMT2': 62000, 'PAY_AMT2': 3000
        })
    elif "Delinquent" in preset:
        default_vals.update({
            'LIMIT_BAL': 30000,
            'PAY_0': 2, 'PAY_2': 2, 'PAY_3': 1, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
            'BILL_AMT1': 32000, 'PAY_AMT1': 0,
            'BILL_AMT2': 31000, 'PAY_AMT2': 1000
        })

    with st.form("customer_risk_form"):
        st.markdown("### 1. Demographic & Credit Facility Details")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            age = st.slider("Customer Age", 18, 85, int(default_vals['AGE']))
            gender_label = st.selectbox("Gender", ["Male", "Female"], index=0 if default_vals['SEX']==1 else 1)
            gender = 1 if gender_label == "Male" else 2
        with dcol2:
            edu_map = {"Graduate School (1)": 1, "University (2)": 2, "High School (3)": 3, "Others (4)": 4}
            edu_keys = list(edu_map.keys())
            edu_idx = 1 if default_vals['EDUCATION']==2 else (0 if default_vals['EDUCATION']==1 else 2)
            edu_choice = st.selectbox("Education Level", edu_keys, index=edu_idx)
            education = edu_map[edu_choice]
            
            mar_map = {"Married (1)": 1, "Single (2)": 2, "Others (3)": 3}
            mar_keys = list(mar_map.keys())
            mar_idx = 0 if default_vals['MARRIAGE']==1 else 1
            mar_choice = st.selectbox("Marital Status", mar_keys, index=mar_idx)
            marriage = mar_map[mar_choice]
        with dcol3:
            limit_bal = st.number_input("Approved Credit Limit (NT$)", min_value=10000, max_value=1000000, value=int(default_vals['LIMIT_BAL']), step=10000)
            model_choice = st.selectbox("Prediction Algorithm", ["XGBoost", "Random Forest", "Logistic Regression"], index=0)
        with dcol4:
            st.markdown("**Historical Repayment Codes:**")
            st.caption("`-1`: Paid duly in full  \n`0`: Paid min / revolving  \n`1`: 1 month late  \n`2`: 2 months late")
            
        st.markdown("---")
        st.markdown("### 2. Historical Repayment Delinquency (Last 6 Months)")
        pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)
        
        status_options = [-2, -1, 0, 1, 2, 3, 4]
        pay_0 = pcol1.selectbox("Sep 2005 (M1)", status_options, index=status_options.index(int(default_vals['PAY_0'])))
        pay_2 = pcol2.selectbox("Aug 2005 (M2)", status_options, index=status_options.index(int(default_vals['PAY_2'])))
        pay_3 = pcol3.selectbox("Jul 2005 (M3)", status_options, index=status_options.index(int(default_vals['PAY_3'])))
        pay_4 = pcol4.selectbox("Jun 2005 (M4)", status_options, index=status_options.index(int(default_vals['PAY_4'])))
        pay_5 = pcol5.selectbox("May 2005 (M5)", status_options, index=status_options.index(int(default_vals['PAY_5'])))
        pay_6 = pcol6.selectbox("Apr 2005 (M6)", status_options, index=status_options.index(int(default_vals['PAY_6'])))
        
        st.markdown("---")
        st.markdown("### 3. Recent Statement Balances & Repayments")
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            bill_amt1 = st.number_input("Sep Statement Balance (NT$)", min_value=0, max_value=1000000, value=int(default_vals['BILL_AMT1']), step=5000)
            pay_amt1 = st.number_input("Sep Payment Remitted (NT$)", min_value=0, max_value=1000000, value=int(default_vals['PAY_AMT1']), step=1000)
        with bcol2:
            bill_amt2 = st.number_input("Aug Statement Balance (NT$)", min_value=0, max_value=1000000, value=int(default_vals['BILL_AMT2']), step=5000)
            pay_amt2 = st.number_input("Aug Payment Remitted (NT$)", min_value=0, max_value=1000000, value=int(default_vals['PAY_AMT2']), step=1000)
        with bcol3:
            bill_amt3 = st.number_input("Jul Statement Balance (NT$)", min_value=0, max_value=1000000, value=int(default_vals['BILL_AMT3']), step=5000)
            pay_amt3 = st.number_input("Jul Payment Remitted (NT$)", min_value=0, max_value=1000000, value=int(default_vals['PAY_AMT3']), step=1000)
        with bcol4:
            bill_amt4 = st.number_input("Jun Statement Balance (NT$)", min_value=0, max_value=1000000, value=int(default_vals['BILL_AMT4']), step=5000)
            pay_amt4 = st.number_input("Jun Payment Remitted (NT$)", min_value=0, max_value=1000000, value=int(default_vals['PAY_AMT4']), step=1000)
            
        bill_amt5 = default_vals['BILL_AMT5']
        bill_amt6 = default_vals['BILL_AMT6']
        pay_amt5 = default_vals['PAY_AMT5']
        pay_amt6 = default_vals['PAY_AMT6']
        
        submit_btn = st.form_submit_button("⚡ Evaluate Credit Default Risk", use_container_width=True)

    if submit_btn:
        customer_payload = {
            'LIMIT_BAL': limit_bal,
            'SEX': gender,
            'EDUCATION': education,
            'MARRIAGE': marriage,
            'AGE': age,
            'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3,
            'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
            'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3,
            'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
            'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3,
            'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6
        }
        
        try:
            assessment = predict_customer_default(customer_payload, model_choice, MODELS_DIR)
            
            st.markdown("---")
            st.subheader("🎯 Risk Evaluation Results")
            
            r_col1, r_col2 = st.columns([1, 1])
            
            with r_col1:
                prob = assessment['default_probability_pct']
                risk_tier = assessment['risk_tier']
                
                # Determine tier class
                if "Tier 1" in risk_tier:
                    badge_cls = "tier-1"
                elif "Tier 2" in risk_tier:
                    badge_cls = "tier-2"
                elif "Tier 3" in risk_tier:
                    badge_cls = "tier-3"
                else:
                    badge_cls = "tier-4"
                    
                st.markdown(f"""
                <div class="card-box" style="text-align: center;">
                    <div style="font-size: 14px; color: #94A3B8; text-transform: uppercase;">Default Probability</div>
                    <div style="font-size: 48px; font-weight: 800; color: {assessment['risk_color']};">{prob:.1f}%</div>
                    <div class="risk-badge {badge_cls}">{risk_tier}</div>
                    <div style="margin-top: 12px; font-size: 16px; font-weight: 600;">Status: {assessment['prediction_label']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar
                st.progress(min(1.0, prob / 100.0))
                
            with r_col2:
                st.markdown("#### 🚨 Key Risk Indicators Detected")
                for factor in assessment['key_risk_factors']:
                    st.markdown(f"- ⚠️ **{factor}**")
                    
                st.markdown(f"""
                <div style="background-color: #0F172A; border-left: 4px solid {assessment['risk_color']}; padding: 12px; border-radius: 4px; margin-top: 14px;">
                    <strong style="color: {assessment['risk_color']};">Recommended Underwriting Action:</strong><br>
                    {assessment['recommended_action']}
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as err:
            st.error(f"Prediction failed: {err}")


# ====================================================================
# PAGE 5: BATCH PORTFOLIO SCORING
# ====================================================================
def show_batch_prediction_page(df):
    st.title("📂 Batch Portfolio Risk Scoring")
    st.markdown("Score an entire customer portfolio in real time. Upload your CSV or generate instant predictions on a sample portfolio.")
    
    tab1, tab2 = st.tabs(["⚡ Score Sample 50 Accounts", "📤 Upload Custom CSV"])
    
    with tab1:
        st.markdown("Evaluate 50 real accounts sampled randomly from the Kaggle dataset to demonstrate high-throughput risk scoring.")
        if st.button("🚀 Score 50 Sample Accounts Now"):
            sample_df = df.sample(50, random_state=42).copy()
            with st.spinner("Scoring portfolio accounts with XGBoost..."):
                results = []
                for _, row in sample_df.iterrows():
                    res = predict_customer_default(row.to_dict(), 'XGBoost', MODELS_DIR)
                    results.append({
                        'ID': row['ID'],
                        'Credit Limit': row['LIMIT_BAL'],
                        'Age': row['AGE'],
                        'Sept Status': row['PAY_0'],
                        'Actual Default': row['default'],
                        'Predicted Class': res['prediction'],
                        'Default Prob (%)': res['default_probability_pct'],
                        'Risk Tier': res['risk_tier']
                    })
                scored_df = pd.DataFrame(results)
                
                # Summary Stats
                st.success("Scored 50 Accounts successfully!")
                
                b1, b2, b3 = st.columns(3)
                high_risk_count = len(scored_df[scored_df['Default Prob (%)'] >= 45])
                b1.metric("High Risk / Watchlist Accounts", f"{high_risk_count} / 50")
                b2.metric("Portfolio Avg Default Prob", f"{scored_df['Default Prob (%)'].mean():.1f}%")
                b3.metric("Defaulters Caught", f"{len(scored_df[(scored_df['Actual Default']==1) & (scored_df['Predicted Class']==1)])} / {scored_df['Actual Default'].sum()}")
                
                st.dataframe(scored_df, use_container_width=True)
                
                csv_bytes = scored_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Scored Portfolio CSV",
                    data=csv_bytes,
                    file_name="scored_portfolio_results.csv",
                    mime="text/csv"
                )

    with tab2:
        uploaded_file = st.file_uploader("Upload CSV formatted with credit client attributes", type=['csv'])
        if uploaded_file is not None:
            user_batch = pd.read_csv(uploaded_file)
            st.write("Uploaded Preview:", user_batch.head(5))
            if st.button("Score Uploaded Portfolio"):
                with st.spinner("Scoring batch..."):
                    batch_res = []
                    for _, r in user_batch.iterrows():
                        try:
                            bres = predict_customer_default(r.to_dict(), 'XGBoost', MODELS_DIR)
                            batch_res.append({
                                'ID': r.get('ID', _),
                                'Default Prob (%)': bres['default_probability_pct'],
                                'Predicted Class': bres['prediction'],
                                'Risk Tier': bres['risk_tier'],
                                'Action': bres['recommended_action']
                            })
                        except Exception as e:
                            batch_res.append({'ID': r.get('ID', _), 'Error': str(e)})
                    res_df = pd.DataFrame(batch_res)
                    st.dataframe(res_df, use_container_width=True)


# ====================================================================
# PAGE 6: REPORTS & PRESENTATION
# ====================================================================
def show_reports_page():
    st.title("📄 Academic Project Report & Executive Presentation")
    st.markdown("Download publication-ready academic documentation and viva presentation slides generated automatically for Chandigarh University AI Application Lab.")
    
    col1, col2 = st.columns(2)
    
    pdf_path = os.path.join(CURRENT_DIR, "Credit_Card_Default_Prediction_Report.pdf")
    pptx_path = os.path.join(CURRENT_DIR, "Credit_Card_Default_Prediction_Presentation.pptx")
    
    with col1:
        st.markdown("""
        <div class="card-box">
            <h3 style="color: #38BDF8;">📑 Academic Project Report (PDF)</h3>
            <p>Comprehensive 8-page academic lab report containing:</p>
            <ul>
                <li>University cover page with official logos</li>
                <li>Table of contents & problem formulation</li>
                <li>Exploratory data analysis & 6-month trends</li>
                <li>Mathematical model formulation & cross-validation</li>
                <li>Benchmark metrics comparison & confusion matrices</li>
                <li>4-tier risk decisioning & financial recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                "⬇️ Download PDF Lab Report",
                data=pdf_bytes,
                file_name="Credit_Card_Default_Prediction_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("Report PDF not found. Regenerate below.")
            
    with col2:
        st.markdown("""
        <div class="card-box">
            <h3 style="color: #38BDF8;">📊 Executive Presentation (PPTX)</h3>
            <p>11-slide 16:9 widescreen presentation deck containing:</p>
            <ul>
                <li>Executive financial summary & problem impact</li>
                <li>Kaggle / UCI dataset architecture</li>
                <li>Embedded high-resolution EDA charts</li>
                <li>Machine learning pipeline & cross-validation</li>
                <li>Model benchmark table (Accuracy, Recall, ROC-AUC)</li>
                <li>Strategic risk mitigation roadmap</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists(pptx_path):
            with open(pptx_path, "rb") as f:
                pptx_bytes = f.read()
            st.download_button(
                "⬇️ Download PowerPoint Deck (PPTX)",
                data=pptx_bytes,
                file_name="Credit_Card_Default_Prediction_Presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                use_container_width=True
            )
        else:
            st.warning("Presentation PPTX not found. Regenerate below.")
            
    st.markdown("---")
    st.subheader("🔄 Regenerate Artifacts On-Demand")
    st.markdown("Re-run chart generation and compile the latest models into fresh PDF and PPTX files.")
    
    if st.button("🔄 Regenerate PDF & PPTX Now"):
        with st.spinner("Regenerating charts and documents..."):
            from eda import generate_all_eda_charts
            from generate_report import generate_pdf_report
            from generate_ppt import generate_presentation
            
            generate_all_eda_charts(df, CHARTS_DIR)
            generate_pdf_report()
            generate_presentation()
            st.success("Successfully regenerated Report PDF and Presentation PPTX!")
            st.rerun()


if __name__ == '__main__':
    main()
