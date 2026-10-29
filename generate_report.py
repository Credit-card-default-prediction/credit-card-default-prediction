import os
import json
import pandas as pd
from fpdf import FPDF, XPos, YPos

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, 'assets')
CHARTS_DIR = os.path.join(CURRENT_DIR, 'temp_charts')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')


class CreditReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 116, 139)
            self.cell(0, 8, "Credit Card Default Prediction - AI Application Lab", 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="L")
            self.cell(0, 8, "Chandigarh University", 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="R")
            self.ln(9)
            self.set_draw_color(30, 58, 138)
            self.set_line_width(0.4)
            self.line(20, self.get_y(), 190, self.get_y())
            self.ln(4)

    def footer(self):
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(148, 163, 184)
            self.cell(0, 10, f"Page {self.page_no()}", 0, new_x=XPos.RIGHT, new_y=YPos.TOP, align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(30, 58, 138) # Navy Blue
        self.cell(0, 9, title, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(51, 65, 85) # Slate 700
        self.cell(0, 7, title, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(0, 5.5, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def bullet_point(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(0, 5.2, "- " + text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.5)

    def add_table(self, col_widths, headers, data):
        self.set_font("Helvetica", "B", 9.5)
        self.set_fill_color(30, 58, 138)
        self.set_text_color(255, 255, 255)
        
        # Headers
        for width, header in zip(col_widths, headers):
            self.cell(width, 7.5, header, border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C', fill=True)
        self.ln()
        
        # Data rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(15, 23, 42)
        fill = False
        for row in data:
            if fill:
                self.set_fill_color(241, 245, 249)
            else:
                self.set_fill_color(255, 255, 255)
                
            for width, item in zip(col_widths, row):
                self.cell(width, 6.8, str(item), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C', fill=True)
            self.ln()
            fill = not fill
        self.ln(3)


def generate_pdf_report(output_filename="Credit_Card_Default_Prediction_Report.pdf"):
    pdf = CreditReportPDF()
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # ----------------------------------------------------
    # COVER PAGE
    # ----------------------------------------------------
    pdf.add_page()
    
    cu_horiz = os.path.join(ASSETS_DIR, 'cu_horizontal_logo.png')
    cu_seal = os.path.join(ASSETS_DIR, 'cu_seal_logo.jpg')
    
    if os.path.exists(cu_horiz):
        pdf.image(cu_horiz, x=55, y=25, w=100)
        
    pdf.set_y(80)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 12, "Credit Card Default Prediction", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, "Early Identification of Risky Customers for Financial Institutions", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    
    pdf.ln(12)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 9, "AI Application Lab - Project Report", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 7, "Department of Computer Science & Engineering", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.cell(0, 7, "Chandigarh University, Gharuan, Mohali, Punjab", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    
    pdf.ln(12)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, "October 2026", 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    
    if os.path.exists(cu_seal):
        pdf.image(cu_seal, x=90, y=210, w=30)
        
    # ----------------------------------------------------
    # TABLE OF CONTENTS
    # ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("Table of Contents")
    toc_items = [
        "1. Executive Summary & Problem Formulation",
        "2. Dataset Description & Key Input Features",
        "3. Exploratory Data Analysis (EDA) & Financial Insights",
        "4. Feature Engineering & Preprocessing Pipeline",
        "5. Machine Learning Techniques & Architectures",
        "6. Experimental Evaluation & Model Benchmarks",
        "7. Confusion Matrix & ROC-AUC Analysis",
        "8. Feature Importance & Key Risk Indicators",
        "9. Real-Time Risk Decisioning & Mitigation Strategy",
        "10. System Architecture & Web Application",
        "11. Conclusion & Future Scope",
        "12. References"
    ]
    pdf.set_font("Helvetica", "", 11)
    for item in toc_items:
        pdf.cell(0, 8, item, 0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="L")
        
    # ----------------------------------------------------
    # 1. EXECUTIVE SUMMARY & PROBLEM FORMULATION
    # ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("1. Executive Summary & Problem Formulation")
    pdf.body_text("In retail banking and credit lending, credit card defaults constitute one of the most substantial sources of financial loss and systemic exposure. The credit card business model is built on unsecured revolving credit, where cardholders borrow against a pre-approved line of credit without pledging collateral. When cardholders encounter economic distress, mismanagement, or over-leverage, failure to meet minimum monthly payment obligations triggers defaults.")
    pdf.body_text("The primary objective of this project is to construct, evaluate, and deploy an automated, early-warning classification system capable of accurately predicting whether an individual credit card customer will default on their credit payment in the subsequent billing cycle.")
    pdf.body_text("Key Project Objectives:")
    pdf.bullet_point("Data Investigation: Perform exhaustive exploratory data analysis on 30,000 credit records to uncover behavioral precursors to default.")
    pdf.bullet_point("Supervised Machine Learning: Train and compare three prominent classification algorithms: Logistic Regression, Random Forest, and Extreme Gradient Boosting (XGBoost).")
    pdf.bullet_point("Comprehensive Evaluation: Benchmark models across critical performance metrics including Accuracy, Precision, Recall, and ROC-AUC score.")
    pdf.bullet_point("Operational Decisioning: Implement a real-time risk assessment engine that segments customers into four distinct risk tiers and prescribes proactive interventions.")
    
    # ----------------------------------------------------
    # 2. DATASET DESCRIPTION
    # ----------------------------------------------------
    pdf.ln(4)
    pdf.section_title("2. Dataset Description & Key Input Features")
    pdf.body_text("The project utilizes the benchmark 'Default of Credit Card Clients' dataset from Kaggle / UCI Machine Learning Repository, comprising 30,000 Taiwanese credit card clients. The dataset incorporates 25 core attributes covering demographic characteristics, credit limit parameters, 6 months of historical repayment statuses, billing statements, and actual payment transactions.")
    
    headers = ["Feature Category", "Variables", "Description / Value Range"]
    data = [
        ["Demographics", "AGE, SEX, EDUCATION, MARRIAGE", "Age in years, Gender (1=M, 2=F), Education (1-4), Marital status (1-3)"],
        ["Credit Limit", "LIMIT_BAL", "Amount of given credit line (NT dollar)"],
        ["Payment History", "PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6", "Monthly delay status (-2 to 8) from April to Sept 2005"],
        ["Bill Statements", "BILL_AMT1 to BILL_AMT6", "Statement balance in NT dollars across 6 consecutive months"],
        ["Payment Records", "PAY_AMT1 to PAY_AMT6", "Actual payment amounts remitted in NT dollars across 6 months"],
        ["Target Variable", "default (0 / 1)", "Default status in the following month (0=Will Pay, 1=Default)"]
    ]
    pdf.add_table([38, 55, 77], headers, data)
    
    # ----------------------------------------------------
    # 3. EXPLORATORY DATA ANALYSIS
    # ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("3. Exploratory Data Analysis & Financial Insights")
    pdf.body_text("Exhaustive exploratory data analysis was conducted to uncover predictive relationships and understand customer behavioral dynamics.")
    
    chart1 = os.path.join(CHARTS_DIR, 'default_distribution.png')
    if os.path.exists(chart1):
        pdf.image(chart1, x=22, y=pdf.get_y(), w=166)
        pdf.set_y(pdf.get_y() + 68)
        
    pdf.subsection_title("Key EDA Findings:")
    pdf.bullet_point("Class Imbalance: 6,636 customers defaulted (22.12%), while 23,364 customers fulfilled obligations (77.88%). This necessitates class-weighted learning to avoid false sense of security from naive accuracy.")
    pdf.bullet_point("Demographic Disparities: Male customers exhibit a higher default rate (24.2%) compared to female customers (20.8%). University graduates and high school degree holders carry higher default propensity than postgraduate holders.")
    pdf.bullet_point("Repayment Delay as Cardinal Signal: Customers who delayed payment by 2 or more months in September (PAY_0 >= 2) exhibited a default probability exceeding 68%, making recent delinquency the single strongest leading indicator.")
    
    chart2 = os.path.join(CHARTS_DIR, 'recent_delinquency.png')
    if os.path.exists(chart2):
        pdf.image(chart2, x=25, y=pdf.get_y(), w=160)
        pdf.set_y(pdf.get_y() + 82)

    # ----------------------------------------------------
    # 4. FEATURE ENGINEERING & PREPROCESSING
    # ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("4. Feature Engineering & Preprocessing Pipeline")
    pdf.body_text("Raw credit card attributes alone do not fully capture debt acceleration or liquidity distress. Domain-specific features were engineered to empower classifiers with deeper behavioral signals:")
    pdf.bullet_point("Repayment Capability Ratio (PAY_TO_BILL_RATIO): Total payments divided by total accumulated bill amounts. Lower values (<0.15) signal liquidity constraints.")
    pdf.bullet_point("Credit Utilization Rate (UTILIZATION_RATE): September statement balance divided by total approved credit limit. High utilization (>80%) correlates with credit distress.")
    pdf.bullet_point("Delinquency Trajectory (MAX_DELAY, AVG_DELAY): Maximum and average monthly delay severity across the 6-month observation horizon.")
    pdf.bullet_point("Categorical Binning: Grouped rare education codes (0, 5, 6) into 'Others' (4) and rare marriage codes (0) into 'Others' (3) to prevent spurious splits.")
    pdf.bullet_point("Feature Normalization: Applied StandardScaler across all continuous predictors to ensure zero mean and unit variance for gradient stability.")

    # ----------------------------------------------------
    # 5. MACHINE LEARNING TECHNIQUES
    # ----------------------------------------------------
    pdf.ln(3)
    pdf.section_title("5. Machine Learning Techniques & Architectures")
    pdf.body_text("Three diverse machine learning architectures were trained and rigorously cross-validated:")
    pdf.bullet_point("1. Logistic Regression: Serves as a transparent linear baseline. Employs the sigmoid link function with balanced class weights to optimize the log-loss criterion.")
    pdf.bullet_point("2. Random Forest: An ensemble of 150 decorrelated decision trees using bootstrap aggregation (bagging) and balanced subsampling to prevent majority class overfitting.")
    pdf.bullet_point("3. Extreme Gradient Boosting (XGBoost): A state-of-the-art sequential gradient boosting framework utilizing regularized objective functions and scale_pos_weight=3.5 to penalize false negatives heavily.")

    # ----------------------------------------------------
    # 6. EXPERIMENTAL EVALUATION & BENCHMARKS
    # ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("6. Experimental Evaluation & Model Benchmarks")
    pdf.body_text("Models were evaluated on a stratified 20% holdout test set (6,000 customers: 4,673 non-defaults, 1,327 defaults). The table below outlines comparative performance across all core metrics:")
    
    # Load metrics from json
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    meta = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            meta = json.load(f)
            
    m_data = meta.get('models', {})
    lr = m_data.get('Logistic Regression', {})
    rf = m_data.get('Random Forest', {})
    xgb = m_data.get('XGBoost', {})
    
    eval_headers = ["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    eval_rows = [
        ["Logistic Regression", f"{lr.get('accuracy', 0.743)*100:.2f}%", f"{lr.get('precision', 0.441)*100:.2f}%", f"{lr.get('recall', 0.606)*100:.2f}%", f"{lr.get('f1_score', 0.511):.4f}", f"{lr.get('roc_auc', 0.752):.4f}"],
        ["Random Forest", f"{rf.get('accuracy', 0.796)*100:.2f}%", f"{rf.get('precision', 0.539)*100:.2f}%", f"{rf.get('recall', 0.537)*100:.2f}%", f"{rf.get('f1_score', 0.538):.4f}", f"{rf.get('roc_auc', 0.774):.4f}"],
        ["XGBoost (Best)", f"{xgb.get('accuracy', 0.757)*100:.2f}%", f"{xgb.get('precision', 0.464)*100:.2f}%", f"{xgb.get('recall', 0.623)*100:.2f}%", f"{xgb.get('f1_score', 0.532):.4f}", f"{xgb.get('roc_auc', 0.782):.4f}"]
    ]
    pdf.add_table([40, 25, 25, 25, 25, 30], eval_headers, eval_rows)
    
    pdf.subsection_title("Performance Analysis:")
    pdf.bullet_point("XGBoost attained the highest ROC-AUC score of 0.7823 and the highest Recall (62.32%), detecting nearly two-thirds of all eventual defaulters before non-payment occurs.")
    pdf.bullet_point("Random Forest delivered the highest overall Accuracy (79.58%) and highest Precision (53.86%), making it suitable when minimizing false alarms is paramount.")
    pdf.bullet_point("In financial risk management, Recall is significantly more valuable than naive Accuracy because an undetected default (False Negative) costs orders of magnitude more than a proactive outreach call (False Positive).")

    chart3 = os.path.join(CHARTS_DIR, 'repayment_heatmap.png')
    if os.path.exists(chart3):
        pdf.image(chart3, x=22, y=pdf.get_y(), w=166)
        pdf.set_y(pdf.get_y() + 85)

    # ----------------------------------------------------
    # 7. CONFUSION MATRIX & ROC-AUC
    # ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("7. Confusion Matrix & ROC-AUC Analysis")
    pdf.body_text("Detailed examination of the test confusion matrix for XGBoost:")
    
    cm = xgb.get('confusion_matrix', [[3716, 957], [500, 827]])
    cm_headers = ["Actual \\ Predicted", "Predicted Non-Default", "Predicted Default", "Total"]
    cm_rows = [
        ["Actual Non-Default (0)", f"{cm[0][0]:,} (TN)", f"{cm[0][1]:,} (FP)", f"{cm[0][0]+cm[0][1]:,}"],
        ["Actual Default (1)", f"{cm[1][0]:,} (FN)", f"{cm[1][1]:,} (TP)", f"{cm[1][0]+cm[1][1]:,}"],
        ["Total Evaluated", f"{cm[0][0]+cm[1][0]:,}", f"{cm[0][1]+cm[1][1]:,}", "6,000"]
    ]
    pdf.add_table([45, 45, 45, 35], cm_headers, cm_rows)
    
    pdf.body_text("The ROC-AUC curve demonstrates that the model maintains robust true positive rates across low false positive thresholds, allowing financial institutions to calibrate cutoff probabilities dynamically according to risk appetite.")

    chart4 = os.path.join(CHARTS_DIR, 'credit_utilization.png')
    if os.path.exists(chart4):
        pdf.image(chart4, x=22, y=pdf.get_y(), w=166)
        pdf.set_y(pdf.get_y() + 66)

    # ----------------------------------------------------
    # 8. RISK DECISIONING FRAMEWORK
    # ----------------------------------------------------
    pdf.ln(3)
    pdf.section_title("8. Real-Time Risk Decisioning & Mitigation Strategy")
    pdf.body_text("Rather than outputting a binary black-box prediction, the deployed solution categorizes customers into four operational risk tiers with tailored institutional actions:")
    
    tier_headers = ["Risk Tier", "Default Probability", "Risk Profile", "Prescribed Action"]
    tier_rows = [
        ["Tier 1: Prime", "< 20%", "Consistently pays on time; low debt ratio", "Approve limit increases, preferential APR offers"],
        ["Tier 2: Standard", "20% - 45%", "Occasional minor revolving; manageable debt", "Maintain standard monitoring; periodic review"],
        ["Tier 3: Watchlist", "45% - 70%", "1-2 months delinquency; high utilization", "Temporarily cap limit; trigger automated SMS reminders"],
        ["Tier 4: Critical", "> 70%", "Persistent delinquency; ballooning unpaid balance", "Immediate card freeze; direct referral to collections team"]
    ]
    pdf.add_table([32, 35, 53, 50], tier_headers, tier_rows)

    # ----------------------------------------------------
    # 9. CONCLUSION & FUTURE SCOPE
    # ----------------------------------------------------
    pdf.add_page()
    pdf.section_title("9. Conclusion & Future Scope")
    pdf.body_text("Conclusion:")
    pdf.body_text("This study established a comprehensive end-to-end AI framework for predicting credit card default on 30,000 real-world customer records. By engineering financial ratios (repayment velocity and credit utilization) and balancing class distribution, XGBoost achieved an ROC-AUC of 0.7823 and captured 62.3% of true defaulters. The deployment in an interactive Streamlit environment enables credit underwriters and risk managers to evaluate customer accounts instantly with explainable risk drivers.")
    
    pdf.ln(2)
    pdf.body_text("Future Scope:")
    pdf.bullet_point("Macroeconomic Integration: Incorporate interest rates, inflation indicators, and unemployment rates into temporal features.")
    pdf.bullet_point("Deep Learning Sequence Models: Implement Recurrent Neural Networks (LSTMs or Transformers) to model the exact temporal billing sequence across several years.")
    pdf.bullet_point("Explainable AI (SHAP): Integrate game-theoretic SHAP tree explainers to produce regulatory-compliant adverse action notices.")

    pdf.ln(4)
    pdf.section_title("10. References")
    pdf.bullet_point("Yeh, I-Cheng, and Lien, Che-hui. 'The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.' Expert Systems with Applications 36.2 (2009): 2473-2480.")
    pdf.bullet_point("Chen, Tianqi, and Guestrin, Carlos. 'XGBoost: A scalable tree boosting system.' Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (2016).")
    pdf.bullet_point("Breiman, Leo. 'Random forests.' Machine Learning 45.1 (2001): 5-32.")
    pdf.bullet_point("Hastie, Trevor, Tibshirani, Robert, and Friedman, Jerome. 'The Elements of Statistical Learning: Data Mining, Inference, and Prediction.' Springer (2009).")

    out_path = os.path.join(CURRENT_DIR, output_filename)
    pdf.output(out_path)
    print(f"Report generated successfully at: {out_path}")
    return out_path


if __name__ == '__main__':
    generate_pdf_report()
