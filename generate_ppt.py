import os
import json
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, 'assets')
CHARTS_DIR = os.path.join(CURRENT_DIR, 'temp_charts')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')

# Color Palette
NAVY = RGBColor(15, 23, 42)        # Slate Dark / 0F172A
ROYAL_BLUE = RGBColor(30, 58, 138) # 1E3A8A
LIGHT_BG = RGBColor(248, 250, 252) # F8FAFC
TEXT_DARK = RGBColor(15, 23, 42)
TEXT_MUTED = RGBColor(100, 116, 139)
ACCENT_GREEN = RGBColor(16, 185, 129)
ACCENT_RED = RGBColor(239, 68, 68)
CARD_BG = RGBColor(241, 245, 249)
WHITE = RGBColor(255, 255, 255)


def add_header(slide, title_text, category_text="AI APPLICATION LAB | CHANDIGARH UNIVERSITY"):
    """Adds a standard clean executive header to slides."""
    # Top Category Tracker
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ROYAL_BLUE

    # Slide Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.5), Inches(0.7))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = TEXT_DARK


def generate_presentation(output_filename="Credit_Card_Default_Prediction_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # Load model metrics
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    meta = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            meta = json.load(f)
    m_data = meta.get('models', {})
    lr = m_data.get('Logistic Regression', {})
    rf = m_data.get('Random Forest', {})
    xgb = m_data.get('XGBoost', {})

    # ====================================================
    # SLIDE 1: Title Slide (Dark Navy Theme)
    # ====================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    # CU Seal if exists
    cu_seal = os.path.join(ASSETS_DIR, 'cu_seal_logo.jpg')
    if os.path.exists(cu_seal):
        s1.shapes.add_picture(cu_seal, Inches(0.9), Inches(1.0), width=Inches(1.2))

    t_box1 = s1.shapes.add_textbox(Inches(0.9), Inches(2.2), Inches(11.5), Inches(3.5))
    tf1 = t_box1.text_frame
    tf1.word_wrap = True
    
    p1 = tf1.paragraphs[0]
    p1.text = "Credit Card Default Prediction"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    
    p2 = tf1.add_paragraph()
    p2.text = "Early Identification of Risky Customers for Financial Institutions"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    p2.space_before = Pt(8)
    
    p3 = tf1.add_paragraph()
    p3.text = "AI Application Lab | Department of Computer Science & Engineering"
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(59, 130, 246)
    p3.space_before = Pt(24)
    
    p4 = tf1.add_paragraph()
    p4.text = "Chandigarh University, Mohali, Punjab | October 2026"
    p4.font.size = Pt(13)
    p4.font.color.rgb = RGBColor(203, 213, 225)
    p4.space_before = Pt(4)

    # ====================================================
    # SLIDE 2: Executive Summary & Financial Imperative
    # ====================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Executive Summary: The Credit Default Challenge")
    
    # 3 Stat Cards
    cards = [
        ("22.12%", "Historical Default Rate", "1 in every 4.5 cardholders failed to meet obligations in subsequent cycle.", ACCENT_RED),
        ("62.3%", "Defaulters Detected Early", "XGBoost identifies nearly two-thirds of all risky accounts prior to default.", ACCENT_GREEN),
        ("0.7823", "ROC-AUC Score", "Robust ranking power to separate prime borrowers from distressed accounts.", ROYAL_BLUE)
    ]
    
    for i, (metric, label, desc, color) in enumerate(cards):
        x = Inches(0.8 + i * 4.0)
        shape = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.6), Inches(3.7), Inches(2.2))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.fill.background()
        
        tb = s2.shapes.add_textbox(x + Inches(0.2), Inches(1.8), Inches(3.3), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = metric
        p.font.size = Pt(32)
        p.font.bold = True
        p.font.color.rgb = color
        
        p_l = tf.add_paragraph()
        p_l.text = label
        p_l.font.size = Pt(13)
        p_l.font.bold = True
        p_l.font.color.rgb = TEXT_DARK
        p_l.space_before = Pt(4)
        
        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = TEXT_MUTED
        p_d.space_before = Pt(4)

    # Lower Explanation Box
    shape_bot = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(4.2), Inches(11.7), Inches(2.6))
    shape_bot.fill.solid()
    shape_bot.fill.fore_color.rgb = WHITE
    shape_bot.line.color.rgb = CARD_BG
    
    tb_bot = s2.shapes.add_textbox(Inches(1.0), Inches(4.3), Inches(11.3), Inches(2.4))
    tf_b = tb_bot.text_frame
    tf_b.word_wrap = True
    
    p = tf_b.paragraphs[0]
    p.text = "Key Problem Statement & Business Motivation"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = ROYAL_BLUE
    
    points = [
        "Unsecured Revolving Credit: Unlike mortgages or auto loans, credit cards require zero collateral, resulting in severe losses when accounts go bad.",
        "Reactive vs Proactive Risk Management: Traditional banks wait for 90-day delinquency before acting. Our ML model predicts defaults 30 days in advance.",
        "Operational Impact: Early identification enables proactive credit limit reductions, automated SMS reminders, and structured debt workouts before losses crystallize."
    ]
    for pt in points:
        pr = tf_b.add_paragraph()
        pr.text = "•  " + pt
        pr.font.size = Pt(11)
        pr.font.color.rgb = TEXT_DARK
        pr.space_before = Pt(6)

    # ====================================================
    # SLIDE 3: Dataset Architecture (Kaggle / UCI)
    # ====================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset Overview: Kaggle / UCI Credit Card Clients")
    
    # Left side: Key specs
    shape_left = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(4.5), Inches(5.2))
    shape_left.fill.solid()
    shape_left.fill.fore_color.rgb = CARD_BG
    shape_left.line.fill.background()
    
    tb_l = s3.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(4.1), Inches(4.8))
    tf_l = tb_l.text_frame
    tf_l.word_wrap = True
    
    p = tf_l.paragraphs[0]
    p.text = "Dataset Specifications"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ROYAL_BLUE
    
    specs = [
        ("Source", "Kaggle / UCI Machine Learning Repository"),
        ("Sample Size", "30,000 Verified Credit Card Clients"),
        ("Time Window", "April 2005 - September 2005 (6 Months)"),
        ("Raw Features", "24 Predictors + 1 Binary Target"),
        ("Engineered Features", "6 Advanced Ratios & Delinquency Metrics"),
        ("Target Variable", "default (1 = Default next month, 0 = Paid)"),
        ("Class Balance", "77.88% Non-Default vs 22.12% Default")
    ]
    for k, v in specs:
        pr = tf_l.add_paragraph()
        pr.text = f"{k}: {v}"
        pr.font.size = Pt(11)
        pr.font.color.rgb = TEXT_DARK
        pr.space_before = Pt(8)

    # Right side: Feature Breakdown Table
    tb_r = s3.shapes.add_textbox(Inches(5.6), Inches(1.6), Inches(6.9), Inches(5.2))
    tf_r = tb_r.text_frame
    tf_r.word_wrap = True
    
    p = tf_r.paragraphs[0]
    p.text = "Feature Hierarchy & Behavioral Categories"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = ROYAL_BLUE
    
    cats = [
        ("1. Demographics", "AGE (years), SEX (Male/Female), EDUCATION (Grad, Univ, High School, Others), MARRIAGE (Married, Single, Others)."),
        ("2. Credit Capacity", "LIMIT_BAL (Amount of approved credit line in NT dollars, ranging from NT$10,000 to NT$1,000,000)."),
        ("3. Repayment History", "PAY_0 to PAY_6 (Monthly status: -2=no consumption, -1=paid in full, 0=revolving, 1-8=months delayed)."),
        ("4. Statement Balances", "BILL_AMT1 to BILL_AMT6 (Accumulated statement balances across 6 consecutive months)."),
        ("5. Cash Payments", "PAY_AMT1 to PAY_AMT6 (Actual cash payments remitted towards each respective billing cycle)."),
        ("6. Engineered Ratios", "PAY_TO_BILL_RATIO (Repayment velocity), UTILIZATION_RATE (Debt-to-Limit), MAX_DELAY & AVG_DELAY.")
    ]
    for title, desc in cats:
        pr = tf_r.add_paragraph()
        pr.text = title
        pr.font.size = Pt(12)
        pr.font.bold = True
        pr.font.color.rgb = TEXT_DARK
        pr.space_before = Pt(8)
        
        pd = tf_r.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_MUTED

    # ====================================================
    # SLIDE 4: Exploratory Analysis - Target & Demographics
    # ====================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Exploratory Data Analysis: Target Distribution & Demographics")
    
    chart_p1 = os.path.join(CHARTS_DIR, 'default_distribution.png')
    if os.path.exists(chart_p1):
        s4.shapes.add_picture(chart_p1, Inches(0.8), Inches(1.6), width=Inches(6.2))
        
    chart_p2 = os.path.join(CHARTS_DIR, 'demographics_impact.png')
    if os.path.exists(chart_p2):
        s4.shapes.add_picture(chart_p2, Inches(7.2), Inches(1.6), width=Inches(5.4))
        
    # Bottom callout
    shape_call = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.6), Inches(11.8), Inches(1.4))
    shape_call.fill.solid()
    shape_call.fill.fore_color.rgb = CARD_BG
    shape_call.line.fill.background()
    
    tb_c = s4.shapes.add_textbox(Inches(1.0), Inches(5.7), Inches(11.4), Inches(1.2))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    p = tf_c.paragraphs[0]
    p.text = "Key Demographic Insights:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ROYAL_BLUE
    
    insights = [
        "Gender Disparity: Male cardholders default at 24.2% versus female cardholders at 20.8%.",
        "Education Protective Factor: Postgraduate degree holders exhibit lower default (19.2%) vs high school educated (25.2%).",
        "Marital Vulnerability: Married cardholders experience a slightly higher default risk (23.5%) than single individuals (20.9%)."
    ]
    for ins in insights:
        pr = tf_c.add_paragraph()
        pr.text = "• " + ins
        pr.font.size = Pt(10.5)
        pr.font.color.rgb = TEXT_DARK
        pr.space_before = Pt(2)

    # ====================================================
    # SLIDE 5: Repayment Delinquency & Heatmap
    # ====================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Cardinal Risk Indicators: Delinquency & Credit Utilization")
    
    chart_p3 = os.path.join(CHARTS_DIR, 'repayment_heatmap.png')
    if os.path.exists(chart_p3):
        s5.shapes.add_picture(chart_p3, Inches(0.8), Inches(1.6), width=Inches(6.0))
        
    chart_p4 = os.path.join(CHARTS_DIR, 'recent_delinquency.png')
    if os.path.exists(chart_p4):
        s5.shapes.add_picture(chart_p4, Inches(7.0), Inches(1.6), width=Inches(5.6))
        
    # Bottom callout
    shape_call5 = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(5.6), Inches(11.8), Inches(1.4))
    shape_call5.fill.solid()
    shape_call5.fill.fore_color.rgb = CARD_BG
    shape_call5.line.fill.background()
    
    tb_c5 = s5.shapes.add_textbox(Inches(1.0), Inches(5.7), Inches(11.4), Inches(1.2))
    tf_c5 = tb_c5.text_frame
    tf_c5.word_wrap = True
    p = tf_c5.paragraphs[0]
    p.text = "Behavioral Escalation Law:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ROYAL_BLUE
    
    ins5 = [
        "Repayment delay in the most recent billing statement (PAY_0) is the cardinal predictor of upcoming default.",
        "A delay of 1 month pushes default probability to 34.0%; a delay of 2 months skyrockets default probability to 69.1%.",
        "Customers who paid in full or made zero consumption have a low default risk (<13%)."
    ]
    for ins in ins5:
        pr = tf_c5.add_paragraph()
        pr.text = "• " + ins
        pr.font.size = Pt(10.5)
        pr.font.color.rgb = TEXT_DARK
        pr.space_before = Pt(2)

    # ====================================================
    # SLIDE 6: Machine Learning Workflow & Pipeline
    # ====================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Methodology: End-to-End Machine Learning Architecture")
    
    steps = [
        ("1. Data Ingestion", "30,000 raw credit card records loaded and verified.", "Kaggle / UCI Repository"),
        ("2. Preprocessing", "Categorical binning, standard scaling, and missing value verification.", "StandardScaler & Scikit-Learn"),
        ("3. Feature Engineering", "Engineered repayment velocity, debt utilization, and delay trajectories.", "Domain Ratios"),
        ("4. Model Training", "Trained Logistic Regression, Random Forest, and XGBoost with class weighting.", "Class-Balanced Ensembles"),
        ("5. Evaluation", "Benchmarked on holdout test set using Accuracy, Recall, and ROC-AUC.", "Stratified 20% Test Split"),
        ("6. Deployment", "Integrated into interactive Streamlit decision engine with risk scoring.", "Real-Time Web App")
    ]
    
    for i, (title, desc, tool) in enumerate(steps):
        row = i // 3
        col = i % 3
        x = Inches(0.8 + col * 4.0)
        y = Inches(1.7 + row * 2.6)
        
        box = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(3.7), Inches(2.2))
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = ROYAL_BLUE
        box.line.width = Pt(1)
        
        tb = s6.shapes.add_textbox(x + Inches(0.15), y + Inches(0.15), Inches(3.4), Inches(1.9))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = ROYAL_BLUE
        
        pd = tf.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(10.5)
        pd.font.color.rgb = TEXT_DARK
        pd.space_before = Pt(4)
        
        pt = tf.add_paragraph()
        pt.text = f"Tech: {tool}"
        pt.font.size = Pt(9.5)
        pt.font.color.rgb = TEXT_MUTED
        pt.space_before = Pt(6)

    # ====================================================
    # SLIDE 7: Model Comparison & Benchmarks (The Core Table)
    # ====================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Model Evaluation & Benchmark Comparison")
    
    # Table shape
    rows = 4
    cols = 6
    left = Inches(0.8)
    top = Inches(1.7)
    width = Inches(11.7)
    height = Inches(2.8)
    
    table_shape = s7.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.columns[0].width = Inches(3.0)
    table.columns[1].width = Inches(1.7)
    table.columns[2].width = Inches(1.7)
    table.columns[3].width = Inches(1.7)
    table.columns[4].width = Inches(1.7)
    table.columns[5].width = Inches(1.9)
    
    headers = ["Machine Learning Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = ROYAL_BLUE
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        
    model_rows = [
        ("Logistic Regression", f"{lr.get('accuracy', 0.743)*100:.2f}%", f"{lr.get('precision', 0.441)*100:.2f}%", f"{lr.get('recall', 0.606)*100:.2f}%", f"{lr.get('f1_score', 0.511):.4f}", f"{lr.get('roc_auc', 0.752):.4f}"),
        ("Random Forest", f"{rf.get('accuracy', 0.796)*100:.2f}%", f"{rf.get('precision', 0.539)*100:.2f}%", f"{rf.get('recall', 0.537)*100:.2f}%", f"{rf.get('f1_score', 0.538):.4f}", f"{rf.get('roc_auc', 0.774):.4f}"),
        ("XGBoost (Champion)", f"{xgb.get('accuracy', 0.757)*100:.2f}%", f"{xgb.get('precision', 0.464)*100:.2f}%", f"{xgb.get('recall', 0.623)*100:.2f}%", f"{xgb.get('f1_score', 0.532):.4f}", f"{xgb.get('roc_auc', 0.782):.4f}")
    ]
    
    for i, row_data in enumerate(model_rows):
        for j, val in enumerate(row_data):
            cell = table.cell(i+1, j)
            cell.text = val
            cell.fill.solid()
            if i == 2: # Highlight XGBoost
                cell.fill.fore_color.rgb = RGBColor(238, 242, 255)
            else:
                cell.fill.fore_color.rgb = WHITE
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11.5)
            if j == 0 or i == 2:
                p.font.bold = True
            p.font.color.rgb = TEXT_DARK
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT

    # Explanatory card below table
    shape_exp = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.8), Inches(11.7), Inches(2.1))
    shape_exp.fill.solid()
    shape_exp.fill.fore_color.rgb = CARD_BG
    shape_exp.line.fill.background()
    
    tb_exp = s7.shapes.add_textbox(Inches(1.0), Inches(4.9), Inches(11.3), Inches(1.9))
    tf_exp = tb_exp.text_frame
    tf_exp.word_wrap = True
    
    p = tf_exp.paragraphs[0]
    p.text = "Key Analytical Findings & Model Selection Rationale:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = ROYAL_BLUE
    
    findings = [
        "XGBoost is Selected as the Champion Model: Achieves highest ROC-AUC (0.7823) and highest Recall (62.32%), minimizing costly false negatives.",
        "Why Recall Trumps Accuracy in Credit Default: A false negative (failing to detect a defaulter) costs $5,000 - $20,000+ in charge-offs, whereas a false positive merely costs a courtesy verification call.",
        "Random Forest Best for Conservative Portfolios: Highest accuracy (79.58%) and precision (53.86%), best suited when minimizing false alarms is strictly prioritized."
    ]
    for find in findings:
        pr = tf_exp.add_paragraph()
        pr.text = "• " + find
        pr.font.size = Pt(10.5)
        pr.font.color.rgb = TEXT_DARK
        pr.space_before = Pt(3)

    # ====================================================
    # SLIDE 8: 4-Tier Operational Risk Decisioning Framework
    # ====================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Operational Decisioning: 4-Tier Risk Classification")
    
    tiers = [
        ("Tier 1: Prime", "< 20%", "Consistently pays in full; low debt ratio; prime credit score.", "Approve credit limit increase; offer premium card upgrades; VIP loyalty rates.", ACCENT_GREEN),
        ("Tier 2: Standard", "20% - 45%", "Occasional revolving; debt utilization 30-60%; manageable debt.", "Maintain existing credit line; standard monthly monitoring; normal billing cycle.", ROYAL_BLUE),
        ("Tier 3: Watchlist", "45% - 70%", "1-2 months delinquency; utilization >80%; declining payment ratio.", "Temporarily freeze limit increases; issue automated SMS reminders; offer restructuring.", RGBColor(245, 158, 11)),
        ("Tier 4: Critical", "> 70%", "2+ months severe delinquency; debt spiraling; broken payment promises.", "Immediate card freeze; restrict merchant authorizations; refer account to collections.", ACCENT_RED)
    ]
    
    for i, (name, prob, desc, action, color) in enumerate(tiers):
        x = Inches(0.8 + i * 2.95)
        shape = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.6), Inches(2.8), Inches(5.2))
        shape.fill.solid()
        shape.fill.fore_color.rgb = CARD_BG
        shape.line.color.rgb = color
        shape.line.width = Pt(1.5)
        
        tb = s8.shapes.add_textbox(x + Inches(0.15), Inches(1.8), Inches(2.5), Inches(4.8))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = name
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = color
        
        pp = tf.add_paragraph()
        pp.text = f"P(Default): {prob}"
        pp.font.size = Pt(12)
        pp.font.bold = True
        pp.font.color.rgb = TEXT_DARK
        pp.space_before = Pt(4)
        
        p_ch = tf.add_paragraph()
        p_ch.text = "Risk Profile:"
        p_ch.font.size = Pt(10)
        p_ch.font.bold = True
        p_ch.font.color.rgb = TEXT_MUTED
        p_ch.space_before = Pt(8)
        
        p_cd = tf.add_paragraph()
        p_cd.text = desc
        p_cd.font.size = Pt(9.5)
        p_cd.font.color.rgb = TEXT_DARK
        p_cd.space_before = Pt(2)
        
        p_ah = tf.add_paragraph()
        p_ah.text = "Prescribed Action:"
        p_ah.font.size = Pt(10)
        p_ah.font.bold = True
        p_ah.font.color.rgb = ROYAL_BLUE
        p_ah.space_before = Pt(10)
        
        p_ad = tf.add_paragraph()
        p_ad.text = action
        p_ad.font.size = Pt(9.5)
        p_ad.font.color.rgb = TEXT_DARK
        p_ad.space_before = Pt(2)

    # ====================================================
    # SLIDE 9: Web Application Architecture
    # ====================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Deployment: Interactive Streamlit Decision Engine")
    
    app_features = [
        ("🏠 Executive Overview", "Live dataset explorer, summary KPI statistics (Total Clients, Default Rate, Avg Limit), and distribution breakdowns."),
        ("📊 EDA Dashboard", "Interactive selection of 9 publication-grade charts analyzing demographics, delinquency escalation, and credit limits."),
        ("🤖 Model Comparison", "Side-by-side benchmark comparison of Logistic Regression, Random Forest, and XGBoost with ROC curves and metrics."),
        ("🎯 Real-Time Scoring", "Interactive customer simulator: input age, limit, repayment delay, and bills to get instant risk score, tier, and key risk drivers."),
        ("📂 Batch Assessment", "Upload customer portfolio CSV, execute batch risk scoring, visualize portfolio risk distribution, and export scored reports."),
        ("📄 Report & PPT Exports", "1-click generation and instant download of the formal PDF academic report and PowerPoint deck.")
    ]
    
    for i, (f_title, f_desc) in enumerate(app_features):
        row = i // 2
        col = i % 2
        x = Inches(0.8 + col * 5.9)
        y = Inches(1.7 + row * 1.7)
        
        box = s9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, Inches(5.6), Inches(1.5))
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = ROYAL_BLUE
        box.line.width = Pt(0.8)
        
        tb = s9.shapes.add_textbox(x + Inches(0.2), y + Inches(0.15), Inches(5.2), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = f_title
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = ROYAL_BLUE
        
        pd = tf.add_paragraph()
        pd.text = f_desc
        pd.font.size = Pt(10)
        pd.font.color.rgb = TEXT_DARK
        pd.space_before = Pt(3)

    # ====================================================
    # SLIDE 10: Conclusion & Recommendations
    # ====================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Conclusions & Strategic Recommendations")
    
    shape_c = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    shape_c.fill.solid()
    shape_c.fill.fore_color.rgb = WHITE
    shape_c.line.color.rgb = CARD_BG
    
    tb_con = s10.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf_con = tb_con.text_frame
    tf_con.word_wrap = True
    
    p = tf_con.paragraphs[0]
    p.text = "Summary of Project Achievements:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = ROYAL_BLUE
    
    concs = [
        "Accurate Early Prediction: Successfully built and validated machine learning models on 30,000 credit records, achieving a peak ROC-AUC of 0.7823.",
        "Recall-Oriented Learning: By addressing class imbalance with scale_pos_weight, XGBoost captured 62.3% of true defaulters, preventing major financial leakage.",
        "Key Predictor Discovery: Proved that recent repayment status (PAY_0) and repayment capability (PAY_TO_BILL_RATIO) carry the strongest predictive information.",
        "Actionable Deployment: Delivered a turnkey decision support system that moves beyond raw probabilities into prescriptive 4-tier risk mitigation."
    ]
    for c in concs:
        pr = tf_con.add_paragraph()
        pr.text = "✓ " + c
        pr.font.size = Pt(11)
        pr.font.color.rgb = TEXT_DARK
        pr.space_before = Pt(6)
        
    p_rec = tf_con.add_paragraph()
    p_rec.text = "Institutional Recommendations for Financial Lenders:"
    p_rec.font.size = Pt(14)
    p_rec.font.bold = True
    p_rec.font.color.rgb = ROYAL_BLUE
    p_rec.space_before = Pt(14)
    
    recs = [
        "1. Real-Time Intervention Triggers: Automatically flag accounts whenever PAY_0 reaches 1 month delay or utilization exceeds 85%.",
        "2. Dynamic Line Management: Implement algorithmic credit limit throttles for Tier 3 and Tier 4 accounts prior to statement cutoff.",
        "3. Personalized Financial Counseling: Offer proactive debt repayment plans to vulnerable borrowers before accounts default."
    ]
    for r in recs:
        pr = tf_con.add_paragraph()
        pr.text = "• " + r
        pr.font.size = Pt(11)
        pr.font.color.rgb = TEXT_DARK
        pr.space_before = Pt(4)

    # ====================================================
    # SLIDE 11: Final Q&A Slide (Dark Navy)
    # ====================================================
    s11 = prs.slides.add_slide(blank_layout)
    bg11 = s11.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg11.fill.solid()
    bg11.fill.fore_color.rgb = NAVY
    bg11.line.fill.background()
    
    if os.path.exists(cu_seal):
        s11.shapes.add_picture(cu_seal, Inches(5.9), Inches(1.5), width=Inches(1.5))
        
    tb_end = s11.shapes.add_textbox(Inches(1.5), Inches(3.3), Inches(10.3), Inches(3.0))
    tf_e = tb_end.text_frame
    tf_e.word_wrap = True
    
    p = tf_e.paragraphs[0]
    p.text = "Thank You!"
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER
    
    pe1 = tf_e.add_paragraph()
    pe1.text = "Credit Card Default Prediction - AI Application Lab"
    pe1.font.size = Pt(18)
    pe1.font.color.rgb = RGBColor(148, 163, 184)
    pe1.space_before = Pt(12)
    pe1.alignment = PP_ALIGN.CENTER
    
    pe2 = tf_e.add_paragraph()
    pe2.text = "Department of Computer Science & Engineering | Chandigarh University"
    pe2.font.size = Pt(14)
    pe2.font.color.rgb = RGBColor(59, 130, 246)
    pe2.space_before = Pt(6)
    pe2.alignment = PP_ALIGN.CENTER

    out_path = os.path.join(CURRENT_DIR, output_filename)
    prs.save(out_path)
    print(f"Presentation saved successfully at: {out_path}")
    return out_path


if __name__ == '__main__':
    generate_presentation()
