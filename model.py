import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

ENGINEERED_COLS = [
    'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT', 'PAY_TO_BILL_RATIO',
    'UTILIZATION_RATE', 'MAX_DELAY', 'AVG_DELAY'
]

ALL_FEATURES = FEATURE_COLS + ENGINEERED_COLS


def engineer_features(df):
    """
    Engineer domain-specific financial risk features.
    """
    df_feat = df.copy()
    
    # 1. Clean categorical features
    # EDUCATION: 1=grad school, 2=university, 3=high school, 4=others (group 0, 5, 6 into 4)
    if 'EDUCATION' in df_feat.columns:
        df_feat['EDUCATION'] = df_feat['EDUCATION'].apply(lambda x: 4 if x not in [1, 2, 3] else x)
    
    # MARRIAGE: 1=married, 2=single, 3=others (group 0 into 3)
    if 'MARRIAGE' in df_feat.columns:
        df_feat['MARRIAGE'] = df_feat['MARRIAGE'].apply(lambda x: 3 if x not in [1, 2] else x)
        
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    # 2. Cumulative Financial Totals
    df_feat['TOTAL_BILL_AMT'] = df_feat[bill_cols].sum(axis=1)
    df_feat['TOTAL_PAY_AMT'] = df_feat[pay_cols].sum(axis=1)
    
    # 3. Repayment Capability Ratio (Total Pay / Total Bill)
    df_feat['PAY_TO_BILL_RATIO'] = np.where(
        df_feat['TOTAL_BILL_AMT'] > 0,
        df_feat['TOTAL_PAY_AMT'] / (df_feat['TOTAL_BILL_AMT'] + 1e-5),
        1.0
    ).clip(0, 5)
    
    # 4. Credit Utilization Rate (Most recent bill / Credit limit)
    df_feat['UTILIZATION_RATE'] = (df_feat['BILL_AMT1'] / np.maximum(df_feat['LIMIT_BAL'], 1.0)).clip(-0.5, 3.0)
    
    # 5. Delinquency Metrics
    df_feat['MAX_DELAY'] = df_feat[status_cols].max(axis=1).clip(lower=0)
    df_feat['AVG_DELAY'] = df_feat[status_cols].mean(axis=1).clip(lower=0)
    
    return df_feat


def preprocess_data(df, fit_scaler=True, scaler=None):
    """
    Apply feature engineering and standardize numerical features.
    Returns X_scaled, y, scaler, feature_names.
    """
    df_eng = engineer_features(df)
    
    X = df_eng[ALL_FEATURES].copy()
    y = df_eng['default'].values if 'default' in df_eng.columns else None
    
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler=False")
        X_scaled = scaler.transform(X)
        
    return X_scaled, y, scaler, ALL_FEATURES


def train_and_evaluate_models(df, save_dir=MODELS_DIR):
    """
    Train Logistic Regression, Random Forest, and XGBoost classifiers.
    Evaluate on test set using Accuracy, Precision, Recall, F1, and ROC-AUC.
    Saves trained models, scaler, and evaluation metrics.
    """
    print("=" * 60)
    print("       TRAINING CREDIT CARD DEFAULT CLASSIFICATION MODELS       ")
    print("=" * 60)
    
    df_eng = engineer_features(df)
    X = df_eng[ALL_FEATURES]
    y = df_eng['default'].values
    
    # Stratified Train-Test Split (80/20) to maintain class ratio
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Compute class ratio for XGBoost scale_pos_weight
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos = neg_count / pos_count
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=0.1,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=42
        )
    }
    
    results = {}
    best_auc = -1.0
    best_model_name = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions & Probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()
        cr = classification_report(y_test, y_pred, output_dict=True)
        
        fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
        pr_curve_p, pr_curve_r, pr_thresh = precision_recall_curve(y_test, y_prob)
        
        # Feature importances
        if hasattr(model, 'feature_importances_'):
            importances = dict(zip(ALL_FEATURES, model.feature_importances_.tolist()))
        elif hasattr(model, 'coef_'):
            importances = dict(zip(ALL_FEATURES, np.abs(model.coef_[0]).tolist()))
        else:
            importances = {}
            
        print(f"[{name}] Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        
        # Save individual model
        model_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)
        
        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4),
            'confusion_matrix': cm,
            'classification_report': cr,
            'feature_importances': importances,
            'roc_curve': {
                'fpr': [float(x) for x in fpr[::max(1, len(fpr)//100)]],
                'tpr': [float(x) for x in tpr[::max(1, len(tpr)//100)]]
            },
            'model_file': model_path
        }
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            
    # Save Scaler and Feature Names
    scaler_path = os.path.join(save_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    meta_info = {
        'best_model': best_model_name,
        'features': ALL_FEATURES,
        'feature_cols': FEATURE_COLS,
        'engineered_cols': ENGINEERED_COLS,
        'models': results,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'class_balance': {
            'train_pos': int(pos_count),
            'train_neg': int(neg_count),
            'default_rate_pct': round((pos_count / len(y_train)) * 100, 2)
        }
    }
    
    metrics_path = os.path.join(save_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(meta_info, f, indent=4)
        
    print(f"\nAll models saved to: {save_dir}")
    print(f"Best Performing Model by ROC-AUC: {best_model_name} (ROC-AUC = {best_auc:.4f})")
    
    return meta_info


def load_model_and_scaler(model_name='XGBoost', models_dir=MODELS_DIR):
    """Load a specific saved model and the pre-fitted scaler."""
    file_map = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Random Forest': 'random_forest.joblib',
        'XGBoost': 'xgboost.joblib'
    }
    fname = file_map.get(model_name, 'xgboost.joblib')
    model_path = os.path.join(models_dir, fname)
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_customer_default(customer_data, model_name='XGBoost', models_dir=MODELS_DIR):
    """
    Perform real-time customer default prediction.
    
    Parameters:
        customer_data: dict containing customer attributes
        model_name: 'Logistic Regression', 'Random Forest', or 'XGBoost'
        
    Returns:
        dict with prediction (0 or 1), probability (0.0 - 1.0), risk tier,
        risk drivers, and recommended financial action.
    """
    model, scaler = load_model_and_scaler(model_name, models_dir)
    if model is None:
        raise FileNotFoundError("Trained models not found. Please train models first.")
        
    df_single = pd.DataFrame([customer_data])
    df_eng = engineer_features(df_single)
    
    X_input = df_eng[ALL_FEATURES]
    X_scaled = scaler.transform(X_input)
    
    pred_class = int(model.predict(X_scaled)[0])
    prob_default = float(model.predict_proba(X_scaled)[0][1])
    
    # Determine Risk Tier
    if prob_default < 0.20:
        risk_tier = "Tier 1: Low Risk (Prime)"
        risk_color = "#10B981" # Green
        action = "Approve Standard / Enhanced Credit Limit. Low risk of default."
    elif prob_default < 0.45:
        risk_tier = "Tier 2: Moderate Risk (Standard)"
        risk_color = "#F59E0B" # Amber
        action = "Approve with standard monitoring. Keep current credit line."
    elif prob_default < 0.70:
        risk_tier = "Tier 3: High Risk (Watchlist)"
        risk_color = "#F97316" # Orange
        action = "Place on Watchlist. Reduce credit limit by 25-50% and send payment reminders."
    else:
        risk_tier = "Tier 4: Critical Risk (Severe)"
        risk_color = "#EF4444" # Red
        action = "Immediate Intervention. Freeze card transactions, restrict credit line, and refer to Collections."

    # Identify Key Risk Factors
    risk_factors = []
    
    pay_0 = customer_data.get('PAY_0', 0)
    if pay_0 >= 2:
        risk_factors.append(f"Severe recent delinquency: {pay_0} months past due in September")
    elif pay_0 == 1:
        risk_factors.append("Payment delay of 1 month in latest statement")
        
    util_rate = df_eng['UTILIZATION_RATE'].iloc[0] * 100
    if util_rate > 90:
        risk_factors.append(f"Dangerous credit utilization: {util_rate:.1f}% of limit exhausted")
    elif util_rate > 70:
        risk_factors.append(f"Elevated credit utilization: {util_rate:.1f}%")
        
    ratio = df_eng['PAY_TO_BILL_RATIO'].iloc[0]
    if ratio < 0.15 and df_eng['TOTAL_BILL_AMT'].iloc[0] > 10000:
        risk_factors.append("Low repayment velocity: paying less than 15% of accumulated bills")
        
    max_del = df_eng['MAX_DELAY'].iloc[0]
    if max_del >= 3:
        risk_factors.append(f"Persistent historical delinquency: max delay reached {int(max_del)} months")
        
    if not risk_factors:
        risk_factors.append("Healthy payment history with regular on-time settlements.")
        
    return {
        'prediction': pred_class,
        'prediction_label': 'Default (High Risk)' if pred_class == 1 else 'Non-Default (Reliable)',
        'default_probability': round(prob_default, 4),
        'default_probability_pct': round(prob_default * 100, 2),
        'risk_tier': risk_tier,
        'risk_color': risk_color,
        'recommended_action': action,
        'key_risk_factors': risk_factors,
        'model_used': model_name,
        'utilization_rate_pct': round(util_rate, 2),
        'pay_to_bill_ratio': round(ratio, 2)
    }


def get_model_comparison(models_dir=MODELS_DIR):
    """Retrieve saved model metrics comparison table."""
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, 'r') as f:
        meta = json.load(f)
    return meta


if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_of_credit_card_clients.csv')
    if os.path.exists(csv_path):
        df_full = pd.read_csv(csv_path)
        train_and_evaluate_models(df_full)
