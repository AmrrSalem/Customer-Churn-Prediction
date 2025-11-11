#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit App: Telco Customer Churn (Interactive)
-------------------------------------------------
Run locally:
    streamlit run app.py
"""

from pathlib import Path
import io

import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, average_precision_score, classification_report
)

# Advanced ensemble models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

PRIMARY_URL_IBM = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/fenago/datasets/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ----------------- Utils -----------------
@st.cache_data(show_spinner=False)
def load_telco_df(source: str, path_or_url: str | None):
    na_vals = [" ", ""]
    if source == "Local file":
        p = Path(path_or_url).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Local path not found: {p}")
        df = pd.read_csv(p, na_values=na_vals)
        return df, str(p)
    elif source == "Web (IBM GitHub)":
        df = pd.read_csv(PRIMARY_URL_IBM, na_values=na_vals)
        return df, PRIMARY_URL_IBM
    elif source == "Web (Fallback mirror)":
        df = pd.read_csv(FALLBACK_URL, na_values=na_vals)
        return df, FALLBACK_URL
    else:
        raise ValueError("Unknown source")

def engineer_features(df: pd.DataFrame):
    """Advanced feature engineering for customer churn prediction"""
    df = df.copy()

    # Tenure groups (categorical bins)
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, float('inf')],
            labels=["0-12", "12-24", "24-48", "48+"]
        ).astype(str)

    # Total charges per month ratio (efficiency metric)
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        df["charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)  # +1 to avoid division by zero
        df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Service combinations (interaction features)
    if "InternetService" in df.columns and "PhoneService" in df.columns:
        df["has_internet_and_phone"] = ((df["InternetService"] != "No") &
                                         (df["PhoneService"] == "Yes")).astype(int)

    if "OnlineSecurity" in df.columns and "OnlineBackup" in df.columns:
        df["security_and_backup"] = ((df["OnlineSecurity"] == "Yes") &
                                      (df["OnlineBackup"] == "Yes")).astype(int)

    # Count of services (total number of services subscribed)
    service_cols = ["PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    available_service_cols = [c for c in service_cols if c in df.columns]
    if available_service_cols:
        df["total_services"] = df[available_service_cols].apply(
            lambda row: sum(1 for val in row if val == "Yes"), axis=1
        )

    # Contract risk score (ordinal encoding with business logic)
    if "Contract" in df.columns:
        contract_risk = {"Month-to-month": 2, "One year": 1, "Two year": 0}
        df["contract_risk"] = df["Contract"].map(contract_risk).fillna(1)

    # Payment method risk (some payment methods correlate with churn)
    if "PaymentMethod" in df.columns:
        df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)

    return df

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Apply feature engineering first
    df = engineer_features(df)

    # Coerce TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop id
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Encode categoricals except target
    if "Churn" not in df.columns:
        raise KeyError("The dataset does not contain a 'Churn' column.")

    label_encoders = {}
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "Churn" in cat_cols:
        cat_cols.remove("Churn")

    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        label_encoders[c] = le

    # target
    target_encoder = LabelEncoder()
    df["Churn"] = target_encoder.fit_transform(df["Churn"].astype(str))

    return df, label_encoders, target_encoder

def train_models(X_train, X_test, y_train, y_test, params):
    models = {}

    if "Random Forest" in params["models"]:
        rf = RandomForestClassifier(
            n_estimators=params.get("rf_n_estimators", 200),
            max_depth=params.get("rf_max_depth", 10) if params.get("rf_max_depth", 10) > 0 else None,
            min_samples_split=params.get("rf_min_samples_split", 2),
            random_state=42,
        )
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]
        models["Random Forest"] = (rf, proba)

    if "Logistic Regression" in params["models"]:
        lr = LogisticRegression(
            C=params.get("lr_C", 1.0),
            max_iter=params.get("lr_max_iter", 1000),
            solver=params.get("lr_solver", "lbfgs"),
            random_state=42
        )
        lr.fit(params["scaler"].transform(X_train), y_train)
        proba = lr.predict_proba(params["scaler"].transform(X_test))[:, 1]
        models["Logistic Regression"] = (lr, proba)

    if "XGBoost" in params["models"]:
        xgb_model = xgb.XGBClassifier(
            n_estimators=params.get("xgb_n_estimators", 200),
            max_depth=params.get("xgb_max_depth", 6),
            learning_rate=params.get("xgb_learning_rate", 0.1),
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        proba = xgb_model.predict_proba(X_test)[:, 1]
        models["XGBoost"] = (xgb_model, proba)

    if "LightGBM" in params["models"]:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=params.get("lgb_n_estimators", 200),
            max_depth=params.get("lgb_max_depth", 6),
            learning_rate=params.get("lgb_learning_rate", 0.1),
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        proba = lgb_model.predict_proba(X_test)[:, 1]
        models["LightGBM"] = (lgb_model, proba)

    if "CatBoost" in params["models"]:
        cat_model = CatBoostClassifier(
            iterations=params.get("cat_iterations", 200),
            depth=params.get("cat_depth", 6),
            learning_rate=params.get("cat_learning_rate", 0.1),
            random_state=42,
            verbose=0
        )
        cat_model.fit(X_train, y_train)
        proba = cat_model.predict_proba(X_test)[:, 1]
        models["CatBoost"] = (cat_model, proba)

    if "Gradient Boosting" in params["models"]:
        gb_model = GradientBoostingClassifier(
            n_estimators=params.get("gb_n_estimators", 200),
            max_depth=params.get("gb_max_depth", 6),
            learning_rate=params.get("gb_learning_rate", 0.1),
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        proba = gb_model.predict_proba(X_test)[:, 1]
        models["Gradient Boosting"] = (gb_model, proba)

    if "AdaBoost" in params["models"]:
        ada_model = AdaBoostClassifier(
            n_estimators=params.get("ada_n_estimators", 200),
            learning_rate=params.get("ada_learning_rate", 1.0),
            random_state=42
        )
        ada_model.fit(X_train, y_train)
        proba = ada_model.predict_proba(X_test)[:, 1]
        models["AdaBoost"] = (ada_model, proba)

    # compute metrics at default threshold 0.5 for display; threshold slider later
    metrics = {}
    for name, (model, proba) in models.items():
        preds = (proba >= 0.5).astype(int)
        metrics[name] = {
            "AUC": roc_auc_score(y_test, proba),
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "AP": average_precision_score(y_test, proba),  # Average Precision for PR curve
            "proba": proba,
            "model": model,
        }
    return metrics

def plot_confusion(cm, labels=("Retained","Churned")):
    z = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"Pred {labels[0]}", f"Pred {labels[1]}"],
        y=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
        text=z, texttemplate="%{text}",
        colorscale="Blues"
    ))
    fig.update_layout(height=350, margin=dict(l=40,r=40,t=40,b=40))
    return fig

def plot_roc(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      height=350, margin=dict(l=40,r=40,t=40,b=40))
    return fig

def plot_precision_recall(y_true, y_proba):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR (AP={ap:.3f})"))
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_roc_comparison(metrics, y_test):
    """Plot ROC curves for all models on one chart"""
    fig = go.Figure()
    for name, m in metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m["proba"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={m['AUC']:.3f})"
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Chance", line=dict(dash="dash", color="gray")
    ))
    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_pr_comparison(metrics, y_test):
    """Plot Precision-Recall curves for all models on one chart"""
    fig = go.Figure()
    for name, m in metrics.items():
        precision, recall, _ = precision_recall_curve(y_test, m["proba"])
        fig.add_trace(go.Scatter(
            x=recall, y=precision, mode="lines",
            name=f"{name} (AP={m['AP']:.3f})"
        ))
    fig.update_layout(
        title="Precision-Recall Curve Comparison",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


# ----------------- UI -----------------
st.set_page_config(page_title="Telco Customer Churn (Interactive)", layout="wide")
st.title("📉 Telco Customer Churn — Interactive Demo")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose data source:", ["Web (IBM GitHub)", "Web (Fallback mirror)", "Local file"])
    local_path = st.text_input("Local CSV path (if using Local file)", value="./data/Telco-Customer-Churn.csv")
    st.caption("Tip: If your CSV is inside a `data/` folder next to this app, use `./data/Telco-Customer-Churn.csv`.")

    st.header("Train/Test Split")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    st.header("Models")
    all_models = ["Random Forest", "Logistic Regression", "XGBoost", "LightGBM",
                  "CatBoost", "Gradient Boosting", "AdaBoost"]
    choose_models = st.multiselect(
        "Select models",
        all_models,
        default=["Random Forest", "XGBoost", "LightGBM"]
    )

    # Dynamic hyperparameters based on selected models
    model_params = {}

    if choose_models:
        st.header("⚙️ Hyperparameters")

        for model_name in choose_models:
            with st.expander(f"{model_name} Parameters", expanded=False):
                if model_name == "Random Forest":
                    model_params["rf_n_estimators"] = st.slider(
                        "n_estimators", 50, 500, 200, 50, key="rf_n_est"
                    )
                    model_params["rf_max_depth"] = st.slider(
                        "max_depth (0 = None)", 0, 30, 10, 1, key="rf_depth"
                    )
                    model_params["rf_min_samples_split"] = st.slider(
                        "min_samples_split", 2, 20, 2, 1, key="rf_min_split"
                    )

                elif model_name == "Logistic Regression":
                    model_params["lr_C"] = st.slider(
                        "C (Inverse regularization)", 0.01, 10.0, 1.0, 0.1, key="lr_c"
                    )
                    model_params["lr_max_iter"] = st.slider(
                        "max_iter", 100, 2000, 1000, 100, key="lr_iter"
                    )
                    model_params["lr_solver"] = st.selectbox(
                        "solver", ["lbfgs", "liblinear", "saga"], key="lr_solver"
                    )

                elif model_name == "XGBoost":
                    model_params["xgb_n_estimators"] = st.slider(
                        "n_estimators", 50, 500, 200, 50, key="xgb_n_est"
                    )
                    model_params["xgb_max_depth"] = st.slider(
                        "max_depth", 3, 15, 6, 1, key="xgb_depth"
                    )
                    model_params["xgb_learning_rate"] = st.slider(
                        "learning_rate", 0.01, 0.3, 0.1, 0.01, key="xgb_lr"
                    )

                elif model_name == "LightGBM":
                    model_params["lgb_n_estimators"] = st.slider(
                        "n_estimators", 50, 500, 200, 50, key="lgb_n_est"
                    )
                    model_params["lgb_max_depth"] = st.slider(
                        "max_depth", 3, 15, 6, 1, key="lgb_depth"
                    )
                    model_params["lgb_learning_rate"] = st.slider(
                        "learning_rate", 0.01, 0.3, 0.1, 0.01, key="lgb_lr"
                    )

                elif model_name == "CatBoost":
                    model_params["cat_iterations"] = st.slider(
                        "iterations", 50, 500, 200, 50, key="cat_iter"
                    )
                    model_params["cat_depth"] = st.slider(
                        "depth", 3, 10, 6, 1, key="cat_depth"
                    )
                    model_params["cat_learning_rate"] = st.slider(
                        "learning_rate", 0.01, 0.3, 0.1, 0.01, key="cat_lr"
                    )

                elif model_name == "Gradient Boosting":
                    model_params["gb_n_estimators"] = st.slider(
                        "n_estimators", 50, 500, 200, 50, key="gb_n_est"
                    )
                    model_params["gb_max_depth"] = st.slider(
                        "max_depth", 3, 15, 6, 1, key="gb_depth"
                    )
                    model_params["gb_learning_rate"] = st.slider(
                        "learning_rate", 0.01, 0.3, 0.1, 0.01, key="gb_lr"
                    )

                elif model_name == "AdaBoost":
                    model_params["ada_n_estimators"] = st.slider(
                        "n_estimators", 50, 500, 200, 50, key="ada_n_est"
                    )
                    model_params["ada_learning_rate"] = st.slider(
                        "learning_rate", 0.1, 2.0, 1.0, 0.1, key="ada_lr"
                    )

    st.header("Decision Threshold")
    threshold = st.slider("Threshold for classification", 0.1, 0.9, 0.5, 0.01)

    st.header("Run")
    run_btn = st.button("Run Training")

# Load data
try:
    df, source_str = load_telco_df(source, local_path if source=="Local file" else None)
    st.success(f"Loaded data from: {source_str}")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Data preview + filters
st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rows", len(df))
with col2:
    st.metric("Columns", len(df.columns))
with col3:
    if "Churn" in df.columns:
        churn_rate = df["Churn"].eq("Yes").mean()
        st.metric("Churn rate", f"{churn_rate:.1%}")

# Exploratory charts
st.markdown("### Exploratory Analysis")
eda_cols = st.multiselect(
    "Choose categorical columns to break down churn by",
    options=[c for c in df.columns if df[c].dtype == "object" and c != "Churn"],
    default=["Contract","InternetService","PaymentMethod"] if all(x in df.columns for x in ["Contract","InternetService","PaymentMethod"]) else []
)

if "Churn" in df.columns and eda_cols:
    for c in eda_cols:
        tmp = df.groupby([c,"Churn"]).size().reset_index(name="count")
        fig = px.bar(tmp, x=c, y="count", color="Churn", barmode="group", title=f"Churn by {c}")
        st.plotly_chart(fig, use_container_width=True)

# Preprocess
try:
    df_processed, encoders, target_encoder = preprocess(df)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

X = df_processed.drop(columns=["Churn"])
y = df_processed["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)

# Merge model parameters with other params
params = {
    "models": choose_models,
    "scaler": scaler,
}
params.update(model_params)  # Add all dynamic model parameters

if run_btn:
    metrics = train_models(X_train, X_test, y_train, y_test, params)

    if not metrics:
        st.warning("Please select at least one model.")
        st.stop()

    # Model comparison table
    st.markdown("## 📊 Model Performance Comparison")

    # Create comparison dataframe
    comparison_data = []
    for name, m in metrics.items():
        comparison_data.append({
            "Model": name,
            "AUC": f"{m['AUC']:.4f}",
            "Accuracy": f"{m['Accuracy']:.4f}",
            "Precision": f"{m['Precision']:.4f}",
            "Recall": f"{m['Recall']:.4f}",
            "F1-Score": f"{m['F1']:.4f}",
            "Avg Precision": f"{m['AP']:.4f}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    # Sort by AUC descending
    comparison_df = comparison_df.sort_values("AUC", ascending=False).reset_index(drop=True)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # Model cards with metrics
    st.markdown("### Individual Model Metrics")
    for name, m in metrics.items():
        with st.expander(f"📈 {name} Details"):
            cols = st.columns(5)
            cols[0].metric("AUC", f"{m['AUC']:.3f}")
            cols[1].metric("Accuracy", f"{m['Accuracy']:.3f}")
            cols[2].metric("Precision", f"{m['Precision']:.3f}")
            cols[3].metric("Recall", f"{m['Recall']:.3f}")
            cols[4].metric("F1-Score", f"{m['F1']:.3f}")

    # Pick best by AUC
    best_name = max(metrics.keys(), key=lambda k: metrics[k]["AUC"])
    best = metrics[best_name]
    st.success(f"🏆 Best model by AUC: **{best_name}** (AUC={best['AUC']:.3f})")

    # Model Comparison Charts
    st.markdown("## 📈 Performance Curves")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_roc_comparison(metrics, y_test), use_container_width=True)
    with col2:
        st.plotly_chart(plot_pr_comparison(metrics, y_test), use_container_width=True)

    # Best model confusion matrix
    st.markdown(f"## 🎯 Best Model Analysis: {best_name}")

    # Threshold application
    proba = best["proba"]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Confusion Matrix (Threshold={threshold:.2f})")
        fig_cm = plot_confusion(cm)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("### Classification Report")
        # Generate classification report
        report = classification_report(y_test, preds, target_names=["Retained", "Churned"], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        # Format and display only the main metrics
        display_report = report_df.loc[["Retained", "Churned", "macro avg", "weighted avg"], ["precision", "recall", "f1-score", "support"]]
        st.dataframe(display_report.style.format("{:.3f}", subset=["precision", "recall", "f1-score"]), use_container_width=True)

    # Feature importance / coefficients
    st.markdown("### Feature Importance / Coefficients")
    if hasattr(best["model"], "feature_importances_"):
        # Tree-based models
        importances = best["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
        fig = px.bar(imp_df.head(20), x="importance", y="feature", orientation="h", title=f"Top 20 Features - {best_name}")
        st.plotly_chart(fig, use_container_width=True)
    elif hasattr(best["model"], "coef_"):
        # Logistic Regression coefficients
        coef = best["model"].coef_.ravel()
        coef_df = pd.DataFrame({"feature": X.columns, "coefficient": coef}).sort_values("coefficient")
        fig = px.bar(coef_df, x="coefficient", y="feature", orientation="h", title=f"Feature Coefficients - {best_name}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

    # Business impact widget
    st.markdown("## Business Impact (what-if)")
    monthly_rev = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 70.0
    retention_cost = st.number_input("Retention program cost per flagged customer ($)", value=50.0, min_value=0.0, step=5.0)
    success_rate = st.slider("Retention success rate", 0.0, 1.0, 0.70, 0.05)

    tn, fp, fn, tp = cm.ravel()
    annual_per_cust = monthly_rev * 12
    revenue_saved = tp * success_rate * annual_per_cust
    retention_total_cost = (tp + fp) * retention_cost
    net_benefit = revenue_saved - retention_total_cost

    cols = st.columns(4)
    cols[0].metric("TP (saved)", tp)
    cols[1].metric("FP (over-treatment)", fp)
    cols[2].metric("FN (missed churners)", fn)
    cols[3].metric("Net benefit ($)", f"{net_benefit:,.0f}")

    st.caption("Note: Simple model; plug your real economics for production use.")

    # Predict on selected rows
    st.markdown("## Try It On a Few Customers")
    n_samples = st.slider("How many random test customers?", 1, 20, 5, 1)
    idx = np.random.choice(len(X_test), size=n_samples, replace=False)
    sample = X_test.iloc[idx]
    if best_name == "Random Forest":
        p = best["model"].predict_proba(sample)[:,1]
    else:
        p = best["model"].predict_proba(scaler.transform(sample))[:,1]

    sample_out = sample.copy()
    sample_out["churn_prob"] = p
    sample_out["prediction"] = np.where(sample_out["churn_prob"] >= threshold, "High Risk", "Low Risk")
    st.dataframe(sample_out, use_container_width=True)

    # Download predictions
    csv_buf = io.StringIO()
    sample_out.to_csv(csv_buf, index=False)
    st.download_button("Download sample predictions CSV", csv_buf.getvalue(), file_name="sample_predictions.csv", mime="text/csv")

else:
    st.info("Set your options in the sidebar and click **Run Training** to begin.")
