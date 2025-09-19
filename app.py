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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)

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

def preprocess(df: pd.DataFrame):
    df = df.copy()
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
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"] if params["rf_max_depth"] > 0 else None,
            min_samples_split=2,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        proba = rf.predict_proba(X_test)[:, 1]
        models["Random Forest"] = (rf, proba)

    if "Logistic Regression" in params["models"]:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(params["scaler"].transform(X_train), y_train)
        proba = lr.predict_proba(params["scaler"].transform(X_test))[:, 1]
        models["Logistic Regression"] = (lr, proba)

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
    choose_models = st.multiselect("Select models", ["Random Forest", "Logistic Regression"],
                                   default=["Random Forest","Logistic Regression"])

    st.subheader("Random Forest params")
    rf_n_estimators = st.slider("n_estimators", 50, 400, 200, 50)
    rf_max_depth = st.slider("max_depth (0 = None)", 0, 30, 10, 1)

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

params = {
    "models": choose_models,
    "rf_n_estimators": rf_n_estimators,
    "rf_max_depth": rf_max_depth,
    "scaler": scaler,
}

if run_btn:
    metrics = train_models(X_train, X_test, y_train, y_test, params)

    if not metrics:
        st.warning("Please select at least one model.")
        st.stop()

    # Model cards
    st.markdown("## Results")
    for name, m in metrics.items():
        st.markdown(f"### {name}")
        cols = st.columns(4)
        cols[0].metric("AUC", f"{m['AUC']:.3f}")
        cols[1].metric("Accuracy@0.5", f"{m['Accuracy']:.3f}")
        cols[2].metric("Precision@0.5", f"{m['Precision']:.3f}")
        cols[3].metric("Recall@0.5", f"{m['Recall']:.3f}")

    # Pick best by AUC
    best_name = max(metrics.keys(), key=lambda k: metrics[k]["AUC"])
    best = metrics[best_name]
    st.success(f"Best model by AUC: **{best_name}** (AUC={best['AUC']:.3f})")

    # Threshold application
    proba = best["proba"]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    fig_cm = plot_confusion(cm)
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC
    st.plotly_chart(plot_roc(y_test, proba), use_container_width=True)

    # Feature importance / coefficients
    st.markdown("### Feature Importance / Coefficients")
    if best_name == "Random Forest":
        importances = best["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
        fig = px.bar(imp_df.head(20), x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Logistic Regression coefficients
        coef = best["model"].coef_.ravel()
        coef_df = pd.DataFrame({"feature": X.columns, "coefficient": coef}).sort_values("coefficient")
        fig = px.bar(coef_df, x="coefficient", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

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
