#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Churn Prediction - Multi-Client Edition
-------------------------------------------------
White-label churn dashboard: upload any CSV, map columns, run 7 ML models.

Client branding and auth via .streamlit/secrets.toml:
    CLIENT_NAME  = "Acme Corp"
    APP_PASSWORD = "secret123"        # omit to disable password gate
    MONTHLY_COL  = "MonthlyCharges"   # optional revenue column hint

Run locally:
    streamlit run app.py
"""
from pathlib import Path
import io, hashlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, precision_recall_curve,
                              average_precision_score, classification_report)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

PRIMARY_URL_IBM = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
                   "/master/data/Telco-Customer-Churn.csv")
FALLBACK_URL    = ("https://raw.githubusercontent.com/fenago/datasets/main"
                   "/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def _get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _check_password(app_password):
    if not app_password:
        return True
    pwd_hash = hashlib.sha256(app_password.encode()).hexdigest()
    if st.session_state.get("authenticated"):
        return True
    with st.form("login"):
        st.subheader("Login")
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if hashlib.sha256(entered.encode()).hexdigest() == pwd_hash:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


@st.cache_data(show_spinner=False)
def load_df(source, path_or_url, uploaded_bytes=None):
    na_vals = [" ", ""]
    if source == "Upload CSV":
        return pd.read_csv(io.BytesIO(uploaded_bytes), na_values=na_vals), "uploaded file"
    elif source == "Local file":
        p = Path(path_or_url).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Local path not found: {p}")
        return pd.read_csv(p, na_values=na_vals), str(p)
    elif source == "Demo - IBM Telco":
        return pd.read_csv(PRIMARY_URL_IBM, na_values=na_vals), PRIMARY_URL_IBM
    elif source == "Demo - Fallback mirror":
        return pd.read_csv(FALLBACK_URL, na_values=na_vals), FALLBACK_URL
    raise ValueError("Unknown source")


def engineer_features(df):
    df = df.copy()
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=[0, 12, 24, 48, float("inf")],
            labels=["0-12", "12-24", "24-48", "48+"]
        ).astype(str)
    if all(c in df.columns for c in ["TotalCharges", "MonthlyCharges", "tenure"]):
        df["charges_per_tenure"]     = df["TotalCharges"] / (df["tenure"] + 1)
        df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
        for col in ["charges_per_tenure", "monthly_to_total_ratio"]:
            df[col] = df[col].replace([float("inf"), -float("inf")], 0).fillna(0)
    if all(c in df.columns for c in ["InternetService", "PhoneService"]):
        df["has_internet_and_phone"] = (
            (df["InternetService"] != "No") & (df["PhoneService"] == "Yes")
        ).astype(int)
    if all(c in df.columns for c in ["OnlineSecurity", "OnlineBackup"]):
        df["security_and_backup"] = (
            (df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes")
        ).astype(int)
    svc_cols = [c for c in ["PhoneService","InternetService","OnlineSecurity",
                             "OnlineBackup","DeviceProtection","TechSupport",
                             "StreamingTV","StreamingMovies"] if c in df.columns]
    if svc_cols:
        df["total_services"] = df[svc_cols].apply(
            lambda row: sum(1 for v in row if v == "Yes"), axis=1)
    if "Contract" in df.columns:
        df["contract_risk"] = df["Contract"].map(
            {"Month-to-month": 2, "One year": 1, "Two year": 0}).fillna(1)
    if "PaymentMethod" in df.columns:
        df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


def preprocess(df, target_col, id_col):
    df = df.copy()
    df = engineer_features(df)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    if id_col and id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    label_encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        label_encoders[c] = le
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))
    return df, label_encoders, target_encoder


def train_models(X_train, X_test, y_train, y_test, params):
    models = {}
    scaler = params["scaler"]
    if "Random Forest" in params["models"]:
        m = RandomForestClassifier(
            n_estimators=params.get("rf_n_estimators", 200),
            max_depth=params.get("rf_max_depth", 10) or None,
            min_samples_split=params.get("rf_min_samples_split", 2),
            random_state=42)
        m.fit(X_train, y_train)
        models["Random Forest"] = (m, m.predict_proba(X_test)[:, 1])
    if "Logistic Regression" in params["models"]:
        m = LogisticRegression(C=params.get("lr_C", 1.0),
                               max_iter=params.get("lr_max_iter", 1000),
                               solver=params.get("lr_solver", "lbfgs"),
                               random_state=42)
        m.fit(scaler.transform(X_train), y_train)
        models["Logistic Regression"] = (m, m.predict_proba(scaler.transform(X_test))[:, 1])
    if "XGBoost" in params["models"]:
        m = xgb.XGBClassifier(n_estimators=params.get("xgb_n_estimators", 200),
                               max_depth=params.get("xgb_max_depth", 6),
                               learning_rate=params.get("xgb_learning_rate", 0.1),
                               random_state=42, eval_metric="logloss")
        m.fit(X_train, y_train)
        models["XGBoost"] = (m, m.predict_proba(X_test)[:, 1])
    if "LightGBM" in params["models"]:
        m = lgb.LGBMClassifier(n_estimators=params.get("lgb_n_estimators", 200),
                                max_depth=params.get("lgb_max_depth", 6),
                                learning_rate=params.get("lgb_learning_rate", 0.1),
                                random_state=42, verbose=-1)
        m.fit(X_train, y_train)
        models["LightGBM"] = (m, m.predict_proba(X_test)[:, 1])
    if "CatBoost" in params["models"]:
        m = CatBoostClassifier(iterations=params.get("cat_iterations", 200),
                               depth=params.get("cat_depth", 6),
                               learning_rate=params.get("cat_learning_rate", 0.1),
                               random_state=42, verbose=0)
        m.fit(X_train, y_train)
        models["CatBoost"] = (m, m.predict_proba(X_test)[:, 1])
    if "Gradient Boosting" in params["models"]:
        m = GradientBoostingClassifier(n_estimators=params.get("gb_n_estimators", 200),
                                       max_depth=params.get("gb_max_depth", 6),
                                       learning_rate=params.get("gb_learning_rate", 0.1),
                                       random_state=42)
        m.fit(X_train, y_train)
        models["Gradient Boosting"] = (m, m.predict_proba(X_test)[:, 1])
    if "AdaBoost" in params["models"]:
        m = AdaBoostClassifier(n_estimators=params.get("ada_n_estimators", 200),
                               learning_rate=params.get("ada_learning_rate", 1.0),
                               random_state=42)
        m.fit(X_train, y_train)
        models["AdaBoost"] = (m, m.predict_proba(X_test)[:, 1])
    metrics = {}
    for name, (model, proba) in models.items():
        preds = (proba >= 0.5).astype(int)
        metrics[name] = {
            "AUC": roc_auc_score(y_test, proba),
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "AP": average_precision_score(y_test, proba),
            "proba": proba, "model": model,
        }
    return metrics


def plot_confusion(cm, labels=("Retained", "Churned")):
    z = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[f"Pred {labels[0]}", f"Pred {labels[1]}"],
        y=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
        text=z, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=40))
    return fig


def plot_roc_comparison(metrics, y_test):
    fig = go.Figure()
    for name, m in metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m["proba"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"{name} (AUC={m['AUC']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              name="Chance", line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Curve Comparison",
                      xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                      height=500, margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_pr_comparison(metrics, y_test):
    fig = go.Figure()
    for name, m in metrics.items():
        precision, recall, _ = precision_recall_curve(y_test, m["proba"])
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                  name=f"{name} (AP={m['AP']:.3f})"))
    fig.update_layout(title="Precision-Recall Curve Comparison",
                      xaxis_title="Recall", yaxis_title="Precision",
                      height=500, margin=dict(l=40, r=40, t=60, b=40))
    return fig


# ============================================================
#  UI
# ============================================================
CLIENT_NAME         = _get_secret("CLIENT_NAME", "Customer")
APP_PASSWORD        = _get_secret("APP_PASSWORD", "")
DEFAULT_MONTHLY_COL = _get_secret("MONTHLY_COL", "MonthlyCharges")

st.set_page_config(page_title=f"{CLIENT_NAME} Churn Dashboard", layout="wide")

if not _check_password(APP_PASSWORD):
    st.stop()

st.title(f"{CLIENT_NAME} - Customer Churn Dashboard")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose data source:",
                      ["Upload CSV", "Demo - IBM Telco", "Demo - Fallback mirror", "Local file"])
    uploaded_file = None
    local_path = "./data/Telco-Customer-Churn.csv"
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    elif source == "Local file":
        local_path = st.text_input("Local CSV path", value=local_path)

    st.header("Train / Test Split")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    st.header("Models")
    all_models = ["Random Forest", "Logistic Regression", "XGBoost", "LightGBM",
                  "CatBoost", "Gradient Boosting", "AdaBoost"]
    choose_models = st.multiselect("Select models", all_models,
                                   default=["Random Forest", "XGBoost", "LightGBM"])
    model_params = {}
    if choose_models:
        st.header("Hyperparameters")
        for model_name in choose_models:
            with st.expander(f"{model_name} Parameters", expanded=False):
                if model_name == "Random Forest":
                    model_params["rf_n_estimators"]      = st.slider("n_estimators", 50, 500, 200, 50, key="rf_n")
                    model_params["rf_max_depth"]          = st.slider("max_depth (0=None)", 0, 30, 10, 1, key="rf_d")
                    model_params["rf_min_samples_split"]  = st.slider("min_samples_split", 2, 20, 2, 1, key="rf_m")
                elif model_name == "Logistic Regression":
                    model_params["lr_C"]        = st.slider("C", 0.01, 10.0, 1.0, 0.1, key="lr_c")
                    model_params["lr_max_iter"] = st.slider("max_iter", 100, 2000, 1000, 100, key="lr_i")
                    model_params["lr_solver"]   = st.selectbox("solver", ["lbfgs", "liblinear", "saga"], key="lr_s")
                elif model_name == "XGBoost":
                    model_params["xgb_n_estimators"]  = st.slider("n_estimators", 50, 500, 200, 50, key="xgb_n")
                    model_params["xgb_max_depth"]      = st.slider("max_depth", 3, 15, 6, 1, key="xgb_d")
                    model_params["xgb_learning_rate"]  = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01, key="xgb_l")
                elif model_name == "LightGBM":
                    model_params["lgb_n_estimators"]  = st.slider("n_estimators", 50, 500, 200, 50, key="lgb_n")
                    model_params["lgb_max_depth"]      = st.slider("max_depth", 3, 15, 6, 1, key="lgb_d")
                    model_params["lgb_learning_rate"]  = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01, key="lgb_l")
                elif model_name == "CatBoost":
                    model_params["cat_iterations"]    = st.slider("iterations", 50, 500, 200, 50, key="cat_i")
                    model_params["cat_depth"]          = st.slider("depth", 3, 10, 6, 1, key="cat_d")
                    model_params["cat_learning_rate"]  = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01, key="cat_l")
                elif model_name == "Gradient Boosting":
                    model_params["gb_n_estimators"]  = st.slider("n_estimators", 50, 500, 200, 50, key="gb_n")
                    model_params["gb_max_depth"]      = st.slider("max_depth", 3, 15, 6, 1, key="gb_d")
                    model_params["gb_learning_rate"]  = st.slider("learning_rate", 0.01, 0.3, 0.1, 0.01, key="gb_l")
                elif model_name == "AdaBoost":
                    model_params["ada_n_estimators"]  = st.slider("n_estimators", 50, 500, 200, 50, key="ada_n")
                    model_params["ada_learning_rate"] = st.slider("learning_rate", 0.1, 2.0, 1.0, 0.1, key="ada_l")

    st.header("Decision Threshold")
    threshold = st.slider("Classification threshold", 0.1, 0.9, 0.5, 0.01)
    st.header("Run")
    run_btn = st.button("Run Training")

# Load data
if source == "Upload CSV" and uploaded_file is None:
    st.info("Upload a CSV file using the sidebar to get started.")
    st.stop()

try:
    ub = uploaded_file.read() if uploaded_file else None
    df, source_str = load_df(source, local_path if source == "Local file" else None, ub)
    st.success(f"Loaded {len(df):,} rows from: {source_str}")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Column mapping
st.subheader("Column Mapping")
all_cols = df.columns.tolist()
obj_cols = df.select_dtypes(include="object").columns.tolist()

col_a, col_b = st.columns(2)
with col_a:
    default_target = "Churn" if "Churn" in all_cols else (obj_cols[0] if obj_cols else all_cols[0])
    target_col = st.selectbox("Target column (what to predict)", all_cols,
                               index=all_cols.index(default_target))
with col_b:
    id_candidates = [c for c in all_cols if "id" in c.lower()]
    default_id    = id_candidates[0] if id_candidates else all_cols[0]
    id_options    = ["(none)"] + all_cols
    id_col_sel    = st.selectbox("Customer ID column (will be dropped)", id_options,
                                  index=id_options.index(default_id) if default_id in id_options else 0)
    id_col = None if id_col_sel == "(none)" else id_col_sel

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
if target_col in df.columns:
    top_val = df[target_col].value_counts().idxmax()
    c3.metric("Event rate", f"{(~df[target_col].isin([top_val])).mean():.1%}")

# EDA
st.markdown("### Exploratory Analysis")
eda_default = [c for c in ["Contract", "InternetService", "PaymentMethod"]
               if c in obj_cols and c != target_col]
eda_cols = st.multiselect("Break down target by:",
                           options=[c for c in obj_cols if c != target_col],
                           default=eda_default)
for c in eda_cols:
    tmp = df.groupby([c, target_col]).size().reset_index(name="count")
    fig = px.bar(tmp, x=c, y="count", color=target_col, barmode="group",
                 title=f"{target_col} by {c}")
    st.plotly_chart(fig, use_container_width=True)

# Preprocess
try:
    df_processed, encoders, target_encoder = preprocess(df, target_col, id_col)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
params = {"models": choose_models, "scaler": scaler, **model_params}

# Training
if run_btn:
    if not choose_models:
        st.warning("Select at least one model.")
        st.stop()

    with st.spinner("Training models..."):
        metrics = train_models(X_train, X_test, y_train, y_test, params)

    st.markdown("## Model Performance Comparison")
    comp = pd.DataFrame([
        {"Model": n, "AUC": f"{m['AUC']:.4f}", "Accuracy": f"{m['Accuracy']:.4f}",
         "Precision": f"{m['Precision']:.4f}", "Recall": f"{m['Recall']:.4f}",
         "F1": f"{m['F1']:.4f}", "Avg Precision": f"{m['AP']:.4f}"}
        for n, m in metrics.items()
    ]).sort_values("AUC", ascending=False).reset_index(drop=True)
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("### Individual Model Metrics")
    for name, m in metrics.items():
        with st.expander(f"{name} Details"):
            cols = st.columns(5)
            for i, (k, v) in enumerate([("AUC", m["AUC"]), ("Accuracy", m["Accuracy"]),
                                         ("Precision", m["Precision"]), ("Recall", m["Recall"]),
                                         ("F1", m["F1"])]):
                cols[i].metric(k, f"{v:.3f}")

    best_name = max(metrics, key=lambda k: metrics[k]["AUC"])
    best = metrics[best_name]
    st.success(f"Best model by AUC: **{best_name}** (AUC={best['AUC']:.3f})")

    st.markdown("## Performance Curves")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_roc_comparison(metrics, y_test), use_container_width=True)
    with col2:
        st.plotly_chart(plot_pr_comparison(metrics, y_test), use_container_width=True)

    st.markdown(f"## Best Model Analysis: {best_name}")
    proba = best["proba"]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Confusion Matrix (threshold={threshold:.2f})")
        st.plotly_chart(plot_confusion(cm), use_container_width=True)
    with col2:
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(
            classification_report(y_test, preds, output_dict=True)).transpose()
        valid_rows = [c for c in ["0", "1", "macro avg", "weighted avg"] if c in report_df.index]
        st.dataframe(
            report_df.loc[valid_rows, ["precision", "recall", "f1-score", "support"]]
            .style.format("{:.3f}", subset=["precision", "recall", "f1-score"]),
            use_container_width=True)

    st.markdown("### Feature Importance / Coefficients")
    if hasattr(best["model"], "feature_importances_"):
        imp_df = pd.DataFrame({"feature": X.columns,
                                "importance": best["model"].feature_importances_}
                               ).sort_values("importance", ascending=False)
        st.plotly_chart(px.bar(imp_df.head(20), x="importance", y="feature",
                                orientation="h", title=f"Top 20 Features - {best_name}"),
                        use_container_width=True)
    elif hasattr(best["model"], "coef_"):
        coef_df = pd.DataFrame({"feature": X.columns,
                                 "coefficient": best["model"].coef_.ravel()}
                                ).sort_values("coefficient")
        st.plotly_chart(px.bar(coef_df, x="coefficient", y="feature",
                                orientation="h", title=f"Coefficients - {best_name}"),
                        use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

    # Business Impact
    st.markdown("## Business Impact (what-if)")
    num_cols = [c for c in df.columns if df[c].dtype in ["float64", "int64"]]
    default_mc_idx = next((i for i, c in enumerate(num_cols) if c == DEFAULT_MONTHLY_COL), 0)
    monthly_col    = st.selectbox("Monthly revenue column", num_cols, index=default_mc_idx)
    monthly_rev    = float(df[monthly_col].mean()) if monthly_col else 70.0
    retention_cost = st.number_input("Retention cost per flagged customer ($)",
                                      value=50.0, min_value=0.0, step=5.0)
    success_rate   = st.slider("Retention success rate", 0.0, 1.0, 0.70, 0.05)
    tn, fp, fn, tp = cm.ravel()
    net_benefit    = (tp * success_rate * monthly_rev * 12) - ((tp + fp) * retention_cost)
    cols = st.columns(4)
    cols[0].metric("TP (saved)", tp)
    cols[1].metric("FP (over-treatment)", fp)
    cols[2].metric("FN (missed)", fn)
    cols[3].metric("Net benefit ($)", f"{net_benefit:,.0f}")

    # Sample predictions
    st.markdown("## Predict on Sample Customers")
    n_samples = st.slider("Random test customers", 1, 20, 5, 1)
    idx = np.random.choice(len(X_test), size=n_samples, replace=False)
    sample = X_test.iloc[idx]
    if best_name == "Logistic Regression":
        p = best["model"].predict_proba(scaler.transform(sample))[:, 1]
    else:
        p = best["model"].predict_proba(sample)[:, 1]
    sample_out = sample.copy()
    sample_out["churn_prob"] = p
    sample_out["prediction"] = np.where(sample_out["churn_prob"] >= threshold,
                                         "High Risk", "Low Risk")
    st.dataframe(sample_out, use_container_width=True)
    csv_buf = io.StringIO()
    sample_out.to_csv(csv_buf, index=False)
    st.download_button("Download predictions CSV", csv_buf.getvalue(),
                        file_name="predictions.csv", mime="text/csv")

else:
    st.info("Configure options in the sidebar and click **Run Training** to begin.")
