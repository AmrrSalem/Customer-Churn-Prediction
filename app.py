#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customer Churn Prediction - Product Edition
--------------------------------------------
Three-tab product:
  1. Train    — upload CSV, map columns, train 7 ML models, save model bundle
  2. Score    — load saved model, upload new customers, get churn predictions
  3. Alert    — email high-risk customer list to stakeholders

Client config via .streamlit/secrets.toml:
    CLIENT_NAME  = "Acme Corp"
    APP_PASSWORD = "secret123"
    MONTHLY_COL  = "MonthlyCharges"
    SMTP_HOST    = "smtp.gmail.com"
    SMTP_PORT    = "587"
    SMTP_USER    = "you@gmail.com"
    SMTP_PASS    = "app-password"
"""
from pathlib import Path
import io, hashlib, pickle, smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

try:
    from supabase import create_client as _sb_create
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False

try:
    import shap as _shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

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
                              f1_score, average_precision_score,
                              precision_recall_curve, classification_report)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

PRIMARY_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
               "/master/data/Telco-Customer-Churn.csv")
FALLBACK_URL = ("https://raw.githubusercontent.com/fenago/datasets/main"
                "/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# ── Secrets ───────────────────────────────────────────────────────────────────
def _s(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ── Supabase Storage ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_supabase():
    if not _SUPABASE_AVAILABLE:
        return None
    url = _s("SUPABASE_URL")
    key = _s("SUPABASE_KEY")
    if not url or not key:
        return None
    try:
        return _sb_create(url, key)
    except Exception:
        return None

def _sb_bucket():
    return _s("SUPABASE_BUCKET", "model-bundles")

def _sb_upload(data: bytes, filename: str) -> bool:
    sb = _get_supabase()
    if not sb:
        return False
    try:
        sb.storage.from_(_sb_bucket()).upload(
            filename, data, {"content-type": "application/octet-stream"}
        )
        return True
    except Exception as e:
        # If file exists, overwrite via update
        try:
            sb.storage.from_(_sb_bucket()).update(
                filename, data, {"content-type": "application/octet-stream"}
            )
            return True
        except Exception:
            st.warning(f"Cloud upload failed: {e}")
            return False

def _sb_list() -> list:
    sb = _get_supabase()
    if not sb:
        return []
    try:
        items = sb.storage.from_(_sb_bucket()).list()
        return sorted(
            [i["name"] for i in items if i.get("name", "").endswith(".pkl")],
            reverse=True,
        )
    except Exception:
        return []

def _sb_download(filename: str):
    sb = _get_supabase()
    if not sb:
        return None
    try:
        return sb.storage.from_(_sb_bucket()).download(filename)
    except Exception as e:
        st.error(f"Cloud download failed: {e}")
        return None

def _sb_delete(filename: str) -> bool:
    sb = _get_supabase()
    if not sb:
        return False
    try:
        sb.storage.from_(_sb_bucket()).remove([filename])
        return True
    except Exception:
        return False


# ── Auth ──────────────────────────────────────────────────────────────────────
def _check_password(pwd):
    if not pwd:
        return True
    h = hashlib.sha256(pwd.encode()).hexdigest()
    if st.session_state.get("auth"):
        return True
    with st.form("login"):
        st.subheader("Login")
        entered = st.text_input("Password", type="password")
        ok = st.form_submit_button("Enter")
    if ok:
        if hashlib.sha256(entered.encode()).hexdigest() == h:
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(source, path_or_url=None, uploaded_bytes=None):
    na = [" ", ""]
    if source == "Upload CSV":
        return pd.read_csv(io.BytesIO(uploaded_bytes), na_values=na), "uploaded file"
    if source == "Local file":
        p = Path(path_or_url).expanduser().resolve()
        return pd.read_csv(p, na_values=na), str(p)
    if source == "Demo - IBM Telco":
        return pd.read_csv(PRIMARY_URL, na_values=na), PRIMARY_URL
    return pd.read_csv(FALLBACK_URL, na_values=na), FALLBACK_URL


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    if "tenure" in df.columns:
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=[0, 12, 24, 48, float("inf")],
            labels=["0-12", "12-24", "24-48", "48+"]).astype(str)
    if all(c in df.columns for c in ["TotalCharges", "MonthlyCharges", "tenure"]):
        df["charges_per_tenure"]     = df["TotalCharges"] / (df["tenure"] + 1)
        df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
        for c in ["charges_per_tenure", "monthly_to_total_ratio"]:
            df[c] = df[c].replace([float("inf"), -float("inf")], 0).fillna(0)
    if all(c in df.columns for c in ["InternetService", "PhoneService"]):
        df["has_internet_and_phone"] = (
            (df["InternetService"] != "No") & (df["PhoneService"] == "Yes")).astype(int)
    if all(c in df.columns for c in ["OnlineSecurity", "OnlineBackup"]):
        df["security_and_backup"] = (
            (df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes")).astype(int)
    svc = [c for c in ["PhoneService","InternetService","OnlineSecurity","OnlineBackup",
                        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
           if c in df.columns]
    if svc:
        df["total_services"] = df[svc].apply(
            lambda r: sum(1 for v in r if v == "Yes"), axis=1)
    if "Contract" in df.columns:
        df["contract_risk"] = df["Contract"].map(
            {"Month-to-month": 2, "One year": 1, "Two year": 0}).fillna(1)
    if "PaymentMethod" in df.columns:
        df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    return df


# ── Training preprocessing ────────────────────────────────────────────────────
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
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != target_col]
    label_encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        label_encoders[c] = le
    te = LabelEncoder()
    df[target_col] = te.fit_transform(df[target_col].astype(str))
    return df, label_encoders, te


# ── Scoring preprocessing (uses saved encoders) ───────────────────────────────
def preprocess_for_scoring(df_new, bundle):
    df = df_new.copy()
    df = engineer_features(df)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Extract IDs before dropping
    id_col = bundle.get("id_col")
    ids = df[id_col].values if id_col and id_col in df.columns else np.arange(len(df))
    if id_col and id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)

    # Drop target if accidentally included
    tc = bundle.get("target_col", "")
    if tc and tc in df.columns:
        df.drop(columns=[tc], inplace=True)

    # Apply saved label encoders
    for col, le in bundle["label_encoders"].items():
        if col in df.columns:
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # Encode remaining string columns (unseen by training)
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Align columns to training feature set
    for col in bundle["feature_cols"]:
        if col not in df.columns:
            df[col] = 0
    return df[bundle["feature_cols"]], ids


# ── Model training ────────────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, params):
    models, scaler = {}, params["scaler"]
    def _fit_predict(model, scaled=False):
        Xt = scaler.transform(X_train) if scaled else X_train
        Xte = scaler.transform(X_test)  if scaled else X_test
        model.fit(Xt, y_train)
        return model, model.predict_proba(Xte)[:, 1]

    specs = {
        "Random Forest":    (RandomForestClassifier(
            n_estimators=params.get("rf_n_estimators",200),
            max_depth=params.get("rf_max_depth",10) or None,
            min_samples_split=params.get("rf_min_samples_split",2), random_state=42), False),
        "Logistic Regression": (LogisticRegression(
            C=params.get("lr_C",1.0), max_iter=params.get("lr_max_iter",1000),
            solver=params.get("lr_solver","lbfgs"), random_state=42), True),
        "XGBoost":          (xgb.XGBClassifier(
            n_estimators=params.get("xgb_n_estimators",200),
            max_depth=params.get("xgb_max_depth",6),
            learning_rate=params.get("xgb_learning_rate",0.1),
            random_state=42, eval_metric="logloss"), False),
        "LightGBM":         (lgb.LGBMClassifier(
            n_estimators=params.get("lgb_n_estimators",200),
            max_depth=params.get("lgb_max_depth",6),
            learning_rate=params.get("lgb_learning_rate",0.1),
            random_state=42, verbose=-1), False),
        "CatBoost":         (CatBoostClassifier(
            iterations=params.get("cat_iterations",200),
            depth=params.get("cat_depth",6),
            learning_rate=params.get("cat_learning_rate",0.1),
            random_state=42, verbose=0), False),
        "Gradient Boosting":(GradientBoostingClassifier(
            n_estimators=params.get("gb_n_estimators",200),
            max_depth=params.get("gb_max_depth",6),
            learning_rate=params.get("gb_learning_rate",0.1), random_state=42), False),
        "AdaBoost":         (AdaBoostClassifier(
            n_estimators=params.get("ada_n_estimators",200),
            learning_rate=params.get("ada_learning_rate",1.0), random_state=42), False),
    }
    metrics = {}
    for name in params["models"]:
        if name not in specs:
            continue
        m, scaled = specs[name]
        model, proba = _fit_predict(m, scaled)
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


# ── Email alert ───────────────────────────────────────────────────────────────
def send_alert_email(high_risk_df, recipient, smtp_cfg, client_name, threshold):
    n = len(high_risk_df)
    subject = f"[{client_name}] Churn Alert: {n} High-Risk Customers"
    rows = high_risk_df.head(25).to_html(index=False, border=0,
                                          float_format=lambda x: f"{x:.1%}" if x < 2 else f"{x:.0f}")
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:700px">
      <h2 style="color:#d62728">{client_name} — Churn Risk Alert</h2>
      <p><strong>{n} customers</strong> exceed the risk threshold of
         <strong>{threshold:.0%}</strong> and require immediate attention.</p>
      <p style="color:#555">Scored on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
      {rows}
      <hr/>
      <p style="font-size:11px;color:#888">
        Sent by {client_name} Churn Dashboard &nbsp;|&nbsp; Do not reply to this message.
      </p>
    </div>"""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_cfg.get("from_addr", smtp_cfg["user"])
    msg["To"]      = recipient
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(smtp_cfg["host"], int(smtp_cfg["port"])) as s:
        s.starttls()
        s.login(smtp_cfg["user"], smtp_cfg["password"])
        s.sendmail(smtp_cfg["user"], recipient, msg.as_string())


# ── Prescriptive action rules ─────────────────────────────────────────────────
def recommend_action(customer_id, prob, risk_level, df_raw, id_col):
    """Rule-based action recommendation enriched with original feature data."""
    contract, tenure = "", 0
    if df_raw is not None and id_col and id_col in df_raw.columns:
        row = df_raw[df_raw[id_col].astype(str) == str(customer_id)]
        if not row.empty:
            r = row.iloc[0]
            contract = str(r.get("Contract", r.get("contract", ""))).lower()
            tenure   = float(r.get("tenure", r.get("Years", r.get("years", 0))) or 0)

    if risk_level == "Low Risk":
        return "Monitor — schedule quarterly check-in"

    if risk_level == "Medium Risk":
        if tenure < 12:
            return "New customer at risk — assign onboarding support"
        return "Send personalised retention offer via email"

    # High Risk
    if "month" in contract or contract in ("", "nan"):
        if prob >= 0.70:
            return "Urgent: escalate to account manager + offer annual plan (20% off)"
        return "Offer annual plan upgrade with incentive"
    if "one" in contract:
        return "Offer two-year renewal with loyalty discount"
    if tenure < 12:
        return "High-risk new customer — dedicated success manager"
    if prob >= 0.70:
        return "Urgent: offer retention package + executive call"
    return "Proactive outreach — loyalty reward or service upgrade"


# ── Chart helpers ─────────────────────────────────────────────────────────────
def plot_confusion(cm, labels=("Retained", "Churned")):
    z = np.array(cm)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Pred {labels[0]}", f"Pred {labels[1]}"],
        y=[f"Actual {labels[0]}", f"Actual {labels[1]}"],
        text=z, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(height=350, margin=dict(l=40,r=40,t=40,b=40))
    return fig

def plot_roc_comparison(metrics, y_test):
    fig = go.Figure()
    for name, m in metrics.items():
        fpr, tpr, _ = roc_curve(y_test, m["proba"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"{name} (AUC={m['AUC']:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance",
                              line=dict(dash="dash", color="gray")))
    fig.update_layout(title="ROC Comparison", xaxis_title="FPR", yaxis_title="TPR",
                      height=450, margin=dict(l=40,r=40,t=60,b=40))
    return fig

def plot_pr_comparison(metrics, y_test):
    fig = go.Figure()
    for name, m in metrics.items():
        p, r, _ = precision_recall_curve(y_test, m["proba"])
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines",
                                  name=f"{name} (AP={m['AP']:.3f})"))
    fig.update_layout(title="Precision-Recall Comparison",
                      xaxis_title="Recall", yaxis_title="Precision",
                      height=450, margin=dict(l=40,r=40,t=60,b=40))
    return fig


def get_shap_explainer(model, model_name, X_background):
    """Return a SHAP explainer appropriate for the model type."""
    if not _SHAP_AVAILABLE:
        return None
    try:
        if model_name == "Logistic Regression":
            return _shap.LinearExplainer(model, X_background)
        return _shap.TreeExplainer(model)
    except Exception:
        try:
            bg = _shap.sample(X_background, min(50, len(X_background)))
            return _shap.KernelExplainer(model.predict_proba, bg)
        except Exception:
            return None


def shap_values_class1(explainer, X, model_name):
    """Extract SHAP values for the positive class."""
    sv = explainer.shap_values(X)
    if isinstance(sv, list):   # tree models return [class0, class1]
        return sv[1]
    return sv                  # linear / kernel return single array


def plot_shap_bar(mean_abs_shap, feature_names, title="Global Feature Impact (SHAP)"):
    df_s = pd.DataFrame({"feature": feature_names,
                          "importance": mean_abs_shap}
                        ).sort_values("importance").tail(20)
    fig = px.bar(df_s, x="importance", y="feature", orientation="h",
                 title=title, color="importance", color_continuous_scale="Reds")
    fig.update_layout(height=500, coloraxis_showscale=False,
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def plot_shap_waterfall(shap_vals_row, feature_names, base_value, customer_id):
    """Plotly waterfall for a single customer's SHAP values."""
    sv   = shap_vals_row
    idx  = np.argsort(np.abs(sv))[-12:]          # top 12 features
    vals = sv[idx]
    feats = [feature_names[i] for i in idx]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h",
                           marker_color=colors))
    fig.update_layout(
        title=f"SHAP Explanation — Customer: {customer_id}",
        xaxis_title="SHAP value (impact on churn probability)",
        height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
CLIENT_NAME  = _s("CLIENT_NAME",  "Customer")
APP_PASSWORD = _s("APP_PASSWORD", "")
DEFAULT_MC   = _s("MONTHLY_COL",  "MonthlyCharges")

st.set_page_config(page_title=f"{CLIENT_NAME} Churn Dashboard",
                   page_icon="📉", layout="wide")

if not _check_password(APP_PASSWORD):
    st.stop()

st.title(f"{CLIENT_NAME} — Customer Churn Dashboard")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source")
    source = st.radio("Source:", ["Upload CSV", "Demo - IBM Telco",
                                   "Demo - Fallback mirror", "Local file"])
    uploaded_file, local_path = None, "./data/Telco-Customer-Churn.csv"
    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload training CSV", type=["csv"])
    elif source == "Local file":
        local_path = st.text_input("Local CSV path", value=local_path)

    st.header("Train / Test Split")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    st.header("Models")
    ALL_MODELS = ["Random Forest","Logistic Regression","XGBoost","LightGBM",
                  "CatBoost","Gradient Boosting","AdaBoost"]
    choose_models = st.multiselect("Select models", ALL_MODELS,
                                   default=["Random Forest","XGBoost","LightGBM"])
    model_params = {}
    if choose_models:
        st.header("Hyperparameters")
        for mn in choose_models:
            with st.expander(f"{mn}", expanded=False):
                if mn == "Random Forest":
                    model_params["rf_n_estimators"]     = st.slider("n_estimators",50,500,200,50,key="rf_n")
                    model_params["rf_max_depth"]         = st.slider("max_depth (0=None)",0,30,10,1,key="rf_d")
                    model_params["rf_min_samples_split"] = st.slider("min_samples_split",2,20,2,1,key="rf_m")
                elif mn == "Logistic Regression":
                    model_params["lr_C"]        = st.slider("C",0.01,10.0,1.0,0.1,key="lr_c")
                    model_params["lr_max_iter"] = st.slider("max_iter",100,2000,1000,100,key="lr_i")
                    model_params["lr_solver"]   = st.selectbox("solver",["lbfgs","liblinear","saga"],key="lr_s")
                elif mn == "XGBoost":
                    model_params["xgb_n_estimators"]  = st.slider("n_estimators",50,500,200,50,key="xgb_n")
                    model_params["xgb_max_depth"]      = st.slider("max_depth",3,15,6,1,key="xgb_d")
                    model_params["xgb_learning_rate"]  = st.slider("learning_rate",0.01,0.3,0.1,0.01,key="xgb_l")
                elif mn == "LightGBM":
                    model_params["lgb_n_estimators"]  = st.slider("n_estimators",50,500,200,50,key="lgb_n")
                    model_params["lgb_max_depth"]      = st.slider("max_depth",3,15,6,1,key="lgb_d")
                    model_params["lgb_learning_rate"]  = st.slider("learning_rate",0.01,0.3,0.1,0.01,key="lgb_l")
                elif mn == "CatBoost":
                    model_params["cat_iterations"]    = st.slider("iterations",50,500,200,50,key="cat_i")
                    model_params["cat_depth"]          = st.slider("depth",3,10,6,1,key="cat_d")
                    model_params["cat_learning_rate"]  = st.slider("learning_rate",0.01,0.3,0.1,0.01,key="cat_l")
                elif mn == "Gradient Boosting":
                    model_params["gb_n_estimators"]  = st.slider("n_estimators",50,500,200,50,key="gb_n")
                    model_params["gb_max_depth"]      = st.slider("max_depth",3,15,6,1,key="gb_d")
                    model_params["gb_learning_rate"]  = st.slider("learning_rate",0.01,0.3,0.1,0.01,key="gb_l")
                elif mn == "AdaBoost":
                    model_params["ada_n_estimators"]  = st.slider("n_estimators",50,500,200,50,key="ada_n")
                    model_params["ada_learning_rate"] = st.slider("learning_rate",0.1,2.0,1.0,0.1,key="ada_l")

    st.header("Decision Threshold")
    threshold = st.slider("Threshold", 0.1, 0.9, 0.5, 0.01)
    st.header("Run")
    run_btn = st.button("Run Training", type="primary")

# ── Four product tabs ─────────────────────────────────────────────────────────
tab_train, tab_score, tab_alert, tab_explain = st.tabs(
    ["Train Model", "Score New Customers", "Email Alert", "Explain (SHAP)"])

# ════════════════════════════════
#  TAB 1 — TRAIN
# ════════════════════════════════
with tab_train:
    if source == "Upload CSV" and uploaded_file is None:
        st.info("Upload a CSV file in the sidebar to get started, or select a Demo source.")
        st.stop()

    try:
        ub = uploaded_file.read() if uploaded_file else None
        df, src_str = load_df(source, local_path if source == "Local file" else None, ub)
        st.success(f"Loaded {len(df):,} rows from: {src_str}")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Column mapping
    st.subheader("Column Mapping")
    all_cols = df.columns.tolist()
    obj_cols  = df.select_dtypes(include="object").columns.tolist()
    ca, cb = st.columns(2)
    with ca:
        default_tgt = "Churn" if "Churn" in all_cols else (obj_cols[0] if obj_cols else all_cols[0])
        target_col  = st.selectbox("Target column", all_cols,
                                    index=all_cols.index(default_tgt))
    with cb:
        id_cands  = [c for c in all_cols if "id" in c.lower()]
        default_id = id_cands[0] if id_cands else all_cols[0]
        id_opts    = ["(none)"] + all_cols
        id_sel     = st.selectbox("Customer ID column (dropped from training)",
                                   id_opts,
                                   index=id_opts.index(default_id) if default_id in id_opts else 0)
        id_col = None if id_sel == "(none)" else id_sel

    # Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    if target_col in df.columns:
        top_v = df[target_col].value_counts().idxmax()
        c3.metric("Event rate", f"{(~df[target_col].isin([top_v])).mean():.1%}")

    # EDA
    st.markdown("### Exploratory Analysis")
    eda_def = [c for c in ["Contract","InternetService","PaymentMethod"]
               if c in obj_cols and c != target_col]
    eda_cols = st.multiselect("Break down target by:",
                               options=[c for c in obj_cols if c != target_col],
                               default=eda_def)
    for c in eda_cols:
        tmp = df.groupby([c, target_col]).size().reset_index(name="count")
        st.plotly_chart(px.bar(tmp, x=c, y="count", color=target_col, barmode="group",
                                title=f"{target_col} by {c}"),
                        use_container_width=True)

    # Preprocess
    try:
        df_proc, encoders, target_encoder = preprocess(df, target_col, id_col)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    X = df_proc.drop(columns=[target_col])
    y = df_proc[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    params = {"models": choose_models, "scaler": scaler, **model_params}

    if run_btn:
        if not choose_models:
            st.warning("Select at least one model.")
            st.stop()
        with st.spinner("Training models..."):
            metrics = train_models(X_train, X_test, y_train, y_test, params)

        # Results table
        st.markdown("## Model Comparison")
        comp = pd.DataFrame([
            {"Model": n, "AUC": f"{m['AUC']:.4f}", "Accuracy": f"{m['Accuracy']:.4f}",
             "Precision": f"{m['Precision']:.4f}", "Recall": f"{m['Recall']:.4f}",
             "F1": f"{m['F1']:.4f}"}
            for n, m in metrics.items()
        ]).sort_values("AUC", ascending=False).reset_index(drop=True)
        st.dataframe(comp, use_container_width=True, hide_index=True)

        best_name = max(metrics, key=lambda k: metrics[k]["AUC"])
        best = metrics[best_name]
        st.success(f"Best model: **{best_name}** — AUC {best['AUC']:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_roc_comparison(metrics, y_test), use_container_width=True)
        with col2:
            st.plotly_chart(plot_pr_comparison(metrics, y_test), use_container_width=True)

        proba = best["proba"]
        preds = (proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Confusion Matrix (threshold={threshold:.2f})")
            st.plotly_chart(plot_confusion(cm), use_container_width=True)
        with col2:
            st.markdown("### Classification Report")
            rdf = pd.DataFrame(classification_report(y_test, preds, output_dict=True)).T
            valid = [r for r in ["0","1","macro avg","weighted avg"] if r in rdf.index]
            st.dataframe(rdf.loc[valid, ["precision","recall","f1-score","support"]]
                         .style.format("{:.3f}", subset=["precision","recall","f1-score"]),
                         use_container_width=True)

        # Feature importance
        if hasattr(best["model"], "feature_importances_"):
            imp = pd.DataFrame({"feature": X.columns,
                                 "importance": best["model"].feature_importances_}
                               ).sort_values("importance", ascending=False)
            st.plotly_chart(px.bar(imp.head(20), x="importance", y="feature",
                                    orientation="h", title=f"Feature Importance — {best_name}"),
                            use_container_width=True)

        # Business Impact
        st.markdown("## Business Impact")
        num_cols = [c for c in df.columns if df[c].dtype in ["float64","int64"]]
        def_idx  = next((i for i, c in enumerate(num_cols) if c == DEFAULT_MC), 0)
        mc_col   = st.selectbox("Monthly revenue column", num_cols, index=def_idx)
        monthly  = float(df[mc_col].mean()) if mc_col else 70.0
        ret_cost = st.number_input("Retention cost per customer ($)", value=50.0, step=5.0)
        succ     = st.slider("Retention success rate", 0.0, 1.0, 0.70, 0.05)
        tn, fp, fn, tp = cm.ravel()
        net = (tp * succ * monthly * 12) - ((tp + fp) * ret_cost)
        kpi = st.columns(4)
        kpi[0].metric("TP (saved)", tp)
        kpi[1].metric("FP (over-treatment)", fp)
        kpi[2].metric("FN (missed)", fn)
        kpi[3].metric("Net benefit ($)", f"{net:,.0f}")

        # ── Cohort / Time-series view ─────────────────────────────────────────
        date_col_cands = [c for c in df.columns
                          if any(kw in c.lower() for kw in
                                 ["date","time","month","year","period","signup","join"])]
        if date_col_cands:
            st.markdown("---")
            st.markdown("## Cohort Analysis")
            dc = st.selectbox("Date column", date_col_cands, key="cohort_date_col")
            df_coh = df.copy()
            df_coh["_date"] = pd.to_datetime(df_coh[dc], errors="coerce")
            df_coh = df_coh.dropna(subset=["_date"])
            if len(df_coh) > 10:
                df_coh["_month"] = df_coh["_date"].dt.to_period("M").astype(str)
                churn_val = df_coh[target_col].value_counts().index[-1]  # minority = churn
                monthly = (df_coh.groupby("_month")
                           .apply(lambda g: pd.Series({
                               "total":   len(g),
                               "churned": (g[target_col] == churn_val).sum()}))
                           .reset_index())
                monthly["churn_rate"] = monthly["churned"] / monthly["total"]
                fig_coh = px.line(monthly, x="_month", y="churn_rate",
                                  markers=True, title="Monthly Churn Rate Over Time",
                                  labels={"_month": "Month", "churn_rate": "Churn Rate"})
                fig_coh.update_yaxes(tickformat=".0%")
                fig_coh.update_layout(height=380, margin=dict(l=10,r=10,t=50,b=10))
                st.plotly_chart(fig_coh, use_container_width=True)

                # Cohort retention table
                df_coh["_cohort"] = df_coh["_date"].dt.to_period("Q").astype(str)
                cohort_stats = (df_coh.groupby("_cohort")
                                .apply(lambda g: pd.Series({
                                    "Customers": len(g),
                                    "Churned":   (g[target_col] == churn_val).sum(),
                                    "Churn Rate": f"{(g[target_col]==churn_val).mean():.1%}"}))
                                .reset_index().rename(columns={"_cohort": "Cohort"}))
                st.markdown("#### Quarterly Cohort Summary")
                st.dataframe(cohort_stats, use_container_width=True, hide_index=True)
            else:
                st.caption("Not enough rows with valid dates for cohort analysis.")

        # ── Save model bundle ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## Save Model")
        bundle = {
            "model":          best["model"],
            "model_name":     best_name,
            "scaler":         scaler,
            "label_encoders": encoders,
            "feature_cols":   X.columns.tolist(),
            "target_col":     target_col,
            "id_col":         id_col,
            "threshold":      threshold,
            "trained_at":     datetime.now().isoformat(),
            "client_name":    CLIENT_NAME,
            "metrics": {
                "AUC": best["AUC"], "F1": best["F1"],
                "Accuracy": best["Accuracy"], "Recall": best["Recall"],
            },
        }
        st.session_state["model_bundle"] = bundle
        # Save training sample for SHAP global analysis
        shap_sample = X_train.sample(min(200, len(X_train)), random_state=42)
        st.session_state["X_train_shap"]  = shap_sample
        st.session_state["train_df_raw"]  = df          # for cohort view

        buf = io.BytesIO()
        pickle.dump(bundle, buf)
        buf.seek(0)
        st.download_button(
            label=f"Download Model Bundle ({best_name})",
            data=buf,
            file_name=f"churn_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl",
            mime="application/octet-stream",
            help="Save this file — reload it in the Score tab to predict without retraining",
        )
        st.info("Model saved to session. Switch to **Score New Customers** tab to predict on new data.")

        # ── Auto-upload to Supabase ───────────────────────────────────────────
        if _s("SUPABASE_URL") and _s("SUPABASE_KEY"):
            safe_client = CLIENT_NAME.replace(" ", "_")
            safe_model  = best_name.replace(" ", "_")
            cloud_fname = (f"{safe_client}_{safe_model}_"
                           f"{datetime.now().strftime('%Y%m%d_%H%M')}.pkl")
            cloud_buf = io.BytesIO()
            pickle.dump(bundle, cloud_buf)
            with st.spinner("Saving to cloud storage..."):
                ok = _sb_upload(cloud_buf.getvalue(), cloud_fname)
            if ok:
                st.success(f"Saved to cloud: **{cloud_fname}**")
                st.session_state.pop("sb_bundle_list", None)  # force refresh

    else:
        st.info("Configure the sidebar and click **Run Training** to begin.")

# ════════════════════════════════
#  TAB 2 — SCORE NEW DATA
# ════════════════════════════════
with tab_score:
    st.subheader("Score New Customers")
    st.markdown("Upload a CSV with the same columns as your training data (no target column needed).")

    # Load model bundle from session or file
    bundle = st.session_state.get("model_bundle")

    # ── Cloud bundle selector ─────────────────────────────────────────────────
    sb_on = bool(_s("SUPABASE_URL") and _s("SUPABASE_KEY"))
    if sb_on:
        st.markdown("#### Cloud Models")
        cc1, cc2, cc3, cc4 = st.columns([1, 4, 1, 1])
        with cc1:
            do_refresh = st.button("Refresh", key="sb_refresh")
        if do_refresh or "sb_bundle_list" not in st.session_state:
            st.session_state["sb_bundle_list"] = _sb_list()
        bundle_list = st.session_state.get("sb_bundle_list", [])
        with cc2:
            sel = st.selectbox("Saved bundles", ["— select —"] + bundle_list,
                               key="sb_sel")
        with cc3:
            st.write("")
            do_load = st.button("Load", key="sb_load",
                                disabled=(sel == "— select —"))
        with cc4:
            st.write("")
            do_del = st.button("Delete", key="sb_del",
                               disabled=(sel == "— select —"))

        if do_load and sel != "— select —":
            with st.spinner(f"Downloading {sel}…"):
                raw = _sb_download(sel)
            if raw:
                try:
                    bundle = pickle.loads(raw)
                    st.session_state["model_bundle"] = bundle
                    st.success(f"Loaded from cloud: **{bundle['model_name']}** "
                               f"(AUC {bundle['metrics']['AUC']:.3f})")
                except Exception as e:
                    st.error(f"Could not parse bundle: {e}")

        if do_del and sel != "— select —":
            if _sb_delete(sel):
                st.session_state["sb_bundle_list"] = _sb_list()
                st.success(f"Deleted: {sel}")
                st.rerun()

        if not bundle_list:
            st.caption("No bundles found in cloud. Train a model and it will appear here.")
        st.markdown("---")

    # ── Session status + local upload ─────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if bundle:
            trained = bundle.get("trained_at", "unknown")[:16].replace("T", " ")
            st.success(f"Model in session: **{bundle['model_name']}** "
                       f"(AUC {bundle['metrics']['AUC']:.3f}) — trained {trained}")
        else:
            st.warning("No model loaded. Train a model, load from cloud, or upload a .pkl below.")
    with col_b:
        pkl_file = st.file_uploader("Or upload a saved .pkl model bundle", type=["pkl"])
        if pkl_file:
            try:
                bundle = pickle.load(pkl_file)
                st.session_state["model_bundle"] = bundle
                st.success(f"Loaded: **{bundle['model_name']}** "
                           f"(AUC {bundle['metrics']['AUC']:.3f})")
            except Exception as e:
                st.error(f"Could not load bundle: {e}")

    if not bundle:
        st.stop()

    # Model summary
    with st.expander("Model details", expanded=False):
        m_info = bundle["metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model", bundle["model_name"])
        mc2.metric("AUC",      f"{m_info['AUC']:.3f}")
        mc3.metric("F1",       f"{m_info['F1']:.3f}")
        mc4.metric("Accuracy", f"{m_info['Accuracy']:.3f}")
        score_thr = st.slider("Risk threshold", 0.1, 0.9,
                               float(bundle.get("threshold", 0.5)), 0.01,
                               key="score_thr")
    score_thr = st.session_state.get("score_thr", bundle.get("threshold", 0.5))

    new_file = st.file_uploader("Upload new customer CSV to score", type=["csv"],
                                 key="new_customers")
    if new_file is None:
        st.info("Upload a CSV of new customers to generate churn predictions.")
        st.stop()

    try:
        df_new = pd.read_csv(new_file, na_values=[" ", ""])
        st.success(f"Loaded {len(df_new):,} customers")
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    try:
        X_new, ids = preprocess_for_scoring(df_new, bundle)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # Predict
    model = bundle["model"]
    if bundle["model_name"] == "Logistic Regression":
        probas = model.predict_proba(bundle["scaler"].transform(X_new))[:, 1]
    else:
        probas = model.predict_proba(X_new)[:, 1]

    risk_labels = np.where(probas >= score_thr, "High Risk",
                           np.where(probas >= score_thr * 0.6, "Medium Risk", "Low Risk"))
    actions = [recommend_action(cid, p, r, df_new, bundle.get("id_col"))
               for cid, p, r in zip(ids, probas, risk_labels)]

    results = pd.DataFrame({
        "Customer ID":        ids,
        "Churn Probability":  probas,
        "Risk Level":         risk_labels,
        "Recommended Action": actions,
    }).sort_values("Churn Probability", ascending=False).reset_index(drop=True)

    # Save to session for Alert tab + Explain tab
    st.session_state["score_results"]   = results
    st.session_state["score_threshold"] = score_thr
    st.session_state["X_scored"]        = X_new
    st.session_state["score_ids"]       = ids
    st.session_state["df_scored_raw"]   = df_new

    # KPIs
    high  = (results["Risk Level"] == "High Risk").sum()
    med   = (results["Risk Level"] == "Medium Risk").sum()
    low   = (results["Risk Level"] == "Low Risk").sum()
    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("High Risk",   high,  delta=f"{high/len(results):.1%} of customers",
               delta_color="inverse")
    kc2.metric("Medium Risk", med,   delta=f"{med/len(results):.1%} of customers",
               delta_color="off")
    kc3.metric("Low Risk",    low,   delta=f"{low/len(results):.1%} of customers")

    # Distribution chart
    fig = px.histogram(results, x="Churn Probability", nbins=30,
                       color="Risk Level",
                       color_discrete_map={"High Risk":"#d62728",
                                           "Medium Risk":"#ff7f0e",
                                           "Low Risk":"#2ca02c"},
                       title="Churn Probability Distribution")
    fig.add_vline(x=score_thr, line_dash="dash", line_color="black",
                  annotation_text=f"Threshold {score_thr:.0%}")
    st.plotly_chart(fig, use_container_width=True)

    # Results table
    st.markdown("### Prediction Results")
    st.dataframe(
        results.style.background_gradient(subset=["Churn Probability"],
                                           cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True)

    # Download
    csv_out = results.to_csv(index=False)
    st.download_button("Download Predictions CSV", csv_out,
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv")

    st.info(f"**{high} high-risk customers identified.** Go to the **Email Alert** tab to notify stakeholders.")

# ════════════════════════════════
#  TAB 3 — EMAIL ALERT
# ════════════════════════════════
with tab_alert:
    st.subheader("Email High-Risk Customer Alert")

    results   = st.session_state.get("score_results")
    score_thr = st.session_state.get("score_threshold", 0.5)

    if results is None:
        st.info("Score customers first on the **Score New Customers** tab.")
        st.stop()

    high_risk = results[results["Risk Level"] == "High Risk"]
    st.metric("High-Risk Customers to Alert", len(high_risk))

    if high_risk.empty:
        st.success("No high-risk customers at the current threshold. Lower the threshold to include more.")
        st.stop()

    st.markdown("### Preview (top 10)")
    st.dataframe(high_risk.head(10), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Email Configuration")

    ea, eb = st.columns(2)
    with ea:
        recipient = st.text_input("Recipient email", placeholder="manager@company.com")
    with eb:
        st.markdown("**SMTP settings** (from secrets.toml or enter below)")
        use_secrets = bool(_s("SMTP_USER"))
        if use_secrets:
            st.success("SMTP credentials loaded from secrets.toml")
            smtp_cfg = {
                "host":      _s("SMTP_HOST", "smtp.gmail.com"),
                "port":      _s("SMTP_PORT", "587"),
                "user":      _s("SMTP_USER"),
                "password":  _s("SMTP_PASS"),
                "from_addr": _s("SMTP_FROM", _s("SMTP_USER")),
            }
        else:
            st.warning("No SMTP secrets found. Enter credentials below.")
            smtp_cfg = {
                "host":     st.text_input("SMTP Host", value="smtp.gmail.com"),
                "port":     st.text_input("SMTP Port", value="587"),
                "user":     st.text_input("SMTP Username", placeholder="you@gmail.com"),
                "password": st.text_input("SMTP Password", type="password"),
                "from_addr": "",
            }

    st.markdown("---")
    send_btn = st.button("Send Alert Email", type="primary",
                          disabled=not recipient or not smtp_cfg["user"])

    if send_btn:
        if not recipient:
            st.error("Enter a recipient email address.")
        elif not smtp_cfg["user"] or not smtp_cfg["password"]:
            st.error("SMTP credentials required.")
        else:
            try:
                with st.spinner("Sending email..."):
                    send_alert_email(high_risk, recipient, smtp_cfg,
                                     CLIENT_NAME, score_thr)
                st.success(f"Alert sent to **{recipient}** — {len(high_risk)} high-risk customers reported.")
            except Exception as e:
                st.error(f"Failed to send email: {e}")
                st.markdown("**Tip:** For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833), not your account password.")

    # Manual export fallback
    with st.expander("Or export alert as HTML (no SMTP needed)"):
        rows_html = high_risk.to_html(index=False, border=0)
        html_export = f"""<!DOCTYPE html><html><body>
        <h2>{CLIENT_NAME} — Churn Alert</h2>
        <p><strong>{len(high_risk)} high-risk customers</strong> at threshold {score_thr:.0%}</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        {rows_html}</body></html>"""
        st.download_button("Download Alert as HTML", html_export,
                            file_name="churn_alert.html", mime="text/html")

# ════════════════════════════════
#  TAB 4 — EXPLAIN (SHAP)
# ════════════════════════════════
with tab_explain:
    st.subheader("Explain Predictions with SHAP")

    if not _SHAP_AVAILABLE:
        st.error("SHAP is not installed. Add `shap>=0.44` to requirements.txt and redeploy.")
        st.stop()

    bundle = st.session_state.get("model_bundle")
    if not bundle:
        st.info("Train a model first on the **Train Model** tab.")
        st.stop()

    model      = bundle["model"]
    model_name = bundle["model_name"]
    feat_cols  = bundle["feature_cols"]

    # ── Global SHAP ────────────────────────────────────────────────────────────
    X_bg = st.session_state.get("X_train_shap")
    if X_bg is None:
        st.info("Retrain the model to enable SHAP analysis (training data needed for background).")
        st.stop()

    if model_name == "Logistic Regression":
        X_bg_scaled = bundle["scaler"].transform(X_bg)
        X_bg_for_explainer = pd.DataFrame(X_bg_scaled, columns=feat_cols)
    else:
        X_bg_for_explainer = X_bg

    with st.spinner("Computing SHAP values…"):
        explainer = get_shap_explainer(model, model_name, X_bg_for_explainer)
        if explainer is None:
            st.error("Could not create SHAP explainer for this model type.")
            st.stop()
        sv_global = shap_values_class1(explainer, X_bg_for_explainer, model_name)

    mean_shap = np.abs(sv_global).mean(axis=0)
    st.plotly_chart(plot_shap_bar(mean_shap, feat_cols,
                                   f"Global Feature Impact — {model_name}"),
                    use_container_width=True)
    st.caption("Mean |SHAP value| across training sample. Higher = more influential feature.")

    # ── Per-customer waterfall ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Individual Customer Explanation")

    X_scored = st.session_state.get("X_scored")
    score_ids = st.session_state.get("score_ids")
    results_ex = st.session_state.get("score_results")

    if X_scored is None or score_ids is None:
        st.info("Score customers on the **Score New Customers** tab to enable individual explanations.")
    else:
        if model_name == "Logistic Regression":
            X_sc_exp = pd.DataFrame(bundle["scaler"].transform(X_scored), columns=feat_cols)
        else:
            X_sc_exp = X_scored

        id_options = [str(i) for i in score_ids]
        sel_id = st.selectbox("Select customer to explain", id_options, key="explain_customer")
        sel_idx = id_options.index(sel_id)

        sv_row = shap_values_class1(explainer, X_sc_exp.iloc[[sel_idx]], model_name)
        if sv_row.ndim == 2:
            sv_row = sv_row[0]

        prob_row = (results_ex[results_ex["Customer ID"].astype(str) == sel_id]
                    ["Churn Probability"].values)
        prob_str = f"{prob_row[0]:.1%}" if len(prob_row) else "—"

        risk_row = (results_ex[results_ex["Customer ID"].astype(str) == sel_id]
                    ["Risk Level"].values)
        action_row = (results_ex[results_ex["Customer ID"].astype(str) == sel_id]
                      ["Recommended Action"].values)

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Customer", sel_id)
        ec2.metric("Churn Probability", prob_str)
        if len(risk_row):
            ec3.metric("Risk Level", risk_row[0])

        if len(action_row):
            color = "#d62728" if "Urgent" in action_row[0] else "#ff7f0e" if risk_row[0] == "Medium Risk" else "#2ca02c"
            st.markdown(f"**Recommended Action:** "
                        f"<span style='color:{color}'>{action_row[0]}</span>",
                        unsafe_allow_html=True)

        try:
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1]
        except Exception:
            base_val = 0.0

        st.plotly_chart(plot_shap_waterfall(sv_row, feat_cols, base_val, sel_id),
                        use_container_width=True)
        st.caption("Red bars push churn probability UP. Green bars push it DOWN.")
