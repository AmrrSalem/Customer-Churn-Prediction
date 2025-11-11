# Telco Customer Churn — Streamlit Dashboard

Interactive churn modeling demo using the IBM **Telco Customer Churn** dataset.

## 🚀 Quickstart (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

By default the app loads the dataset from the IBM GitHub URL. To use a local CSV,
set the sidebar to **Local file** and point to `./data/Telco-Customer-Churn.csv`.

## 🌐 Deploy on Streamlit Community Cloud

1. Create a new GitHub repo (e.g., `P-02-telco-churn`).
2. Push this folder (see commands below).
3. Go to https://share.streamlit.io, connect your repo, and select `app.py` as the app file.
4. The app will build using `requirements.txt` automatically.

## 📁 Project structure

```
P#02/
├─ app.py
├─ requirements.txt
├─ .gitignore
├─ .streamlit/
│  └─ config.toml
└─ data/
   └─ .gitkeep
```

## 🗂 Data sources

- IBM GitHub raw CSV (default):  
  `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

- Fallback mirror CSV:  
  `https://raw.githubusercontent.com/fenago/datasets/main/WA_Fn-UseC_-Telco-Customer-Churn.csv`

You can also place a local CSV at `data/Telco-Customer-Churn.csv`.

## ⬆️ GitHub push

```bash
cd "P#02"
git init
git add .
git commit -m "P#02: Streamlit churn dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## ✨ Features

### Machine Learning Models (7 Total)
- **Random Forest** - Ensemble decision tree classifier
- **Logistic Regression** - Linear baseline model
- **XGBoost** - Extreme gradient boosting
- **LightGBM** - Fast gradient boosting framework
- **CatBoost** - Gradient boosting optimized for categorical features
- **Gradient Boosting** - Classic gradient boosting classifier
- **AdaBoost** - Adaptive boosting ensemble

### Advanced Feature Engineering
- **Tenure groups** - Customer lifecycle segments (0-12, 12-24, 24-48, 48+ months)
- **Charge ratios** - Monthly to total charges, charges per tenure
- **Service combinations** - Internet + phone, security + backup interactions
- **Contract risk scores** - Ordinal encoding based on business logic
- **Total services count** - Number of subscribed services
- **Payment method risk** - Electronic check flagging

### Performance Metrics & Visualizations
- **Model Comparison Table** - Side-by-side metrics for all models
- **ROC Curve Comparison** - AUC scores across all models
- **Precision-Recall Curves** - Especially useful for imbalanced churn data
- **Classification Reports** - Detailed precision, recall, F1-scores
- **Feature Importance** - Top predictive features visualization
- **Confusion Matrix** - True/false positives and negatives
- **Business Impact Analysis** - What-if scenario modeling

### Interactive Controls
- Model selection (train single or multiple models)
- Train/test split ratio adjustment
- Random Forest hyperparameter tuning
- Classification threshold slider
- Sample predictions with download

## 🧪 Notes

- No secrets are required. If you add APIs, put keys in `.streamlit/secrets.toml` (never commit).
- The app performs advanced feature engineering and supports 7 different ML models.
- For large/private data, switch to **Local file** in the sidebar.
- Default models selected: Random Forest, XGBoost, LightGBM (best performers for tabular data)
