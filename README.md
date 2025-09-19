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

## 🧪 Notes

- No secrets are required. If you add APIs, put keys in `.streamlit/secrets.toml` (never commit).
- The app encodes categoricals and trains **Random Forest** and **Logistic Regression** with sliders for parameters and threshold.
- For large/private data, switch to **Local file** in the sidebar.
