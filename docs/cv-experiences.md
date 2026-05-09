# CV Experience Entries — Data Scientist / ML Engineer
> Generated from GitHub profile: github.com/amrrsalem

---

## Production ML Applications

---

### Customer Churn Prediction Platform
**Role:** Machine Learning Engineer  
**Duration:** Sep 2025 – Mar 2026 (6 months)  
**Stack:** Python · Streamlit · XGBoost · LightGBM · CatBoost · scikit-learn · SHAP · Supabase · Pandas

- Built an end-to-end churn prediction platform supporting 7 classifiers (Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost, Logistic Regression) with automated model selection optimized for Recall to minimize false negatives.
- Engineered 8+ domain-specific features (tenure cohorts, monthly-to-total charge ratios, contract risk scores, service-bundle flags) that improved model signal over raw features.
- Integrated SHAP explainability with three-tier fallback (TreeExplainer → LinearExplainer → KernelExplainer), delivering per-customer waterfall charts and global feature importance for non-technical stakeholders.
- Implemented Supabase cloud model storage (upload / download / delete `.pkl` bundles) enabling multi-session model persistence and team sharing.
- Automated retention action recommendations based on churn probability, contract type, and tenure; routed high-risk alerts (>70% probability) via HTML-formatted SMTP email to account managers.
- Deployed live on Streamlit Community Cloud with SHA-256 password authentication and a zero-credential-committed secrets architecture.

---

### Supply Chain Optimization Dashboard
**Role:** Data Scientist  
**Duration:** Sep 2025 – Mar 2026 (6 months)  
**Stack:** Python · Streamlit · Pandas · Plotly

- Designed an interactive inventory optimization dashboard enabling supply chain managers to monitor stock levels, demand trends, and reorder points in real time.
- Built an automated executive summary report generator that distills key KPIs (stockout rate, carrying cost, fill rate) into a single-page export for C-suite review.
- Deployed live on Streamlit Community Cloud; publicly accessible and used as a portfolio demonstration of applied operations research concepts.

---

### Sales Analytics Dashboard
**Role:** Data Analyst / Data Scientist  
**Duration:** Sep 2025 – Mar 2026 (6 months)  
**Stack:** Python · Streamlit · Pandas · Plotly

- Created a business intelligence dashboard tracking core sales KPIs including revenue by segment, conversion rates, cohort retention, and product performance across multiple time horizons.
- Designed interactive Plotly visualizations (waterfall charts, funnel analysis, time-series decomposition) enabling drill-down exploration without SQL or BI-tool expertise.
- Deployed live on Streamlit Community Cloud, demonstrating end-to-end delivery from raw CSV ingestion to published web application.

---

### Sentiment Analysis Tool
**Role:** NLP Engineer  
**Duration:** Sep 2025 – Mar 2026 (6 months)  
**Stack:** Python · Streamlit · scikit-learn / Transformers · REST API · Pandas

- Built a full NLP pipeline for sentiment classification covering data ingestion, text preprocessing, model training, evaluation (accuracy, F1, confusion matrix), and live inference.
- Served trained models via a REST API, decoupling the ML backend from the Streamlit frontend and enabling integration with external systems.
- Tracked and visualized evaluation metrics (precision, recall, ROC-AUC) within the app, enabling iterative model comparison without rerunning notebooks.
- Deployed live on Streamlit Community Cloud with public demo access.

---

### Recommender System App
**Role:** Data Scientist  
**Duration:** Aug 2025 – Mar 2026 (7 months)  
**Stack:** Python · Streamlit · Pandas · scikit-learn

- Implemented a data-driven recommendation engine using collaborative and/or content-based filtering techniques with an interactive Streamlit interface for live item recommendations.
- Exposed filtering controls and similarity parameters in the UI, allowing business users to tune recommendation behavior without code changes.

---

## Machine Learning Research & Engineering

---

### Reinforcement Learning Recommendation System (DS540)
**Role:** ML Researcher (Academic Project)  
**Duration:** Nov 2025 (intensive sprint)  
**Stack:** Python · PyTorch · Jupyter Notebook · Streamlit

- Designed and trained a reinforcement learning agent for sequential recommendation, formulating item selection as a Markov Decision Process (MDP) with a reward signal based on user engagement.
- Implemented the policy network in PyTorch with experience replay and target network stabilization; evaluated against collaborative filtering baselines.
- Packaged results in an interactive Streamlit demo showcasing learned recommendation sequences and reward curves.

---

### Production Data Science Package (dsnd_p02)
**Role:** Data Engineer / Software Engineer  
**Duration:** Aug 2025  
**Stack:** Python · SQLite · pytest

- Built a modular, pip-installable Python package applying software engineering best practices to a data science workflow: separation of concerns, clean public API, and SQLite-backed persistence.
- Wrote structured tests to validate data transformations and database operations, demonstrating production readiness beyond notebook-level prototyping.

---

### PySpark Local Development Environment
**Role:** Data Engineer  
**Duration:** Dec 2025 – Mar 2026 (3 months)  
**Stack:** PySpark 3.5 · Python · Jupyter Notebook · Docker / virtual environment

- Engineered a portable, self-contained PySpark 3.5 environment that resolves Python 3.12+ compatibility issues blocking standard Big Data coursework setups.
- Documented the configuration so teammates could replicate the environment in under 10 minutes, eliminating setup friction across the cohort.
- Used the environment to complete distributed data processing exercises covering RDDs, DataFrames, Spark SQL, and MLlib.

---

### Stack Overflow Developer Survey — EDA & Insights
**Role:** Data Analyst  
**Duration:** Jul 2025 – Mar 2026 (8 months)  
**Stack:** Python · Jupyter Notebook · Pandas · Matplotlib · Seaborn

- Performed exploratory data analysis on the Stack Overflow Annual Developer Survey dataset, extracting insights on compensation trends, technology adoption, and developer satisfaction by geography and role.
- Produced annotated visualizations identifying key differentiators in salary distribution across experience levels, programming languages, and remote-work status.

---

## Web Development & Full-Stack

---

### Hypercircle — AI & Data Science Consultancy Website
**Role:** Frontend Developer  
**Duration:** Nov 2025 – Mar 2026 (4 months)  
**Stack:** TypeScript · React · Tailwind CSS · Framer Motion · Vite

- Built the full company website for an AI and Data Science consultancy firm, architected with React and TypeScript for type-safe component development and long-term maintainability.
- Implemented smooth page transitions and scroll-triggered animations using Framer Motion, improving perceived performance and brand polish.
- Optimized bundle size and build pipeline with Vite, achieving fast cold-start load times for the marketing site.

---

## Database Engineering

---

### Dental Clinic Management System
**Role:** Database Developer  
**Duration:** Feb 2025 – Mar 2026 (13 months)  
**Stack:** T-SQL · SQL Server

- Designed and implemented a relational database system for a dental clinic, covering patient records, appointment scheduling, treatment history, and billing workflows.
- Wrote stored procedures and views in T-SQL to encapsulate business logic (appointment conflict detection, patient lookup, invoice generation) and enforce data integrity at the database layer.
- Applied normalization (3NF) to eliminate data redundancy and ensure referential integrity across patient, staff, appointment, and payment entities.

---

## Summary Table

| Project | Role | Duration | Stars | Live |
|---------|------|----------|-------|------|
| Customer Churn Prediction | ML Engineer | Sep 2025 – Mar 2026 | ⭐ 1 | ✅ |
| Supply Chain Optimization | Data Scientist | Sep 2025 – Mar 2026 | ⭐ 1 | ✅ |
| Sales Analytics Dashboard | Data Analyst | Sep 2025 – Mar 2026 | ⭐ 1 | ✅ |
| Sentiment Analysis Tool | NLP Engineer | Sep 2025 – Mar 2026 | ⭐ 1 | ✅ |
| Recommender System App | Data Scientist | Aug 2025 – Mar 2026 | — | ✅ |
| RL Recommendation System (DS540) | ML Researcher | Nov 2025 | — | — |
| Production Data Science Package | Data Engineer | Aug 2025 | — | — |
| PySpark Local Environment | Data Engineer | Dec 2025 – Mar 2026 | — | — |
| SO Survey Insights | Data Analyst | Jul 2025 – Mar 2026 | — | — |
| Hypercircle Website | Frontend Dev | Nov 2025 – Mar 2026 | ⭐ 1 | ✅ |
| Dental Clinic Mgmt System | DB Developer | Feb 2025 – Mar 2026 | — | — |
