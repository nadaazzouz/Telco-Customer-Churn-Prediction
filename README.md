#  Customer Churn Prediction
### Binary Classification · Telecom Domain · CRISP-DM Methodology

---

##  Project Overview

This project builds an end-to-end machine learning pipeline to predict customer churn in the telecommunications industry. The goal is to identify customers at high risk of cancelling their service so the marketing team can proactively target them with retention offers.

> **Business Question:** *How can we identify customers at high risk of cancelling their service in the next month so that we can target them with specific offers?*

---
##  Streamlit App link
https://telco-customer-churn-prediction-ml-dl.streamlit.app/
---
##  Project Structure

```
churn-prediction/
│
├── churn_app.py              # Streamlit web application (all pages)
├── churn_prediction.py       # Standalone analysis & training pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
│
├── plots/                    # Auto-generated plots (created on first run)
│   ├── 01_target_distribution.png
│   ├── 02_univariate_numerical.png
│   ├── 03_univariate_categorical.png
│   ├── 04_bivariate_numerical_churn.png
│   ├── 05_bivariate_categorical_churn.png
│   ├── 06_mutual_information.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_pairplot.png
│   ├── 09_feature_importance.png
│   ├── 10_ml_evaluation.png
│   ├── 11_ml_comparison.png
│   ├── 12_ann_training_curves.png
│   ├── 13_ann_evaluation.png
│   ├── 14_ann_comparison.png
│   └── 15_final_comparison.png
│
└── results/                  # Auto-generated CSVs (created on first run)
    ├── ml_results.csv
    ├── ann_results.csv
    └── all_models_results.csv
```

---

##  Dataset

- **Name:** IBM Telco Customer Churn  
- **Source:** [IBM Sample Data on GitHub](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)  
- **Records:** 7,043 customers  
- **Features:** 21 attributes (demographics, account info, services)  
- **Target:** `Churn` — Yes / No (binary classification)  

The dataset is loaded automatically from the URL on first run. If you have no internet access, download it manually and place it as `Telco-Customer-Churn.csv` in the project root.

---

##  Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

##  Running the App

### Streamlit Web App
```bash
streamlit run churn_app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Standalone Pipeline (generates all plots + CSV results)
```bash
python churn_prediction.py
```

---

##  App Pages

| Page | Description |
|------|-------------|
|  Home | Project summary, KPIs, hypotheses, CRISP-DM overview |
|  Data Overview | Feature catalogue, descriptive statistics, raw data browser |
|  Exploratory Analysis | Univariate, bivariate, multivariate analysis + hypothesis validation |
|  Data Preparation | Step-by-step preprocessing pipeline with interactive visualisations |
|  ML Modeling | Architecture table, feature importance, confusion matrices, ROC curves |
|  ANN Modeling | Architecture table, training curves, confusion matrices, ROC curves |
|  Final Comparison | All 6 models side-by-side, overlaid ROC curves, model selection |
|  Live Prediction | Enter customer details → get churn risk score + recommended action |

---

##  Business Hypotheses Validated

| # | Hypothesis | Result |
|---|-----------|--------|
| H1 | Month-to-month customers churn more than yearly contract customers |  Confirmed — 42.7% vs 2.8% |
| H2 | Fiber optic users have higher churn than DSL users |  Confirmed — 41.9% vs 18.9% |
| H3 | Higher monthly charges correlate with churn |  Confirmed — r ≈ +0.19 |

---

##  Models Evaluated

### Machine Learning
| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline, interpretable coefficients |
| Random Forest | Ensemble of 200 decision trees (bagging) |
| XGBoost | Gradient boosting, 200 estimators, lr=0.05 |

### Artificial Neural Networks
| Model | Architecture |
|-------|-------------|
| ANN-1 Shallow | Input → 64 → Output (1 hidden layer) |
| ANN-2 Deep | Input → 128 → 64 → 32 → Output (3 hidden layers) |
| ANN-3 Wide+L2 | Input → 256 → 128 → Output (2 hidden layers, L2 regularisation) |

**All models trained on SMOTE-balanced data. Evaluated on original-distribution test set (80/20 split).**

---

##  Data Preparation Pipeline

```
Raw Data (7,043 × 21)
    │
    ├── 1. Type correction (TotalCharges: string → float)
    ├── 2. Outlier detection (IQR method → treat as missing)
    ├── 3. Missing value imputation (median)
    ├── 4. Drop customerID
    ├── 5. Encode target (Churn: Yes→1, No→0)
    ├── 6. Feature engineering (+4 new features)
    ├── 7. Binary encoding (Yes/No → 1/0)
    ├── 8. One-hot encoding (multi-class categoricals)
    ├── 9. StandardScaler normalisation
    ├── 10. Feature selection (top 20 by RF importance)
    ├── 11. Train/test split (80/20, stratified)
    └── 12. SMOTE oversampling (training set only)
         │
         └── Model-ready: X_train (balanced) · X_test (original distribution)
```

---

##  Model Selection

**XGBoost** is selected as the final deployment model based on best F1-Score and ROC-AUC across all 6 candidates.

> Deep learning (ANN) models do not outperform gradient boosting on this dataset. This is consistent with published benchmarks — for small structured tabular datasets (~7,000 rows), tree-based ensembles consistently outperform MLPs. The ANN advantage typically requires 100,000+ samples.

---

##  Risk Segmentation

| Risk Level | Churn Probability | Action |
|------------|------------------|--------|
|  High Risk | > 75% | Immediate personal outreach — offer contract upgrade or discount |
|  Medium Risk | 55–75% | Targeted email/SMS — loyalty reward |
|  Low Risk | 35–54% | Monitor — satisfaction survey |
|  Stable | < 35% | No intervention needed |

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Web App | Streamlit |
| Data | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Balancing | imbalanced-learn (SMOTE) |
| Methodology | CRISP-DM |

---

##  Notes

- Models are cached with `@st.cache_data` / `@st.cache_resource` — training only happens once per session.
- Run all cells top-to-bottom in `churn_prediction.py` to avoid state issues from partial re-execution.
- The `plots/` and `results/` directories are created automatically on first run.
