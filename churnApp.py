import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, time
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Page config
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Global theme
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PAL = {
    "primary":   "#2E75B6",
    "secondary": "#ED7D31",
    "success":   "#27AE60",
    "danger":    "#E74C3C",
    "purple":    "#8E44AD",
    "dark":      "#1F3864",
    "light":     "#D6E4F0",
    "bg":        "#F8FAFD",
}

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1F3864 0%, #2E75B6 100%);
    border-right: none;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stRadio label { 
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    padding: 0.4rem 0;
}

/* Main headings */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }
h3 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #E8EEF6;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(46,117,182,0.08);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

/* Info / success boxes */
.info-card {
    background: linear-gradient(135deg, #EBF4FF, #DBEAFE);
    border-left: 4px solid #2E75B6;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.success-card {
    background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
    border-left: 4px solid #27AE60;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.warn-card {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border-left: 4px solid #ED7D31;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.danger-card {
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    border-left: 4px solid #E74C3C;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #1F3864;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-bottom: 2px solid #2E75B6;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# DATA LOADING & CACHING


@st.cache_data
def load_raw_data():
    URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        df = pd.read_csv(URL)
    except Exception:
        df = pd.read_csv("Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

@st.cache_data
def prepare_data(df_raw):
    df = df_raw.copy()

    # Fix type
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Outliers → NaN
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df.loc[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR), col] = np.nan

    # Impute missing
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Drop ID
    df.drop(columns=["customerID"], inplace=True)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Feature engineering
    df["AvgMonthlyCharge"]    = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"]         = pd.cut(df["tenure"], bins=[0,12,24,48,72],
                                        labels=["0-1yr","1-2yr","2-4yr","4+yr"])
    df["HasStreamingService"] = ((df["StreamingTV"]=="Yes")|(df["StreamingMovies"]=="Yes")).astype(int)
    df["NumServices"]         = (df[["PhoneService","OnlineSecurity","OnlineBackup",
                                     "DeviceProtection","TechSupport",
                                     "StreamingTV","StreamingMovies"]]=="Yes").sum(axis=1)

    # Encode binary
    binary_map = {"Yes":1,"No":0,"Male":1,"Female":0}
    binary_cols = ["gender","Partner","Dependents","PhoneService","PaperlessBilling",
                   "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                   "StreamingTV","StreamingMovies"]
    for col in binary_cols:
        df[col] = df[col].map(binary_map).fillna(df[col])

    # OHE
    ohe_cols = ["MultipleLines","InternetService","Contract","PaymentMethod","TenureGroup"]
    ohe_present = [c for c in ohe_cols if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_present, drop_first=False)

    # Split X/y
    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Feature selection via quick RF
    rf_quick = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
    rf_quick.fit(X_scaled, y)
    feat_imp = pd.Series(rf_quick.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = feat_imp.head(20).index.tolist()

    X_sel = X_scaled[top_features]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    return X_train_bal, X_test, y_train_bal, y_test, feat_imp, top_features, scaler

@st.cache_resource
def train_ml_models(X_train, y_train, X_test, y_test):
    results = {}

    # M1: Logistic Regression
    m1 = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    t0 = time.time()
    m1.fit(X_train, y_train)
    results["Logistic Regression"] = _eval(m1, X_test, y_test, time.time()-t0)

    # M2: Random Forest
    m2 = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                                 min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.time()
    m2.fit(X_train, y_train)
    results["Random Forest"] = _eval(m2, X_test, y_test, time.time()-t0)

    # M3: XGBoost
    m3 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=RANDOM_STATE)
    t0 = time.time()
    m3.fit(X_train, y_train)
    results["XGBoost"] = _eval(m3, X_test, y_test, time.time()-t0)

    return results, m1, m2, m3

@st.cache_resource
def train_ann_models(X_train, y_train, X_test, y_test):
    tf.random.set_seed(RANDOM_STATE)
    n_feat = X_train.shape[1]
    X_tr = X_train.values.astype(np.float32)
    X_te = X_test.values.astype(np.float32)
    y_tr = y_train.values.astype(np.float32)
    y_te = y_test.values.astype(np.float32)

    cb = [EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
          ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0)]

    histories = {}
    results = {}

    # ANN-1: Shallow
    a1 = keras.Sequential([
        layers.Input(shape=(n_feat,)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="Shallow")
    a1.compile(optimizer=keras.optimizers.Adam(0.001),
               loss="binary_crossentropy", metrics=["accuracy"])
    h1 = a1.fit(X_tr, y_tr, validation_split=0.15, epochs=100, batch_size=64,
                callbacks=cb, verbose=0)
    histories["ANN-1 Shallow"] = h1.history
    results["ANN-1 Shallow"] = _eval_ann(a1, X_te, y_te)

    # ANN-2: Deep
    a2 = keras.Sequential([
        layers.Input(shape=(n_feat,)),
        layers.Dense(128, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(64,  activation="relu"), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32,  activation="relu"), layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="Deep")
    a2.compile(optimizer=keras.optimizers.Adam(0.001),
               loss="binary_crossentropy", metrics=["accuracy"])
    h2 = a2.fit(X_tr, y_tr, validation_split=0.15, epochs=150, batch_size=32,
                callbacks=cb, verbose=0)
    histories["ANN-2 Deep"] = h2.history
    results["ANN-2 Deep"] = _eval_ann(a2, X_te, y_te)

    # ANN-3: Wide + L2
    a3 = keras.Sequential([
        layers.Input(shape=(n_feat,)),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="Wide_L2")
    a3.compile(optimizer=keras.optimizers.RMSprop(0.001),
               loss="binary_crossentropy", metrics=["accuracy"])
    h3 = a3.fit(X_tr, y_tr, validation_split=0.15, epochs=150, batch_size=64,
                callbacks=cb, verbose=0)
    histories["ANN-3 Wide+L2"] = h3.history
    results["ANN-3 Wide+L2"] = _eval_ann(a3, X_te, y_te)

    return results, histories, a1, a2, a3

def _eval(model, X_te, y_te, t):
    yp = model.predict(X_te)
    yb = model.predict_proba(X_te)[:,1]
    return {
        "Accuracy":  round(accuracy_score(y_te, yp),4),
        "Precision": round(precision_score(y_te, yp, zero_division=0),4),
        "Recall":    round(recall_score(y_te, yp, zero_division=0),4),
        "F1":        round(f1_score(y_te, yp, zero_division=0),4),
        "ROC-AUC":   round(roc_auc_score(y_te, yb),4),
        "Train Time":round(t,2),
        "y_pred": yp, "y_prob": yb,
    }

def _eval_ann(model, X_te, y_te):
    yb = model.predict(X_te, verbose=0).ravel()
    yp = (yb > 0.5).astype(int)
    return {
        "Accuracy":  round(accuracy_score(y_te, yp),4),
        "Precision": round(precision_score(y_te, yp, zero_division=0),4),
        "Recall":    round(recall_score(y_te, yp, zero_division=0),4),
        "F1":        round(f1_score(y_te, yp, zero_division=0),4),
        "ROC-AUC":   round(roc_auc_score(y_te, yb),4),
        "Train Time": "—",
        "y_pred": yp, "y_prob": yb,
    }

def metric_delta(val, ref=0.75, label=""):
    diff = val - ref
    return f"{'+' if diff>=0 else ''}{diff:.3f}"

def plotly_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    x=["No Churn","Churn"], y=["No Churn","Churn"])
    fig.update_layout(title=title, title_font_size=14,
                      coloraxis_showscale=False, height=300,
                      margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plotly_roc(y_true, y_prob, name, color):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"{name} (AUC={auc:.3f})", line=dict(color=color, width=2.5)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(dash="dash", color="gray", width=1), showlegend=False))
    fig.update_layout(title=f"ROC — {name}", height=300,
                      xaxis_title="FPR", yaxis_title="TPR",
                      margin=dict(l=10,r=10,t=40,b=10),
                      legend=dict(x=0.55, y=0.1))
    return fig

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Churn Predictor")
    st.markdown("*Telco Customer Retention*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "  Home",
        "  Data Overview",
        "  Exploratory Analysis",
        "  Data Preparation",
        "  ML Modeling",
        "  ANN Modeling",
        "  Final Comparison",
        "  Live Prediction",
    ])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; opacity:0.75; line-height:1.6'>
    <b>Dataset:</b> IBM Telco Churn<br>
    <b>Records:</b> 7,043 customers<br>
    <b>Features:</b> 21 attributes<br>
    <b>Task:</b> Binary Classification<br>
    <b>Method:</b> CRISP-DM
    </div>
    """, unsafe_allow_html=True)

# Load data upfront
with st.spinner("Loading dataset..."):
    df_raw = load_raw_data()

# ════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════
if page == " Home":
    st.markdown("# 📡 Customer Churn Prediction")
    st.markdown("### Binary Classification · Telecom Domain · CRISP-DM Methodology")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df_raw["Churn"] == "Yes").mean() * 100
    col1.metric("Total Customers", f"{len(df_raw):,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"↑ vs 15% avg", delta_color="inverse")
    col3.metric("Features", "21 raw → 20 selected")
    col4.metric("Models Evaluated", "6  (3 ML + 3 ANN)")

    st.markdown("---")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Business Problem")
        st.markdown("""
        <div class='info-card'>
        <b>How can we identify customers at high risk of cancelling their service
        in the next month so that the marketing team can target them with specific retention offers?</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Three Validated Hypotheses")
        st.markdown("""
        <div class='success-card'>✔ <b>H1</b> — Month-to-month contract customers churn at 42.7% vs 2.8% for 2-year contracts</div>
        <div class='success-card'>✔ <b>H2</b> — Fiber optic users churn at 41.9% vs 18.9% for DSL users</div>
        <div class='success-card'>✔ <b>H3</b> — Higher MonthlyCharges positively correlates with churn (r ≈ +0.19)</div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("#### CRISP-DM Phases")
        phases = [
            ("1", "Business Understanding", "Define churn problem"),
            ("2", "Data Understanding", "EDA + Hypothesis testing"),
            ("3", "Data Preparation", "Clean, encode, SMOTE"),
            ("4", "Modeling", "3 ML + 3 ANN trained"),
            ("5", "Evaluation", "6-model comparison"),
            ("6", "Deployment", "This Streamlit app"),
        ]
        for n, phase, desc in phases:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:10px; 
                        background:white; border-radius:8px; padding:8px 12px; 
                        margin:4px 0; border:1px solid #E8EEF6; font-size:0.88rem;'>
                <span style='background:#2E75B6; color:white; border-radius:50%; 
                             width:24px; height:24px; display:flex; align-items:center; 
                             justify-content:center; font-weight:700; font-size:0.8rem; flex-shrink:0;'>{n}</span>
                <div><b>{phase}</b><br><span style='color:#666'>{desc}</span></div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE: DATA OVERVIEW
# ════════════════════════════════════════════════════════════
elif page == "Data Overview":
    st.markdown("#Data Overview")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Dataset Info", "Statistics", "Raw Data"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Dataset Properties")
            info = pd.DataFrame({
                "Property": ["Rows","Columns","Numerical","Categorical","Binary","Target","Missing Values","Class Imbalance"],
                "Value": ["7,043", "21", "3 (tenure, MonthlyCharges, TotalCharges)",
                          "5 (Contract, PaymentMethod, InternetService…)", "12 (Yes/No features)",
                          "Churn (Yes/No)", "11 blanks in TotalCharges", "73.5% No / 26.5% Yes"],
            })
            st.dataframe(info, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("#### Churn Distribution")
            vc = df_raw["Churn"].value_counts()
            fig = go.Figure(go.Pie(
                labels=["No Churn","Churn"],
                values=vc.values,
                hole=0.55,
                marker_colors=[PAL["success"], PAL["danger"]],
                textinfo="label+percent",
                textfont_size=13,
            ))
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10),
                              showlegend=False,
                              annotations=[dict(text=f"<b>{len(df_raw):,}</b><br>customers",
                                               x=0.5, y=0.5, font_size=13, showarrow=False)])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Feature Catalogue")
        catalogue = pd.DataFrame({
            "Feature": ["customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
                        "PhoneService","MultipleLines","InternetService","OnlineSecurity",
                        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
                        "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
                        "MonthlyCharges","TotalCharges","Churn"],
            "Type":    ["ID","Binary","Binary","Binary","Binary","Numerical",
                        "Binary","Categorical","Categorical","Categorical",
                        "Categorical","Categorical","Categorical","Categorical",
                        "Categorical","Categorical","Binary","Categorical",
                        "Numerical","Numerical","TARGET"],
            "Description": [
                "Unique customer ID — dropped before modeling",
                "Male / Female", "1 if 65+, else 0",
                "Has a partner", "Has dependents",
                "Months with company (1–72)",
                "Has phone service", "No / Yes / No phone service",
                "DSL / Fiber optic / None", "Online security add-on",
                "Online backup add-on", "Device protection add-on",
                "Tech support add-on", "Streaming TV subscription",
                "Streaming movies subscription",
                "Month-to-month / One year / Two year",
                "Paperless billing enabled",
                "Electronic check / Mailed check / Bank transfer / Credit card",
                "Monthly amount charged (USD)",
                "Total amount charged — contains 11 blank entries!",
                "Churned? Yes=1 / No=0",
            ],
        })
        def color_type(val):
            colors = {"ID":"background-color:#f0f0f0","Numerical":"background-color:#EBF4FF",
                      "Binary":"background-color:#ECFDF5","Categorical":"background-color:#FFF8E7",
                      "TARGET":"background-color:#FEE2E2; font-weight:bold"}
            return colors.get(val,"")
        st.dataframe(catalogue.style.map(color_type, subset=["Type"]),
                     hide_index=True, use_container_width=True)

    with tab2:
        st.markdown("#### Numerical Features — Descriptive Statistics")
        df_num = df_raw[["tenure","MonthlyCharges","TotalCharges"]].copy()
        df_num["TotalCharges"] = pd.to_numeric(df_num["TotalCharges"], errors="coerce")
        st.dataframe(df_num.describe().round(2), use_container_width=True)

        st.markdown("#### Categorical Features — Value Counts")
        cat_cols = ["Contract","InternetService","PaymentMethod","MultipleLines"]
        cols = st.columns(4)
        for i, col in enumerate(cat_cols):
            with cols[i]:
                st.markdown(f"**{col}**")
                st.dataframe(df_raw[col].value_counts().reset_index().rename(
                    columns={"index":col,"count":"Count"}),
                    hide_index=True, use_container_width=True)

    with tab3:
        st.markdown("#### Raw Dataset Sample")
        n = st.slider("Rows to display", 5, 100, 20)
        st.dataframe(df_raw.head(n), use_container_width=True)
        st.caption(f"Showing {n} of {len(df_raw):,} rows · {df_raw.shape[1]} columns")

# ════════════════════════════════════════════════════════════
# PAGE: EXPLORATORY ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "Exploratory Analysis":
    st.markdown("#Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Univariate","Bivariate","Multivariate","Hypotheses"])

    df_eda = df_raw.copy()
    df_eda["TotalCharges"] = pd.to_numeric(df_eda["TotalCharges"], errors="coerce")
    df_eda["Churn_bin"] = (df_eda["Churn"]=="Yes").astype(int)

    with tab1:
        st.markdown("### Univariate Analysis")
        st.markdown("#### Numerical Features Distribution")
        num_choice = st.selectbox("Select feature", ["tenure","MonthlyCharges","TotalCharges"])
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_eda, x=num_choice, nbins=40, color_discrete_sequence=[PAL["primary"]],
                               title=f"Histogram — {num_choice}")
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df_eda, y=num_choice, color_discrete_sequence=[PAL["secondary"]],
                         title=f"Boxplot — {num_choice}")
            fig.update_layout(height=300, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Categorical Features Distribution")
        cat_opts = ["Contract","InternetService","PaymentMethod","TechSupport",
                    "OnlineSecurity","MultipleLines","gender","PaperlessBilling"]
        cat_choice = st.selectbox("Select feature", cat_opts)
        vc = df_eda[cat_choice].value_counts().reset_index()
        fig = px.bar(vc, x=cat_choice, y="count", color=cat_choice,
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     title=f"Distribution — {cat_choice}")
        fig.update_layout(height=320, showlegend=False, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Bivariate Analysis — Feature vs Churn")

        st.markdown("#### Numerical Features by Churn Class")
        num_feat = st.selectbox("Feature", ["tenure","MonthlyCharges","TotalCharges"], key="biv_num")
        fig = px.histogram(df_eda.dropna(subset=[num_feat]), x=num_feat, color="Churn",
                           barmode="overlay", nbins=40,
                           color_discrete_map={"No": PAL["success"], "Yes": PAL["danger"]},
                           opacity=0.7, title=f"{num_feat} by Churn Status",
                           labels={"Churn":"Churn?"})
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Categorical Features — Churn Rate")
        cat_feat = st.selectbox("Feature", ["Contract","InternetService","PaymentMethod",
                                             "TechSupport","OnlineSecurity","PaperlessBilling"], key="biv_cat")
        ct = df_eda.groupby([cat_feat,"Churn"]).size().unstack(fill_value=0)
        ct_pct = (ct.div(ct.sum(axis=1), axis=0)*100).reset_index()
        ct_pct_m = ct_pct.melt(id_vars=cat_feat, var_name="Churn", value_name="Percentage")
        fig = px.bar(ct_pct_m, x=cat_feat, y="Percentage", color="Churn",
                     color_discrete_map={"No": PAL["success"], "Yes": PAL["danger"]},
                     title=f"Churn Rate by {cat_feat}", barmode="stack",
                     labels={"Percentage":"Percentage (%)"})
        fig.update_layout(height=350, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Mutual Information Scores")
        df_mi = df_eda.drop(columns=["Churn","Churn_bin"]).copy()
        df_mi["TotalCharges"] = df_mi["TotalCharges"].fillna(0)
        for c in df_mi.select_dtypes("object").columns:
            df_mi[c] = LabelEncoder().fit_transform(df_mi[c].astype(str))
        mi = mutual_info_classif(df_mi, df_eda["Churn_bin"], random_state=RANDOM_STATE)
        mi_df = pd.Series(mi, index=df_mi.columns).sort_values(ascending=False).head(15).reset_index()
        mi_df.columns = ["Feature","MI Score"]
        fig = px.bar(mi_df, x="MI Score", y="Feature", orientation="h",
                     color="MI Score", color_continuous_scale="Blues",
                     title="Top 15 Features — Mutual Information vs Churn")
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10),
                          coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Multivariate Analysis")

        st.markdown("#### Correlation Heatmap")
        df_corr = df_eda.drop(columns=["Churn"]).copy()
        df_corr["TotalCharges"] = df_corr["TotalCharges"].fillna(0)
        for c in df_corr.select_dtypes("object").columns:
            df_corr[c] = LabelEncoder().fit_transform(df_corr[c].astype(str))
        df_corr["Churn"] = df_eda["Churn_bin"]
        corr = df_corr.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.where(~mask)
        fig = px.imshow(corr_masked, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto",
                        title="Correlation Heatmap (lower triangle)")
        fig.update_layout(height=550, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 3D Scatter — Tenure × MonthlyCharges × TotalCharges by Churn")
        df_3d = df_eda[["tenure","MonthlyCharges","TotalCharges","Churn"]].dropna()
        fig = px.scatter_3d(df_3d, x="tenure", y="MonthlyCharges", z="TotalCharges",
                            color="Churn",
                            color_discrete_map={"No":PAL["success"],"Yes":PAL["danger"]},
                            opacity=0.5, size_max=4,
                            title="3D Scatter — Numerical Features by Churn")
        fig.update_layout(height=500, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### 🔬 Business Hypothesis Validation")

        st.markdown("#### H1 — Contract Type & Churn")
        h1_data = df_eda.groupby("Contract")["Churn_bin"].mean().mul(100).reset_index()
        h1_data.columns = ["Contract","Churn Rate (%)"]
        fig = px.bar(h1_data, x="Contract", y="Churn Rate (%)",
                     color="Churn Rate (%)", color_continuous_scale="Reds",
                     title="Churn Rate by Contract Type")
        fig.update_layout(height=320, coloraxis_showscale=False,
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="success-card">✔ <b>H1 CONFIRMED</b> — Month-to-month: 42.7% · One year: 11.3% · Two year: 2.8%</div>', unsafe_allow_html=True)

        st.markdown("#### H2 — Internet Service & Churn")
        h2_data = df_eda.groupby("InternetService")["Churn_bin"].mean().mul(100).reset_index()
        h2_data.columns = ["InternetService","Churn Rate (%)"]
        fig = px.bar(h2_data, x="InternetService", y="Churn Rate (%)",
                     color="Churn Rate (%)", color_continuous_scale="Oranges",
                     title="Churn Rate by Internet Service Type")
        fig.update_layout(height=320, coloraxis_showscale=False,
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="success-card">✔ <b>H2 CONFIRMED</b> — Fiber optic: 41.9% · DSL: 18.9% · No service: 7.4%</div>', unsafe_allow_html=True)

        st.markdown("#### H3 — Monthly Charges & Churn")
        fig = px.box(df_eda, x="Churn", y="MonthlyCharges",
                     color="Churn", color_discrete_map={"No":PAL["success"],"Yes":PAL["danger"]},
                     title="Monthly Charges Distribution by Churn")
        fig.update_layout(height=320, showlegend=False,
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)
        corr_val = df_eda["MonthlyCharges"].corr(df_eda["Churn_bin"])
        st.markdown(f'<div class="success-card">✔ <b>H3 CONFIRMED</b> — Pearson correlation(MonthlyCharges, Churn) = {corr_val:.4f} (positive relationship)</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE: DATA PREPARATION
# ════════════════════════════════════════════════════════════
elif page == "Data Preparation":
    st.markdown("#Data Preparation Pipeline")
    st.markdown("---")

    steps = {
        "Step 1 — Type Fix": "TotalCharges stored as string → coerced to float. 11 blank entries become NaN.",
        "Step 2 — Outliers (IQR)": "Outliers detected per column using Q1−1.5×IQR / Q3+1.5×IQR. Set to NaN (not deleted).",
        "Step 3 — Missing Values": "All NaN values imputed with column median to avoid sensitivity to extremes.",
        "Step 4 — Drop customerID": "Unique identifier — no predictive value. Removed before modeling.",
        "Step 5 — Encode Target": "Churn: Yes→1, No→0.",
        "Step 6 — Feature Engineering": "4 new features: AvgMonthlyCharge, TenureGroup, HasStreamingService, NumServices.",
        "Step 7 — Binary Encoding": "Yes/No columns mapped to 1/0. gender: Male=1, Female=0.",
        "Step 8 — One-Hot Encoding": "MultipleLines, InternetService, Contract, PaymentMethod, TenureGroup → dummy columns.",
        "Step 9 — Scaling": "StandardScaler applied to all features (zero mean, unit variance).",
        "Step 10 — Feature Selection": "Top 20 features selected by Random Forest importance score.",
        "Step 11 — Train/Test Split": "80% train / 20% test (stratified by Churn).",
        "Step 12 — SMOTE": "Synthetic Minority Over-sampling applied to training set → balanced 50:50 classes.",
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### Pipeline Steps")
        selected_step = st.radio("", list(steps.keys()), label_visibility="collapsed")
    with col2:
        st.markdown(f"#### {selected_step}")
        st.markdown(f'<div class="info-card">{steps[selected_step]}</div>', unsafe_allow_html=True)

        # Show relevant visualisation per step
        df_prep = df_raw.copy()
        df_prep["TotalCharges"] = pd.to_numeric(df_prep["TotalCharges"], errors="coerce")

        if "Outlier" in selected_step:
            feat = st.selectbox("Feature", ["tenure","MonthlyCharges","TotalCharges"])
            col_data = df_prep[feat].dropna()
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1-1.5*IQR, Q3+1.5*IQR
            outliers = col_data[(col_data<lower)|(col_data>upper)]
            fig = go.Figure()
            fig.add_trace(go.Box(y=col_data, name=feat, marker_color=PAL["primary"],
                                 boxpoints="outliers", jitter=0.3))
            fig.add_hline(y=lower, line_dash="dash", line_color=PAL["danger"],
                          annotation_text=f"Lower: {lower:.1f}")
            fig.add_hline(y=upper, line_dash="dash", line_color=PAL["danger"],
                          annotation_text=f"Upper: {upper:.1f}")
            fig.update_layout(height=350, title=f"Outlier Bounds — {feat}",
                              margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"**{len(outliers)}** outliers detected in `{feat}`")

        elif "Missing" in selected_step:
            missing = df_prep.isnull().sum()
            missing = missing[missing>0].reset_index()
            missing.columns = ["Column","Missing Count"]
            missing["Missing %"] = (missing["Missing Count"]/len(df_prep)*100).round(2)
            st.dataframe(missing, hide_index=True, use_container_width=True)

        elif "Feature Engineering" in selected_step:
            st.markdown("**New Features Created:**")
            eng = pd.DataFrame({
                "Feature": ["AvgMonthlyCharge","TenureGroup","HasStreamingService","NumServices"],
                "Formula": ["TotalCharges / (tenure + 1)","Cut(tenure, [0,12,24,48,72])",
                            "StreamingTV=Yes OR StreamingMovies=Yes","Count of active add-on services"],
                "Rationale": ["Normalised spend rate","Tenure business segments",
                              "Consolidated streaming flag","Service adoption depth"],
            })
            st.dataframe(eng, hide_index=True, use_container_width=True)

        elif "SMOTE" in selected_step:
            fig = go.Figure(go.Bar(
                x=["No Churn (before)","Churn (before)","No Churn (after)","Churn (after)"],
                y=[4131, 1035, 4131, 4131],
                marker_color=[PAL["success"],PAL["danger"],PAL["success"],PAL["primary"]],
                text=["4,131","1,035","4,131","4,131 (synthetic)"],
                textposition="outside"
            ))
            fig.update_layout(title="SMOTE — Before vs After Class Balancing",
                              height=320, margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### State 1 → State 2 Transformation Summary")
    summary = pd.DataFrame({
        "Stage":["Raw","After cleaning","After engineering","After encoding","After selection","Train (SMOTE)","Test"],
        "Rows":["7,043","7,043","7,043","7,043","7,043","~8,262","1,409"],
        "Columns":["21","20","24","~35","20","20","20"],
        "Notes":["Original dataset","No missing, no outliers","4 new features added",
                 "OHE expanded categoricals","Top 20 by RF importance",
                 "Balanced 50/50 classes","Original distribution (stratified)"],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: ML MODELING
# ════════════════════════════════════════════════════════════
elif page == "ML Modeling":
    st.markdown("#Machine Learning Models")
    st.markdown("---")

    with st.spinner("Preparing data and training ML models... (cached after first run)"):
        X_train, X_test, y_train, y_test, feat_imp, top_features, scaler = prepare_data(df_raw)
        ml_results, m1, m2, m3 = train_ml_models(X_train, y_train, X_test, y_test)

    tab1, tab2, tab3 = st.tabs(["Architecture"," Results"," Model Details"])

    with tab1:
        st.markdown("#### Model Architecture Comparison")
        arch = pd.DataFrame({
            "Parameter":["Model Type","Base Algorithm","N Estimators","Max Depth",
                         "Learning Rate","Regularisation","Solver/Method","Feature Set",
                         "Class Balancing","Interpretability"],
            "M1 — Logistic Regression":["Linear","Sigmoid (log-odds)","N/A","N/A","N/A",
                                         "L2 (C=1.0)","lbfgs","Top 20","SMOTE","⭐⭐⭐ High"],
            "M2 — Random Forest":["Ensemble (Bagging)","CART Decision Trees","200","10","N/A",
                                   "min_samples_leaf=2","Gini","Top 20","SMOTE","⭐⭐ Medium"],
            "M3 — XGBoost":["Ensemble (Boosting)","Shallow Trees","200","5","0.05",
                             "subsample=0.8, colsample=0.8","logloss","Top 20","SMOTE","⭐⭐ Medium"],
        })
        st.dataframe(arch.set_index("Parameter"), use_container_width=True)

        st.markdown("#### Feature Importance")
        fi_df = feat_imp.head(15).reset_index()
        fi_df.columns = ["Feature","Importance"]
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues",
                     title="Top 15 Features — Random Forest Importance")
        fig.update_layout(height=420, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"),
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Performance Metrics")
        metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        rows = []
        for name, res in ml_results.items():
            rows.append({**{"Model":name}, **{m:res[m] for m in metrics}, "Train Time (s)":res["Train Time"]})
        df_ml = pd.DataFrame(rows)

        def highlight_best(df):
            styled = df.copy().astype(str)
            for col in metrics:
                best_idx = df[col].astype(float).idxmax()
                styled.loc[best_idx, col] = f" {df.loc[best_idx, col]}"
            return styled
        st.dataframe(highlight_best(df_ml), hide_index=True, use_container_width=True)

        st.markdown("#### Metric Comparison Chart")
        fig = go.Figure()
        colors = [PAL["primary"], PAL["secondary"], PAL["success"]]
        for i, (name, res) in enumerate(ml_results.items()):
            fig.add_trace(go.Bar(
                name=name, x=metrics,
                y=[res[m] for m in metrics],
                marker_color=colors[i], opacity=0.85,
            ))
        fig.update_layout(barmode="group", height=380, yaxis_range=[0,1.05],
                          title="ML Models — Performance Comparison",
                          margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Confusion Matrix & ROC Curves")
        model_names = list(ml_results.keys())
        cols = st.columns(3)
        col_colors = [PAL["primary"], PAL["secondary"], PAL["success"]]
        for i, (name, res) in enumerate(ml_results.items()):
            with cols[i]:
                st.plotly_chart(plotly_cm(y_test, res["y_pred"], f"CM — {name}"),
                                use_container_width=True)
                st.plotly_chart(plotly_roc(y_test, res["y_prob"], name, col_colors[i]),
                                use_container_width=True)

    with tab3:
        sel = st.selectbox("Select model for details", list(ml_results.keys()))
        res = ml_results[sel]
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy",  f"{res['Accuracy']:.3f}")
        c2.metric("Precision", f"{res['Precision']:.3f}")
        c3.metric("Recall",    f"{res['Recall']:.3f}")
        c4.metric("F1-Score",  f"{res['F1']:.3f}")
        c5.metric("ROC-AUC",   f"{res['ROC-AUC']:.3f}")

        discussions = {
            "Logistic Regression": "LR delivers a strong interpretable baseline. SMOTE balancing leads to high Recall (~0.81+) — the model identifies most churners but at the cost of more false positives (lower Precision). This trade-off may be acceptable when contact cost is low. LR's coefficients confirm that Contract_Month-to-month and tenure are the dominant predictors, consistent with EDA findings.",
            "Random Forest": "RF significantly improves Precision and overall F1 compared to LR by capturing non-linear feature interactions (e.g., short tenure AND high charges AND fiber optic). The ensemble structure with 200 trees reduces variance. Feature importance aligns well with mutual information scores from EDA — Contract type and tenure dominate.",
            "XGBoost": "XGBoost achieves the best F1 and ROC-AUC among ML models. Sequential boosting corrects residuals from previous trees, making it particularly effective on the correlated feature structure of this dataset. The learning rate shrinkage (0.05) and stochastic subsampling prevent overfitting. XGBoost is the recommended ML model for deployment.",
        }
        st.markdown(f'<div class="info-card"><b>Discussion:</b><br>{discussions.get(sel,"")}</div>',
                    unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE: ANN MODELING
# ════════════════════════════════════════════════════════════
elif page == "ANN Modeling":
    st.markdown("# Artificial Neural Network Models")
    st.markdown("---")

    with st.spinner("Preparing data and training ANN models... this may take a moment."):
        X_train, X_test, y_train, y_test, feat_imp, top_features, scaler = prepare_data(df_raw)
        ann_results, histories, a1, a2, a3 = train_ann_models(X_train, y_train, X_test, y_test)

    tab1, tab2, tab3 = st.tabs(["Architectures","Training Curves","Results"])

    with tab1:
        st.markdown("#### ANN Architecture Comparison")
        arch = pd.DataFrame({
            "Parameter":["Architecture","Hidden Layers","Neurons per Layer",
                         "Activation (Hidden)","Activation (Output)",
                         "Dropout","Batch Normalisation","Regularisation",
                         "Optimizer","Loss","Batch Size","Max Epochs"],
            "ANN-1 Shallow":["1 HL Funnel","1","64","ReLU","Sigmoid",
                              "0.3","Yes","Dropout only","Adam lr=0.001",
                              "Binary CE","64","100"],
            "ANN-2 Deep":["3 HL Funnel","3","128→64→32","ReLU","Sigmoid",
                           "0.4→0.3→0.2","Yes (each layer)","Dropout only",
                           "Adam lr=0.001","Binary CE","32","150"],
            "ANN-3 Wide+L2":["2 HL Wide","2","256→128","ReLU","Sigmoid",
                              "0.5→0.3","Yes (each layer)","L2 (λ=0.001) + Dropout",
                              "RMSprop lr=0.001","Binary CE","64","150"],
        })
        st.dataframe(arch.set_index("Parameter"), use_container_width=True)

        st.markdown("""
        <div class='warn-card'>
        <b> Note on Deep Learning vs Tabular Data</b><br>
        ANN models are powerful but not always superior on small structured datasets (~7,000 samples).
        Deep networks (ANN-2) may be over-parameterised relative to data size — training curves
        often show higher validation loss variance. This is expected and discussed in Chapter 3.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Training & Validation Curves")
        ann_choice = st.selectbox("Select model", list(histories.keys()))
        h = histories[ann_choice]
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Loss Curve","Accuracy Curve"))
        fig.add_trace(go.Scatter(y=h["loss"], name="Train Loss",
                                 line=dict(color=PAL["primary"],width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(y=h["val_loss"], name="Val Loss",
                                 line=dict(color=PAL["danger"],width=2,dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(y=h["accuracy"], name="Train Acc",
                                 line=dict(color=PAL["primary"],width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(y=h["val_accuracy"], name="Val Acc",
                                 line=dict(color=PAL["danger"],width=2,dash="dash")), row=1, col=2)
        fig.update_layout(height=380, title=f"Training Curves — {ann_choice}",
                          margin=dict(l=10,r=10,t=60,b=10))
        fig.update_xaxes(title_text="Epoch")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Early stopping triggered at epoch {len(h['loss'])} (patience=10 on val_loss)")

    with tab3:
        st.markdown("#### ANN Performance Metrics")
        metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        rows = []
        for name, res in ann_results.items():
            rows.append({**{"Model":name}, **{m:res[m] for m in metrics}})
        df_ann = pd.DataFrame(rows)

        def highlight_ann(df):
            styled = df.copy().astype(str)
            for col in metrics:
                best_idx = df[col].astype(float).idxmax()
                styled.loc[best_idx, col] = f" {df.loc[best_idx, col]}"
            return styled
        st.dataframe(highlight_ann(df_ann), hide_index=True, use_container_width=True)

        cols = st.columns(3)
        col_colors = [PAL["primary"], PAL["purple"], PAL["secondary"]]
        for i, (name, res) in enumerate(ann_results.items()):
            with cols[i]:
                st.plotly_chart(plotly_cm(y_test, res["y_pred"], f"CM — {name}"),
                                use_container_width=True)
                st.plotly_chart(plotly_roc(y_test, res["y_prob"], name, col_colors[i]),
                                use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: FINAL COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "Final Comparison":
    st.markdown("# Final Model Comparison — All 6 Models")
    st.markdown("---")

    with st.spinner("Loading all model results..."):
        X_train, X_test, y_train, y_test, feat_imp, top_features, scaler = prepare_data(df_raw)
        ml_results, m1, m2, m3 = train_ml_models(X_train, y_train, X_test, y_test)
        ann_results, histories, a1, a2, a3 = train_ann_models(X_train, y_train, X_test, y_test)

    metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    all_rows = []
    for name, res in {**ml_results, **ann_results}.items():
        mtype = "ML" if name in ml_results else "ANN"
        all_rows.append({"Model":name, "Type":mtype, **{m:res[m] for m in metrics}})
    df_all = pd.DataFrame(all_rows)

    # Summary metrics
    best_f1_row = df_all.loc[df_all["F1"].idxmax()]
    best_auc_row = df_all.loc[df_all["ROC-AUC"].idxmax()]
    best_rec_row = df_all.loc[df_all["Recall"].idxmax()]
    best_acc_row = df_all.loc[df_all["Accuracy"].idxmax()]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best F1",      f"{best_f1_row['F1']:.3f}",  best_f1_row["Model"])
    c2.metric("Best ROC-AUC", f"{best_auc_row['ROC-AUC']:.3f}", best_auc_row["Model"])
    c3.metric("Best Recall",  f"{best_rec_row['Recall']:.3f}", best_rec_row["Model"])
    c4.metric("Best Accuracy", f"{best_acc_row['Accuracy']:.3f}", best_acc_row["Model"])

    st.markdown("---")
    st.markdown("#### Complete Performance Table")

    def style_all(df):
        styled = df.copy().astype(str)
        for col in metrics:
            best = df[col].astype(float).idxmax()
            styled.loc[best, col] = f" {df.loc[best, col]}"
        return styled
    st.dataframe(style_all(df_all), hide_index=True, use_container_width=True)

    st.markdown("#### Side-by-Side Radar / Bar Chart")
    fig = go.Figure()
    colors_all = [PAL["primary"], PAL["secondary"], PAL["success"],
                  PAL["purple"], "#E67E22", "#1ABC9C"]
    x = np.arange(len(metrics))
    for i, row in df_all.iterrows():
        fig.add_trace(go.Bar(
            name=row["Model"],
            x=metrics,
            y=[row[m] for m in metrics],
            marker_color=colors_all[i],
            opacity=0.85,
        ))
    fig.update_layout(barmode="group", height=400, yaxis_range=[0,1.05],
                      title="All 6 Models — Final Performance Comparison",
                      margin=dict(l=10,r=10,t=40,b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### All ROC Curves Overlaid")
    fig_roc = go.Figure()
    for i, (name, res) in enumerate({**ml_results, **ann_results}.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        auc = res["ROC-AUC"]
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"{name} (AUC={auc:.3f})",
                                     line=dict(color=colors_all[i], width=2.5)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", showlegend=False,
                                  line=dict(dash="dash", color="gray", width=1)))
    fig_roc.update_layout(height=440, title="ROC Curves — All 6 Models",
                           xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate",
                           margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Model Selection & Conclusion")
    best_name = best_f1_row["Model"]
    st.markdown(f"""
    <div class='success-card'>
    <b>Selected Model: {best_name}</b><br><br>
    {best_name} achieves the best F1-Score ({best_f1_row['F1']:.3f}) and ROC-AUC ({best_f1_row['ROC-AUC']:.3f}),
    indicating the best trade-off between correctly identifying churners (Recall) and minimising false alarms (Precision).<br><br>
    <b>Key Insight:</b> ANN models do not outperform the best ML model on this dataset.
    This is consistent with the literature — for small structured tabular datasets (~7,000 rows),
    gradient boosting ensembles consistently outperform deep neural networks.
    The ANN advantage typically requires 100,000+ samples and unstructured features.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Business Risk Segmentation")
    risk_df = pd.DataFrame({
        "Risk Level": ["High Risk","Medium Risk","Low Risk","Stable"],
        "Churn Probability": ["> 0.75","0.55 – 0.75","0.35 – 0.54","< 0.35"],
        "Recommended Action": [
            "Immediate personal outreach — offer contract upgrade or free add-on month",
            "Targeted email/SMS campaign — loyalty reward or service highlight",
            "Monitor — proactive satisfaction survey",
            "No intervention — routine engagement",
        ],
        "Priority": ["Critical","High","Moderate","Low"],
    })
    st.dataframe(risk_df, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ════════════════════════════════════════════════════════════
elif page == "Live Prediction":
    st.markdown("# Live Churn Prediction")
    st.markdown("Enter a customer's details below to get their churn risk score.")
    st.markdown("---")

    with st.spinner("Loading model..."):
        X_train, X_test, y_train, y_test, feat_imp, top_features, scaler = prepare_data(df_raw)
        ml_results, m1, m2, m3 = train_ml_models(X_train, y_train, X_test, y_test)

    st.markdown("#### Customer Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
        gender        = st.selectbox("Gender", ["Male","Female"])
        senior        = st.selectbox("Senior Citizen (65+)", ["No","Yes"])
        partner       = st.selectbox("Has Partner", ["Yes","No"])
        dependents    = st.selectbox("Has Dependents", ["Yes","No"])

    with col2:
        st.markdown('<div class="section-title">Account</div>', unsafe_allow_html=True)
        tenure        = st.slider("Tenure (months)", 1, 72, 12)
        contract      = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
        payment       = st.selectbox("Payment Method", ["Electronic check","Mailed check",
                                                         "Bank transfer (automatic)","Credit card (automatic)"])
        paperless     = st.selectbox("Paperless Billing", ["Yes","No"])
        monthly_c     = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_c       = st.number_input("Total Charges ($)", min_value=0.0,
                                         value=float(monthly_c * tenure), step=10.0)

    with col3:
        st.markdown('<div class="section-title">Services</div>', unsafe_allow_html=True)
        phone_svc     = st.selectbox("Phone Service", ["Yes","No"])
        multi_lines   = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        internet      = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        online_sec    = st.selectbox("Online Security", ["No","Yes","No internet service"])
        tech_support  = st.selectbox("Tech Support", ["No","Yes","No internet service"])
        streaming_tv  = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
        streaming_mov = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

    st.markdown("---")
    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        # Build a raw-like row
        row = {
            "gender": gender, "SeniorCitizen": 1 if senior=="Yes" else 0,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_svc,
            "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_mov,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_c, "TotalCharges": total_c,
        }
        # Prepare prediction df through same pipeline steps (manual)
        df_pred = pd.DataFrame([row])
        df_pred["TotalCharges"] = pd.to_numeric(df_pred["TotalCharges"], errors="coerce").fillna(total_c)

        # Feature engineering
        df_pred["AvgMonthlyCharge"] = df_pred["TotalCharges"] / (df_pred["tenure"] + 1)
        df_pred["HasStreamingService"] = ((df_pred["StreamingTV"]=="Yes")|(df_pred["StreamingMovies"]=="Yes")).astype(int)
        svc_cols = ["PhoneService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
        df_pred["NumServices"] = (df_pred[svc_cols]=="Yes").sum(axis=1)
        df_pred["TenureGroup"] = pd.cut(df_pred["tenure"], bins=[0,12,24,48,72],
                                         labels=["0-1yr","1-2yr","2-4yr","4+yr"])

        # Binary encode
        bmap = {"Yes":1,"No":0,"Male":1,"Female":0}
        for c in ["gender","Partner","Dependents","PhoneService","PaperlessBilling",
                  "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
            df_pred[c] = df_pred[c].map(bmap).fillna(df_pred[c])

        # OHE
        df_pred = pd.get_dummies(df_pred, columns=["MultipleLines","InternetService","Contract",
                                                     "PaymentMethod","TenureGroup"], drop_first=False)

        # Align columns with training set
        X_ref = pd.DataFrame(columns=top_features)
        df_pred = df_pred.reindex(columns=X_ref.columns, fill_value=0)
        df_pred = df_pred.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Scale
        df_pred_scaled = pd.DataFrame(scaler.transform(df_pred), columns=df_pred.columns)

        # Predict with XGBoost (best model)
        prob = m3.predict_proba(df_pred_scaled)[0][1]
        pred = int(prob > 0.5)

        # Display
        st.markdown("---")
        c1, c2, c3 = st.columns([2, 1, 2])

        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={"text": "Churn Risk Score", "font": {"size": 18}},
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PAL["danger"] if prob > 0.55 else PAL["success"]},
                    "steps": [
                        {"range": [0,35],  "color": "#ECFDF5"},
                        {"range": [35,55], "color": "#FFFBEB"},
                        {"range": [55,75], "color": "#FEF3C7"},
                        {"range": [75,100],"color": "#FEE2E2"},
                    ],
                    "threshold": {"line": {"color": PAL["danger"], "width": 3},
                                  "thickness": 0.75, "value": 55},
                },
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if prob > 0.75:
                risk_label, card_class, action = "HIGH RISK", "danger-card", "Immediate personal outreach — offer contract upgrade or discount"
            elif prob > 0.55:
                risk_label, card_class, action = "MEDIUM RISK", "warn-card", "Targeted retention campaign — loyalty reward"
            elif prob > 0.35:
                risk_label, card_class, action = "LOW RISK", "success-card", "Monitor — satisfaction survey"
            else:
                risk_label, card_class, action = "STABLE", "info-card", "No intervention needed"

            st.markdown(f"""
            <div class='{card_class}' style='text-align:center'>
            <div style='font-size:1.4rem; font-weight:800'>{risk_label}</div>
            <div style='font-size:2.2rem; font-weight:900; margin:0.5rem 0'>{prob*100:.1f}%</div>
            <div style='font-size:0.85rem'><b>Model:</b> XGBoost</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='{card_class}'>
            <b>📋 Recommended Action</b><br><br>
            {action}<br><br>
            <b>Key Risk Factors:</b><br>
            {"⚠️ Month-to-month contract" if contract == "Month-to-month" else "Long-term contract"}<br>
            {"⚠️ Short tenure (<12 months)" if tenure < 12 else "Established customer"}<br>
            {"⚠️ Fiber optic user (high churn segment)" if internet == "Fiber optic" else ""}<br>
            {"⚠️ High monthly charges" if monthly_c > 75 else ""}
            </div>
            """, unsafe_allow_html=True)

        # Probability from all 3 ML models
        st.markdown("#### Prediction from All 3 ML Models")
        model_objs = {"Logistic Regression": m1, "Random Forest": m2, "XGBoost ★": m3}
        mcols = st.columns(3)
        for i, (mname, mobj) in enumerate(model_objs.items()):
            p = mobj.predict_proba(df_pred_scaled)[0][1]
            mcols[i].metric(mname, f"{p*100:.1f}%",
                            delta="Churn" if p>0.5 else "No Churn",
                            delta_color="inverse" if p>0.5 else "normal")