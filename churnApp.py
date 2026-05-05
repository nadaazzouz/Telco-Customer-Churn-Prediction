"""
Churn Prediction - Streamlit App
Run: streamlit run churn_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="wide")
SEED = 42

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        df = pd.read_csv("Telco-Customer-Churn.csv")
    return df

@st.cache_data
def preprocess(df):
    d = df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d["TotalCharges"].fillna(d["TotalCharges"].median(), inplace=True)
    d.drop(columns=["customerID"], inplace=True)
    d["AvgMonthlyCharge"] = d["TotalCharges"] / (d["tenure"] + 1)
    d["NumServices"] = (d[["PhoneService","OnlineSecurity","OnlineBackup",
                             "DeviceProtection","TechSupport",
                             "StreamingTV","StreamingMovies"]] == "Yes").sum(axis=1)
    d["Churn"] = (d["Churn"] == "Yes").astype(int)
    yes_no = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for c in ["gender","Partner","Dependents","PhoneService","PaperlessBilling",
              "OnlineSecurity","OnlineBackup","DeviceProtection",
              "TechSupport","StreamingTV","StreamingMovies"]:
        d[c] = d[c].map(yes_no).fillna(0)
    d = pd.get_dummies(d, columns=["MultipleLines","InternetService",
                                    "Contract","PaymentMethod"], drop_first=False)
    y = d["Churn"]
    X = d.drop(columns=["Churn"]).apply(pd.to_numeric, errors="coerce").fillna(0)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)
    X_train_bal, y_train_bal = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    return X_train_bal, X_test, y_train_bal, y_test, scaler, X_scaled.columns.tolist()

@st.cache_resource
def train_models(_X_train, _y_train, _X_test, _y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
        "XGBoost":             XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=SEED),
    }
    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(_X_train, _y_train)
        y_pred = model.predict(_X_test)
        y_prob = model.predict_proba(_X_test)[:, 1]
        results[name] = {
            "Accuracy":  round(accuracy_score(_y_test, y_pred), 4),
            "Precision": round(precision_score(_y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(_y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(_y_test, y_pred, zero_division=0), 4),
            "ROC-AUC":   round(roc_auc_score(_y_test, y_prob), 4),
            "y_pred": y_pred,
            "y_prob":  y_prob,
        }
        trained[name] = model
    return results, trained

# Load everything once
df_raw = load_data()
with st.spinner("Preprocessing and training models..."):
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess(df_raw)
    results, trained_models = train_models(X_train, y_train, X_test, y_test)

# Sidebar
st.sidebar.title("📡 Churn Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    " Home",
    " Data & EDA",
    " Model Results",
    " Live Prediction",
])
st.sidebar.markdown("---")
st.sidebar.caption("IBM Telco · 7,043 customers · CRISP-DM")

# ── PAGE 1: HOME ──────────────────────────────────────────
if page == " Home":
    st.title(" Customer Churn Prediction")
    st.markdown("End-to-end binary classification pipeline for telecom customer retention.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{len(df_raw):,}")
    c2.metric("Churn Rate", f"{(df_raw['Churn']=='Yes').mean()*100:.1f}%")
    c3.metric("Features Used", "20")
    c4.metric("Models Trained", "3")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Business Problem")
        st.info("How can we identify customers at high risk of cancelling their service "
                "in the next month so the marketing team can target them with specific retention offers?")
        st.subheader("Validated Hypotheses")
        st.success("H1 — Month-to-month contracts: 42.7% churn vs 2.8% (2-year)")
        st.success("H2 — Fiber optic: 41.9% churn vs 18.9% (DSL)")
        st.success("H3 — Higher monthly charges correlate with churn (r ≈ +0.19)")

    with col_b:
        st.subheader("CRISP-DM Pipeline")
        for i, (phase, desc) in enumerate([
            ("Business Understanding", "Define churn prediction objective"),
            ("Data Understanding",     "EDA and hypothesis testing"),
            ("Data Preparation",       "Clean, encode, SMOTE balance"),
            ("Modeling",               "Train 3 ML classifiers"),
            ("Evaluation",             "Compare Accuracy, F1, ROC-AUC"),
            ("Deployment",             "This Streamlit application"),
        ], 1):
            st.markdown(f"**{i}. {phase}** — {desc}")

# ── PAGE 2: DATA & EDA ────────────────────────────────────
elif page == " Data & EDA":
    st.title(" Data & Exploratory Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Dataset", "Univariate", "Bivariate"])

    with tab1:
        st.subheader("Raw Data Sample")
        st.dataframe(df_raw.head(50), use_container_width=True)

        st.subheader("Class Distribution")
        vc = df_raw["Churn"].value_counts().reset_index()
        vc.columns = ["Churn", "Count"]
        fig = px.pie(vc, names="Churn", values="Count", hole=0.5,
                     color="Churn", color_discrete_map={"No":"#27AE60","Yes":"#E74C3C"},
                     title="Churn vs No Churn")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Descriptive Statistics")
        df_num = df_raw[["tenure","MonthlyCharges","TotalCharges"]].copy()
        df_num["TotalCharges"] = pd.to_numeric(df_num["TotalCharges"], errors="coerce")
        st.dataframe(df_num.describe().round(2), use_container_width=True)

    with tab2:
        st.subheader("Numerical Feature Distribution")
        feat = st.selectbox("Select feature", ["tenure","MonthlyCharges","TotalCharges"])
        df_u = df_raw.copy()
        df_u["TotalCharges"] = pd.to_numeric(df_u["TotalCharges"], errors="coerce")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_u, x=feat, nbins=40,
                               color_discrete_sequence=["#2E75B6"],
                               title=f"Histogram — {feat}")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(df_u, y=feat, color_discrete_sequence=["#ED7D31"],
                         title=f"Boxplot — {feat}")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Categorical Feature Distribution")
        cat = st.selectbox("Select feature",
                           ["Contract","InternetService","PaymentMethod",
                            "TechSupport","OnlineSecurity","gender","PaperlessBilling"])
        vc2 = df_raw[cat].value_counts().reset_index()
        vc2.columns = [cat, "Count"]
        fig = px.bar(vc2, x=cat, y="Count", color=cat,
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     title=f"Distribution — {cat}")
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        df_b = df_raw.copy()
        df_b["TotalCharges"] = pd.to_numeric(df_b["TotalCharges"], errors="coerce")
        df_b["Churn_bin"] = (df_b["Churn"] == "Yes").astype(int)

        st.subheader("Numerical Feature vs Churn")
        num_feat = st.selectbox("Numerical feature",
                                ["tenure","MonthlyCharges","TotalCharges"])
        fig = px.histogram(df_b.dropna(subset=[num_feat]),
                           x=num_feat, color="Churn", barmode="overlay", nbins=40,
                           color_discrete_map={"No":"#27AE60","Yes":"#E74C3C"},
                           opacity=0.7, title=f"{num_feat} by Churn")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Churn Rate by Categorical Feature")
        cat_feat = st.selectbox("Categorical feature",
                                ["Contract","InternetService","PaymentMethod",
                                 "TechSupport","OnlineSecurity"])
        rate = (df_b.groupby(cat_feat)["Churn_bin"]
                    .mean().mul(100).reset_index()
                    .rename(columns={"Churn_bin":"Churn Rate (%)"}))
        fig = px.bar(rate, x=cat_feat, y="Churn Rate (%)",
                     color="Churn Rate (%)", color_continuous_scale="Reds",
                     title=f"Churn Rate by {cat_feat}", text_auto=".1f")
        fig.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

# ── PAGE 3: MODEL RESULTS ─────────────────────────────────
elif page == " Model Results":
    st.title(" Model Results")
    st.markdown("---")

    metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]

    st.subheader("Performance Comparison")
    rows = [{"Model": name, **{m: res[m] for m in metrics}}
            for name, res in results.items()]
    df_res = pd.DataFrame(rows)
    st.dataframe(
        df_res.set_index("Model").style.highlight_max(axis=0, color="#D5F5E3"),
        use_container_width=True,
    )

    st.subheader("Metric Bar Chart")
    fig = go.Figure()
    for color, (name, res) in zip(["#2E75B6","#ED7D31","#27AE60"], results.items()):
        fig.add_trace(go.Bar(name=name, x=metrics,
                             y=[res[m] for m in metrics],
                             marker_color=color))
    fig.update_layout(barmode="group", height=380,
                      yaxis_range=[0,1.05], yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Detail")
    selected = st.selectbox("Select model", list(results.keys()))
    res = results[selected]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  res["Accuracy"])
    c2.metric("Precision", res["Precision"])
    c3.metric("Recall",    res["Recall"])
    c4.metric("F1",        res["F1"])
    c5.metric("ROC-AUC",   res["ROC-AUC"])

    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_test, res["y_pred"])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=["Pred: No Churn","Pred: Churn"],
                           y=["Act: No Churn","Act: Churn"],
                           title=f"Confusion Matrix — {selected}")
        fig_cm.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"AUC = {res['ROC-AUC']:.3f}",
                                     line=dict(color="#2E75B6", width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                     line=dict(dash="dash", color="gray"),
                                     showlegend=False))
        fig_roc.update_layout(title=f"ROC Curve — {selected}", height=350,
                               xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

    best = df_res.loc[df_res["F1"].idxmax(), "Model"]
    st.success(f"Best model by F1-Score: **{best}**")

# ── PAGE 4: LIVE PREDICTION ───────────────────────────────
elif page == " Live Prediction":
    st.title(" Live Churn Prediction")
    st.markdown("Fill in customer details and click **Predict**.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Demographics**")
        gender     = st.selectbox("Gender", ["Male","Female"])
        senior     = st.selectbox("Senior Citizen", ["No","Yes"])
        partner    = st.selectbox("Partner", ["Yes","No"])
        dependents = st.selectbox("Dependents", ["Yes","No"])

    with col2:
        st.markdown("**Account**")
        tenure    = st.slider("Tenure (months)", 1, 72, 12)
        contract  = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        payment   = st.selectbox("Payment Method",
                                  ["Electronic check","Mailed check",
                                   "Bank transfer (automatic)","Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])
        monthly_c = st.slider("Monthly Charges ($)", 18, 120, 65)
        total_c   = st.number_input("Total Charges ($)", min_value=0.0,
                                     value=float(monthly_c * tenure), step=10.0)

    with col3:
        st.markdown("**Services**")
        phone   = st.selectbox("Phone Service", ["Yes","No"])
        multi   = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        internet= st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        sec     = st.selectbox("Online Security", ["No","Yes","No internet service"])
        backup  = st.selectbox("Online Backup", ["No","Yes","No internet service"])
        device  = st.selectbox("Device Protection", ["No","Yes","No internet service"])
        tech    = st.selectbox("Tech Support", ["No","Yes","No internet service"])
        tv      = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
        movies  = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

    st.markdown("---")
    if st.button("Predict Churn Risk", type="primary", use_container_width=True):

        row = {
            "gender":           1 if gender == "Male" else 0,
            "SeniorCitizen":    1 if senior == "Yes" else 0,
            "Partner":          1 if partner == "Yes" else 0,
            "Dependents":       1 if dependents == "Yes" else 0,
            "tenure":           tenure,
            "PhoneService":     1 if phone == "Yes" else 0,
            "OnlineSecurity":   1 if sec == "Yes" else 0,
            "OnlineBackup":     1 if backup == "Yes" else 0,
            "DeviceProtection": 1 if device == "Yes" else 0,
            "TechSupport":      1 if tech == "Yes" else 0,
            "StreamingTV":      1 if tv == "Yes" else 0,
            "StreamingMovies":  1 if movies == "Yes" else 0,
            "PaperlessBilling": 1 if paperless == "Yes" else 0,
            "MonthlyCharges":   monthly_c,
            "TotalCharges":     total_c,
            "AvgMonthlyCharge": total_c / (tenure + 1),
            "NumServices": sum([phone=="Yes", sec=="Yes", backup=="Yes",
                                device=="Yes", tech=="Yes", tv=="Yes", movies=="Yes"]),
            "MultipleLines_No":               1 if multi=="No" else 0,
            "MultipleLines_Yes":              1 if multi=="Yes" else 0,
            "MultipleLines_No phone service": 1 if multi=="No phone service" else 0,
            "InternetService_DSL":            1 if internet=="DSL" else 0,
            "InternetService_Fiber optic":    1 if internet=="Fiber optic" else 0,
            "InternetService_No":             1 if internet=="No" else 0,
            "Contract_Month-to-month":        1 if contract=="Month-to-month" else 0,
            "Contract_One year":              1 if contract=="One year" else 0,
            "Contract_Two year":              1 if contract=="Two year" else 0,
            "PaymentMethod_Bank transfer (automatic)": 1 if payment=="Bank transfer (automatic)" else 0,
            "PaymentMethod_Credit card (automatic)":   1 if payment=="Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check":          1 if payment=="Electronic check" else 0,
            "PaymentMethod_Mailed check":              1 if payment=="Mailed check" else 0,
        }

        df_input = (pd.DataFrame([row])
                      .reindex(columns=feature_cols, fill_value=0)
                      .apply(pd.to_numeric, errors="coerce")
                      .fillna(0))
        df_scaled = pd.DataFrame(scaler.transform(df_input), columns=feature_cols)

        st.markdown("### Predictions from all 3 models")
        r1, r2, r3 = st.columns(3)
        for col_out, (name, model) in zip([r1, r2, r3], trained_models.items()):
            prob = model.predict_proba(df_scaled)[0][1]
            label = "Churn" if prob > 0.5 else "No Churn"
            col_out.metric(name, f"{prob*100:.1f}%", label,
                           delta_color="inverse" if prob > 0.5 else "normal")

        st.markdown("---")
        xgb_prob = trained_models["XGBoost"].predict_proba(df_scaled)[0][1]

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(xgb_prob * 100, 1),
            number={"suffix": "%"},
            title={"text": "Churn Risk — XGBoost"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#E74C3C" if xgb_prob > 0.55 else "#27AE60"},
                "steps": [
                    {"range": [0,  35],  "color": "#ECFDF5"},
                    {"range": [35, 55],  "color": "#FFFBEB"},
                    {"range": [55, 75],  "color": "#FEF3C7"},
                    {"range": [75, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": "#E74C3C", "width": 3},
                    "thickness": 0.75, "value": 55,
                },
            },
        ))
        fig_gauge.update_layout(height=300)

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with g2:
            if xgb_prob > 0.75:
                st.error("HIGH RISK\n\nImmediate personal outreach — offer a contract upgrade or discount.")
            elif xgb_prob > 0.55:
                st.warning("MEDIUM RISK\n\nTargeted retention campaign — loyalty reward or service highlight.")
            elif xgb_prob > 0.35:
                st.info("LOW RISK\n\nMonitor the customer — send a proactive satisfaction survey.")
            else:
                st.success("STABLE\n\nNo intervention needed — routine engagement only.")

            st.markdown("**Risk factors:**")
            flags = []
            if contract == "Month-to-month":
                flags.append("Month-to-month contract (42.7% churn segment)")
            if tenure < 12:
                flags.append("Short tenure — new customer (<12 months)")
            if internet == "Fiber optic":
                flags.append("Fiber optic user (41.9% segment churn rate)")
            if monthly_c > 75:
                flags.append("High monthly charges (>$75)")
            if not flags:
                flags.append("No major risk factors detected")
            for f in flags:
                st.markdown(f"- {f}")