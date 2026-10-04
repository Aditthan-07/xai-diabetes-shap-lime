"""
Streamlit Clinical Decision Support & Explainable AI (XAI) Dashboard.
Real-time diabetes risk prediction with live SHAP and LIME explanations.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import (
    FEATURE_NAMES,
    clean_missing_values,
    get_train_test_split,
    load_raw_data,
)
from src.model import build_pipeline, evaluate_pipeline, train_pipeline
from src.explainers.patient_explainer import PatientExplainer
from src.explainers.shap_engine import TreeSHAPEngine
from src.explainers.visualization import find_most_uncertain_patient

# ── Streamlit Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Clinical XAI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached Pipeline & Explainers Initialization ────────────────────────────────
@st.cache_resource(show_spinner="Training model pipeline and preparing XAI engines...")
def get_model_and_explainers() -> Tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data, train leak-free pipeline, and cache explainers.
    """
    raw_df = load_raw_data()
    cleaned_df = clean_missing_values(raw_df)

    X_train, X_test, y_train, y_test = get_train_test_split(
        cleaned_df, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(n_estimators=100, random_state=42)
    pipeline = train_pipeline(pipeline, X_train, y_train)

    imputer = pipeline.named_steps["imputer"]
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    shap_engine = TreeSHAPEngine(pipeline, feature_names=FEATURE_NAMES)
    explainer = PatientExplainer(
        pipeline_or_model=pipeline,
        training_data=X_train_imp,
        feature_names=FEATURE_NAMES,
        random_state=42,
    )

    return pipeline, explainer, shap_engine, X_train_imp, X_test_imp, y_train, y_test


pipeline, explainer, shap_engine, X_train_imp, X_test_imp, y_train, y_test = (
    get_model_and_explainers()
)

# ── Sidebar Clinical Controls ──────────────────────────────────────────────────
st.sidebar.title("🩺 Patient Controls")
st.sidebar.markdown(
    "Adjust clinical variables or select a diagnostic profile to simulate model inference."
)

# Borderline Patient Detection
uncertain_patient = find_most_uncertain_patient(pipeline, X_test_imp, y_test=y_test)
hardest_features = uncertain_patient["features"]

# Session State for patient features
default_features: Dict[str, float] = {
    "Pregnancies": 3.0,
    "Glucose": 120.0,
    "BloodPressure": 70.0,
    "SkinThickness": 25.0,
    "Insulin": 105.0,
    "BMI": 31.0,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 33.0,
}

if "patient_input" not in st.session_state:
    st.session_state["patient_input"] = default_features.copy()

# Presets & Boundary Explorer
st.sidebar.markdown("### Quick Diagnostic Presets")
col_p1, col_p2 = st.sidebar.columns(2)
if col_p1.button("🎯 Borderline Patient", help="Load test set patient closest to 50% uncertainty"):
    for idx, name in enumerate(FEATURE_NAMES):
        st.session_state["patient_input"][name] = float(hardest_features[idx])
    st.sidebar.success(f"Loaded Patient #{uncertain_patient['index']} (Prob ~ 50%)")

if col_p2.button("🔄 Reset Default"):
    st.session_state["patient_input"] = default_features.copy()

# Clinical Feature Sliders
inputs = {}
inputs["Glucose"] = st.sidebar.slider(
    "Glucose (mg/dL)",
    min_value=40.0,
    max_value=220.0,
    value=float(st.session_state["patient_input"]["Glucose"]),
    step=1.0,
    help="Plasma glucose concentration from a 2-hour oral glucose tolerance test.",
)

inputs["BMI"] = st.sidebar.slider(
    "Body Mass Index (BMI)",
    min_value=15.0,
    max_value=60.0,
    value=float(st.session_state["patient_input"]["BMI"]),
    step=0.1,
    help="Weight in kg / (height in m)^2.",
)

inputs["Age"] = st.sidebar.slider(
    "Age (years)",
    min_value=21.0,
    max_value=85.0,
    value=float(st.session_state["patient_input"]["Age"]),
    step=1.0,
)

inputs["Pregnancies"] = st.sidebar.slider(
    "Pregnancies",
    min_value=0.0,
    max_value=17.0,
    value=float(st.session_state["patient_input"]["Pregnancies"]),
    step=1.0,
)

inputs["BloodPressure"] = st.sidebar.slider(
    "Blood Pressure (mm Hg)",
    min_value=40.0,
    max_value=130.0,
    value=float(st.session_state["patient_input"]["BloodPressure"]),
    step=1.0,
)

inputs["Insulin"] = st.sidebar.slider(
    "2-Hour Serum Insulin (μU/mL)",
    min_value=14.0,
    max_value=850.0,
    value=float(st.session_state["patient_input"]["Insulin"]),
    step=1.0,
)

inputs["SkinThickness"] = st.sidebar.slider(
    "Skin Thickness (mm)",
    min_value=5.0,
    max_value=99.0,
    value=float(st.session_state["patient_input"]["SkinThickness"]),
    step=1.0,
)

inputs["DiabetesPedigreeFunction"] = st.sidebar.slider(
    "Diabetes Pedigree Function",
    min_value=0.05,
    max_value=2.50,
    value=float(st.session_state["patient_input"]["DiabetesPedigreeFunction"]),
    step=0.01,
    help="Genetic diabetes risk score based on family pedigree.",
)

# Assemble input vector in order of FEATURE_NAMES
patient_vector = np.array([inputs[name] for name in FEATURE_NAMES])

# ── Main Dashboard Header & Real-time Prediction Meter ─────────────────────────
st.title("🩺 Explainable AI Diabetes Decision Support")
st.markdown(
    "Real-time clinical machine learning with **zero data leakage**, integrating "
    "**SHAP** (Shapley game-theoretic attributions) and **LIME** (Local surrogate explanations)."
)

# Generate Unified Explanation Report
with st.spinner("Computing real-time prediction and attributions..."):
    explanation_report = explainer.explain_patient(patient_vector)

prob_diabetic = explanation_report["prob_diabetic"]
prob_non_diabetic = explanation_report["prob_non_diabetic"]
pred_class = explanation_report["predicted_class"]

# Prediction Score Cards
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.5, 1.5])

with c1:
    if prob_diabetic >= 0.65:
        st.metric("Predicted Outcome", "Diabetic", delta="High Risk", delta_color="inverse")
    elif prob_diabetic <= 0.35:
        st.metric("Predicted Outcome", "Non-Diabetic", delta="Low Risk", delta_color="normal")
    else:
        st.metric("Predicted Outcome", "Borderline", delta="Uncertain", delta_color="off")

with c2:
    st.metric("Diabetes Probability", f"{prob_diabetic:.1%}")

with c3:
    st.markdown("**Risk Score Meter**")
    st.progress(float(prob_diabetic))
    if prob_diabetic >= 0.65:
        st.caption("🔴 High probability of diabetes onset.")
    elif prob_diabetic <= 0.35:
        st.caption("🟢 Low probability of diabetes onset.")
    else:
        st.caption("🟡 Near decision boundary (50%) — Review XAI attributions carefully.")

with c4:
    agreement = explanation_report["top_feature_agreement"]
    st.metric(
        "XAI Concordance",
        "Concordant" if agreement else "Discordant",
        delta="SHAP & LIME Agree" if agreement else "Methods Differ",
        delta_color="normal" if agreement else "inverse",
    )

st.divider()

# ── Diagnostic Tabs ────────────────────────────────────────────────────────────
tab_compare, tab_shap, tab_lime, tab_global = st.tabs(
    [
        "⚖️ Dual-Method Comparison",
        "🎯 SHAP Deep-Dive",
        "🍋 LIME Surrogate Rules",
        "🌐 Global Cohort Insights",
    ]
)

# ── Tab 1: Dual-Method Comparative Attribution ─────────────────────────────────
with tab_compare:
    st.subheader("Side-by-Side Attribution: SHAP vs. LIME")
    st.markdown(
        "Comparing additive Shapley values against local linear surrogate coefficients. "
        "Red bars increase diabetic risk, while green bars decrease risk."
    )

    fig_comp, comp_df = explainer.plot_comparison(
        patient_idx=0,
        patient_features=patient_vector,
        num_features=8,
        show=False,
    )
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"**Top SHAP Biomarker:** `{explanation_report['top_shap_feature']}`")
        top_shap_attr = explanation_report["shap_attributions"][0]
        st.write(
            f"Contributes **{top_shap_attr['shap_value']:+.3f}** to log-odds "
            f"({top_shap_attr['impact']})."
        )

    with col_t2:
        st.markdown(f"**Top LIME Condition:** `{explanation_report['top_lime_condition']}`")
        top_lime_attr = explanation_report["lime_attributions"][0]
        st.write(
            f"Surrogate weight **{top_lime_attr['weight']:+.3f}** "
            f"({top_lime_attr['impact']})."
        )

# ── Tab 2: SHAP Deep-Dive (Local Waterfall) ────────────────────────────────────
with tab_shap:
    st.subheader("Patient Local SHAP Waterfall")
    st.markdown(
        "Displays how each clinical measurement moves the prediction away from the base cohort expectation."
    )

    pred_label = explanation_report["predicted_label"]
    fig_waterfall = shap_engine.plot_waterfall(
        patient_idx=0,
        shap_values=np.array([[a["shap_value"] for a in explanation_report["shap_attributions"]]]),
        X=patient_vector.reshape(1, -1),
        pred_prob=prob_diabetic,
        pred_label=pred_label,
        show=False,
    )
    st.pyplot(fig_waterfall)
    plt.close(fig_waterfall)

    st.markdown("#### Tabular Feature SHAP Impacts")
    shap_table = pd.DataFrame(explanation_report["shap_attributions"])[
        ["feature", "value", "shap_value", "impact"]
    ]
    st.dataframe(shap_table, use_container_width=True)

# ── Tab 3: LIME Surrogate Rules ────────────────────────────────────────────────
with tab_lime:
    st.subheader("LIME Local Decision Rules")
    st.markdown(
        "LIME perturbs the patient sample within the empirical training distribution to fit a human-interpretable sparse linear surrogate."
    )

    raw_lime = explanation_report["_raw_lime_explanation"]
    fig_lime = explainer.lime_engine.plot_explanation_bar(
        explanation=raw_lime,
        num_features=8,
        show=False,
    )
    st.pyplot(fig_lime)
    plt.close(fig_lime)

    st.markdown("#### Discretized Rule Conditions")
    lime_table = explainer.lime_engine.get_explanation_table(raw_lime)
    st.dataframe(lime_table, use_container_width=True)

# ── Tab 4: Global Cohort Insights ──────────────────────────────────────────────
with tab_global:
    st.subheader("Cohort-Level Feature Importance (Test Set)")
    st.markdown(
        "Global feature importance derived across the 20% stratified test split (154 patients)."
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**Mean Absolute Importance Ranking**")
        shap_test_values = shap_engine.compute_shap_values(X_test_imp)
        fig_bar, imp_df = shap_engine.plot_feature_importance_bar(shap_test_values, show=False)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with col_g2:
        st.markdown("**Summary Beeswarm Plot**")
        fig_sum = shap_engine.plot_summary(shap_test_values, X_test_imp, plot_type="dot", show=False)
        st.pyplot(fig_sum)
        plt.close(fig_sum)

    st.markdown("**Glucose Non-Linear Interaction (Dependence Plot)**")
    fig_dep = shap_engine.plot_dependence("Glucose", shap_test_values, X_test_imp, show=False)
    st.pyplot(fig_dep)
    plt.close(fig_dep)
