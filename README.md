# Explainable AI (XAI) for Diabetes Prediction using SHAP & LIME

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-orange.svg)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular, production-ready Explainable AI (XAI) framework evaluating a **Random Forest Classifier** trained on the **Pima Indians Diabetes Dataset** using **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations).

---

## 📌 Key Architectural Highlights
* **Zero Data Leakage:** Missing values (physiologically implausible zeros in Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI) are imputed strictly within a `scikit-learn` `Pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(...))` trained solely on `X_train`.
* **Decoupled Architecture:** Core logic is modularized across `src.data`, `src.model`, and `src.explainers` to support reusable Python scripting, batch diagnostics, and interactive notebooks.
* **Unified Diagnostic Engine (`PatientExplainer`):** Exposes a standardized single-patient explanation interface returning structured JSON/dictionary reports of prediction probabilities, SHAP Shapley values, LIME surrogate conditions, and attribution concordance metrics.
* **Borderline Uncertainty Diagnosis:** Algorithmic detection of high-uncertainty patients near the decision boundary ($|P(y=1) - 0.5|$) with side-by-side comparative visualizations.

---

## 📁 Repository Structure
```text
xai-diabetes-shap-lime/
├── data/
│   └── pima-indians-diabetes.data.csv    # Local fallback dataset (768 patient records)
├── notebooks/
│   └── exploratory_lab.ipynb             # Streamlined 4-step modular lab notebook
├── src/
│   ├── __init__.py
│   ├── data.py                           # Ingestion, cleaning, and leakage-free splitting
│   ├── model.py                          # Pipeline assembly, training, evaluation & persistence
│   └── explainers/
│       ├── __init__.py                   # Package exports (PatientExplainer, TreeSHAP, LIME)
│       ├── shap_engine.py                # TreeSHAPEngine wrapper & global/local plots
│       ├── lime_engine.py                # LIMETabularEngine & safe discretizer patching
│       ├── patient_explainer.py          # Unified multi-method PatientExplainer service
│       └── visualization.py              # Dual-axis comparator & borderline patient detection
├── SHAP_LIME_Lab_Diabetes ( 727824TUAM005 ).ipynb # Original hands-on lab notebook
├── requirements.txt                      # Pinned production dependencies
├── .gitignore                            # Environment, checkpoint, and figure ignores
└── README.md                             # Architectural overview and documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Aditthan-07/xai-diabetes-shap-lime.git
cd xai-diabetes-shap-lime

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Interactive Notebooks
Launch Jupyter to explore either the streamlined modular notebook or the classic guided lab:
```bash
jupyter lab
```
* **Modular Lab:** `notebooks/exploratory_lab.ipynb`
* **Guided Lab:** `SHAP_LIME_Lab_Diabetes ( 727824TUAM005 ).ipynb`

---

## 💻 Programmatic Usage Examples

### 1. Train the Leak-Free Pipeline
```python
from src.data import get_train_test_split, FEATURE_NAMES
from src.model import build_pipeline, train_pipeline, evaluate_pipeline

# 1. Ingest and split dataset (Zero data leakage)
X_train, X_test, y_train, y_test = get_train_test_split(random_state=42)

# 2. Build and train pipeline
pipeline = build_pipeline(n_estimators=100, random_state=42)
trained_pipeline = train_pipeline(pipeline, X_train, y_train)

# 3. Evaluate performance
metrics = evaluate_pipeline(trained_pipeline, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
```

### 2. Global Explanations with `TreeSHAPEngine`
```python
from src.explainers import TreeSHAPEngine

# Initialize SHAP engine from fitted pipeline
shap_engine = TreeSHAPEngine(trained_pipeline, feature_names=FEATURE_NAMES)
shap_values = shap_engine.compute_shap_values(X_test)

# Generate summary plot
shap_engine.plot_summary(shap_values, X_test, plot_type='dot')
```

### 3. Patient Diagnosis with Unified `PatientExplainer`
```python
from src.explainers import PatientExplainer, find_most_uncertain_patient

# Extract imputed training features for LIME background statistics
imputer = trained_pipeline.named_steps['imputer']
X_train_imputed = imputer.transform(X_train)

# Initialize unified explainer
explainer = PatientExplainer(
    pipeline_or_model=trained_pipeline,
    training_data=X_train_imputed,
    feature_names=FEATURE_NAMES
)

# Identify most uncertain borderline patient sample
uncertain_info = find_most_uncertain_patient(trained_pipeline, X_test, y_test=y_test)
patient_features = uncertain_info['features']

# Generate standardized multi-method report
report = explainer.explain_patient(patient_features)
print(f"Prediction: {report['predicted_class']} (Prob={report['prob_diabetic']:.3f})")
print(f"Top SHAP Factor: {report['top_shap_feature']}")
print(f"Top LIME Factor: {report['top_lime_condition']}")
print(f"Methods Agree: {report['top_feature_agreement']}")

# Plot side-by-side comparative chart
fig, _ = explainer.plot_comparison(uncertain_info['index'], patient_features)
```

---

## 🔬 Dataset Clinical Variables
* `Pregnancies`: Number of times pregnant
* `Glucose`: Plasma glucose concentration (2-hour OGTT, mg/dL)
* `BloodPressure`: Diastolic blood pressure (mm Hg)
* `SkinThickness`: Triceps skin-fold thickness (mm)
* `Insulin`: 2-hour serum insulin (μU/mL)
* `BMI`: Body mass index (kg/m²)
* `DiabetesPedigreeFunction`: Genetic risk score based on family history
* `Age`: Age in years
* `Outcome`: Class label (0: Non-Diabetic, 1: Diabetic)
