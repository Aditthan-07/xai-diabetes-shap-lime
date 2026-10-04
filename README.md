# Explainable AI (XAI) for Diabetes Prediction using SHAP & LIME

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![XAI](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An Explainable AI (XAI) lab and benchmarking framework evaluating a **Random Forest Classifier** trained on the **Pima Indians Diabetes Dataset** using **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations).

---

## 📌 Project Overview
* **Domain:** Healthcare Machine Learning & Clinical Decision Support.
* **Objective:** Predict diabetes onset and provide granular local and global explanations for model decisions.
* **Leakage-Free Preprocessing:** Missing values (physiologically implausible zeros) are imputed using a `scikit-learn` `Pipeline` with `SimpleImputer(strategy='median')` strictly fitted on training splits.
* **Interpretability:**
  * **Global Importance:** SHAP Summary (Beehive) and Mean Absolute Bar plots.
  * **Local Explanations:** SHAP Waterfall / Force plots and LIME rule-based surrogate explanations.
  * **Comparative Analysis:** Side-by-side SHAP vs. LIME feature attribution and boundary uncertainty analysis.

---

## 📁 Repository Structure
```text
xai-diabetes-shap-lime/
├── data/
│   └── pima-indians-diabetes.data.csv    # Local fallback dataset (768 records, 8 features)
├── SHAP_LIME_Lab_Diabetes ( 727824TUAM005 ).ipynb # Main interactive laboratory notebook
├── requirements.txt                      # Pinned production dependencies
├── .gitignore                            # Environment, checkpoint, and figure ignores
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

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

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook
Launch Jupyter Notebook or JupyterLab:
```bash
jupyter lab
```
Open `SHAP_LIME_Lab_Diabetes ( 727824TUAM005 ).ipynb` and run the cells in sequence.

---

## 🔬 Dataset Information
The Pima Indians Diabetes Dataset contains 8 clinical diagnostic measurements from female patients of Pima Indian heritage:
* `Pregnancies`: Number of times pregnant
* `Glucose`: Plasma glucose concentration (2-hour OGTT, mg/dL)
* `BloodPressure`: Diastolic blood pressure (mm Hg)
* `SkinThickness`: Triceps skin-fold thickness (mm)
* `Insulin`: 2-hour serum insulin (μU/mL)
* `BMI`: Body mass index (kg/m²)
* `DiabetesPedigreeFunction`: Genetic family history risk score
* `Age`: Age in years
* `Outcome`: Class variable (0: Non-Diabetic, 1: Diabetic)
