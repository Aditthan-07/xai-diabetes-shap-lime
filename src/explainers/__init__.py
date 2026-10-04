"""
Explainability engines, unified diagnostic services, and visualization tools.
"""

from src.explainers.shap_engine import TreeSHAPEngine
from src.explainers.lime_engine import LIMETabularEngine
from src.explainers.patient_explainer import PatientExplainer
from src.explainers.visualization import (
    plot_shap_vs_lime,
    find_most_uncertain_patient,
)

__all__ = [
    "TreeSHAPEngine",
    "LIMETabularEngine",
    "PatientExplainer",
    "plot_shap_vs_lime",
    "find_most_uncertain_patient",
]
