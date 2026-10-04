"""
Unified PatientExplainer service combining SHAP and LIME into standardized attributions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.explainers.shap_engine import TreeSHAPEngine
from src.explainers.lime_engine import LIMETabularEngine
from src.explainers.visualization import plot_shap_vs_lime


class PatientExplainer:
    """
    Unified multi-method explainability service for single patient predictions.
    Computes model predictions, SHAP Shapley values, and LIME surrogate weights,
    returning a structured, standardized diagnostic report.
    """

    def __init__(
        self,
        pipeline_or_model: Any,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize PatientExplainer with model and training background.

        Parameters
        ----------
        pipeline_or_model : Any
            Fitted scikit-learn Pipeline or model estimator.
        training_data : np.ndarray
            Fitted/imputed training features used for LIME perturbations.
        feature_names : Optional[List[str]]
            Clinical feature names.
        class_names : Optional[List[str]]
            Class target names. Defaults to ['Non-Diabetic', 'Diabetic'].
        random_state : int
            Reproducibility seed.
        """
        self.pipeline_or_model = pipeline_or_model
        self.feature_names = feature_names or [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ]
        self.class_names = class_names or ["Non-Diabetic", "Diabetic"]

        # Initialize underlying engines
        self.shap_engine = TreeSHAPEngine(
            model_or_pipeline=pipeline_or_model,
            feature_names=self.feature_names,
        )
        self.lime_engine = LIMETabularEngine(
            training_data=training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            random_state=random_state,
        )

    def _get_predict_fn(self):
        if hasattr(self.pipeline_or_model, "predict_proba"):
            return self.pipeline_or_model.predict_proba
        raise ValueError("Model must expose a .predict_proba method.")

    def explain_patient(
        self,
        patient_features: np.ndarray,
        num_features: int = 8,
        num_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Generate a unified attribution dictionary for a given patient.

        Parameters
        ----------
        patient_features : np.ndarray
            1D array of patient clinical measurements.
        num_features : int
            Number of top features to report in summaries.
        num_samples : int
            Number of perturbation samples for LIME surrogate fitting.

        Returns
        -------
        Dict[str, Any]
            Standardized explanation dictionary containing prediction metrics,
            SHAP attributions, LIME attributions, and method agreement status.
        """
        patient_row = np.asarray(patient_features).reshape(1, -1)
        patient_1d = patient_row.flatten()

        # 1. Model Prediction
        predict_fn = self._get_predict_fn()
        probas = predict_fn(patient_row)[0]
        prob_non_diabetic = float(probas[0])
        prob_diabetic = float(probas[1])
        predicted_label = int(np.argmax(probas))
        predicted_class = self.class_names[predicted_label]

        # 2. SHAP Values
        raw_shap = self.shap_engine.compute_shap_values(patient_row)
        shap_class1 = self.shap_engine.get_class_shap_values(raw_shap, class_idx=1)[0]

        shap_attributions = []
        for name, val, s_val in zip(self.feature_names, patient_1d, shap_class1):
            shap_attributions.append(
                {
                    "feature": name,
                    "value": float(val),
                    "shap_value": float(s_val),
                    "impact": "increases_risk" if s_val > 0 else "decreases_risk",
                }
            )
        # Sort by absolute impact
        shap_attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # 3. LIME Explanation
        # If pipeline has imputer, pass imputed features to LIME instance
        prepared_1d = self.shap_engine.prepare_data(patient_row).flatten()
        lime_explanation = self.lime_engine.explain_instance(
            data_row=prepared_1d,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

        lime_attributions = []
        for condition, weight in lime_explanation.as_list():
            lime_attributions.append(
                {
                    "condition": condition,
                    "weight": float(weight),
                    "impact": "supports_diabetic" if weight > 0 else "contradicts_diabetic",
                }
            )

        # 4. Agreement Check (Top feature)
        top_shap_feat = shap_attributions[0]["feature"]
        # Check if the top SHAP feature name is contained in the top LIME condition
        top_lime_cond = lime_attributions[0]["condition"] if lime_attributions else ""
        features_agree = top_shap_feat.lower() in top_lime_cond.lower()

        return {
            "patient_features": dict(zip(self.feature_names, [float(x) for x in patient_1d])),
            "predicted_label": predicted_label,
            "predicted_class": predicted_class,
            "prob_diabetic": prob_diabetic,
            "prob_non_diabetic": prob_non_diabetic,
            "shap_attributions": shap_attributions,
            "lime_attributions": lime_attributions,
            "top_shap_feature": top_shap_feat,
            "top_lime_condition": top_lime_cond,
            "top_feature_agreement": features_agree,
            "_raw_shap_values": shap_class1,
            "_raw_lime_explanation": lime_explanation,
        }

    def plot_comparison(
        self,
        patient_idx: int,
        patient_features: np.ndarray,
        num_features: int = 8,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Generate and render the side-by-side SHAP vs LIME comparison figure.
        """
        explanation = self.explain_patient(
            patient_features=patient_features,
            num_features=num_features,
        )
        return plot_shap_vs_lime(
            patient_idx=patient_idx,
            shap_patient_values=explanation["_raw_shap_values"],
            lime_explanation=explanation["_raw_lime_explanation"],
            feature_names=self.feature_names,
            top_n=num_features,
            save_path=save_path,
            show=show,
        )
