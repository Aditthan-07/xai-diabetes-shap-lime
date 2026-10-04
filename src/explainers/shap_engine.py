"""
SHAP Explainer Engine encapsulating TreeExplainer and visualization helpers.
"""

from typing import Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from sklearn.pipeline import Pipeline


class TreeSHAPEngine:
    """
    Encapsulates SHAP TreeExplainer for tree-based diabetes prediction models.
    Supports raw feature inputs through an imputer pipeline step.
    """

    def __init__(
        self,
        model_or_pipeline: Any,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize TreeSHAPEngine.

        Parameters
        ----------
        model_or_pipeline : Any
            Fitted scikit-learn Pipeline containing ('imputer', 'classifier')
            or a direct fitted tree estimator (e.g., RandomForestClassifier).
        feature_names : Optional[List[str]]
            Feature names for interpretation.
        """
        if isinstance(model_or_pipeline, Pipeline):
            self.imputer = model_or_pipeline.named_steps.get("imputer", None)
            self.model = model_or_pipeline.named_steps["classifier"]
        else:
            self.imputer = None
            self.model = model_or_pipeline

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
        self.explainer = shap.TreeExplainer(self.model)

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values if an imputer step was registered.
        """
        if self.imputer is not None:
            return self.imputer.transform(X)
        return X

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for the provided dataset.

        Parameters
        ----------
        X : np.ndarray
            Input features (imputed or raw if pipeline was provided).

        Returns
        -------
        np.ndarray
            SHAP values matrix.
        """
        X_ready = self.prepare_data(X)
        values = self.explainer.shap_values(X_ready)
        return values

    def get_class_shap_values(
        self,
        shap_values: np.ndarray,
        class_idx: int = 1,
    ) -> np.ndarray:
        """
        Extract SHAP values corresponding to the target class (default 1: Diabetic).
        Handles various SHAP version output formats (list vs 3D ndarray).
        """
        if isinstance(shap_values, list):
            return shap_values[class_idx]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            return shap_values[:, :, class_idx]
        return shap_values

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        plot_type: str = "dot",
        class_idx: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Render and optionally save a SHAP summary plot.
        """
        X_ready = self.prepare_data(X)
        class_values = self.get_class_shap_values(shap_values, class_idx=class_idx)

        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            class_values,
            X_ready,
            feature_names=self.feature_names,
            plot_type=plot_type,
            show=False,
        )
        plt.title(
            "SHAP Summary Plot — Global Feature Importance\n(Pima Indians Diabetes Dataset)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_feature_importance_bar(
        self,
        shap_values: np.ndarray,
        class_idx: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """
        Compute mean absolute SHAP values and render a horizontal bar chart.
        """
        class_values = self.get_class_shap_values(shap_values, class_idx=class_idx)
        mean_shap = np.abs(class_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {"Feature": self.feature_names, "Mean |SHAP|": mean_shap}
        ).sort_values("Mean |SHAP|", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(
            importance_df["Feature"],
            importance_df["Mean |SHAP|"],
            color="#378ADD",
            edgecolor="white",
            height=0.6,
        )
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(
            "Global Feature Importance via SHAP\n(Pima Indians Diabetes Dataset)",
            fontsize=13,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, importance_df["Mean |SHAP|"]):
            ax.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, importance_df

    def plot_waterfall(
        self,
        patient_idx: int,
        shap_values: np.ndarray,
        X: np.ndarray,
        pred_prob: float,
        pred_label: int,
        true_label: Optional[int] = None,
        class_idx: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot a custom waterfall / horizontal bar chart showing individual feature contributions.
        """
        X_ready = self.prepare_data(X)
        patient_features = X_ready[patient_idx]
        class_values = self.get_class_shap_values(shap_values, class_idx=class_idx)
        patient_shap = class_values[patient_idx]

        labels = [
            f"{name}={val:.1f}"
            for name, val in zip(self.feature_names, patient_features)
        ]
        shap_df = pd.DataFrame({"Feature": labels, "SHAP": patient_shap}).sort_values(
            "SHAP", key=abs, ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in shap_df["SHAP"]]
        ax.barh(
            shap_df["Feature"],
            shap_df["SHAP"],
            color=colors,
            edgecolor="white",
            height=0.6,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)

        title_pred = "Diabetic" if pred_label == 1 else "Non-Diabetic"
        title_true = (
            f" (True: {'Diabetic' if true_label == 1 else 'Non-Diabetic'})"
            if true_label is not None
            else ""
        )
        ax.set_title(
            f"SHAP Waterfall — Patient #{patient_idx}\n"
            f"Prediction: {title_pred} (prob={pred_prob:.2f}){title_true}",
            fontsize=12,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        red_patch = mpatches.Patch(color="#E24B4A", label="Increases diabetes risk")
        green_patch = mpatches.Patch(color="#1D9E75", label="Decreases diabetes risk")
        ax.legend(handles=[red_patch, green_patch], fontsize=9)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_force(
        self,
        patient_idx: int,
        shap_values: np.ndarray,
        X: np.ndarray,
        class_idx: int = 1,
        save_html_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Generate interactive SHAP force plot and optionally save as HTML.
        """
        X_ready = self.prepare_data(X)
        class_values = self.get_class_shap_values(shap_values, class_idx=class_idx)

        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            base_value = expected_val[class_idx]
        else:
            base_value = expected_val

        force_plot = shap.force_plot(
            base_value=base_value,
            shap_values=class_values[patient_idx],
            features=X_ready[patient_idx],
            feature_names=self.feature_names,
        )

        if save_html_path:
            Path(save_html_path).parent.mkdir(parents=True, exist_ok=True)
            shap.save_html(str(save_html_path), force_plot)

        return force_plot

    def plot_dependence(
        self,
        feature_name: str,
        shap_values: np.ndarray,
        X: np.ndarray,
        class_idx: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Plot SHAP dependence plot for a chosen feature with auto-interaction coloring.
        """
        X_ready = self.prepare_data(X)
        class_values = self.get_class_shap_values(shap_values, class_idx=class_idx)

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feature_name,
            class_values,
            X_ready,
            feature_names=self.feature_names,
            ax=ax,
            show=False,
        )
        ax.set_title(
            f"SHAP Dependence Plot: {feature_name}\n(Pima Indians Diabetes Dataset)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return fig
