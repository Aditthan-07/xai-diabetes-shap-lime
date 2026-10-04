"""
LIME Explainer Engine encapsulating LimeTabularExplainer and safe discretizer patching.
"""

from typing import Any, Callable, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats
import lime
import lime.discretize
import lime.lime_tabular


def patch_lime_discretizer() -> None:
    """
    Patch LIME's BaseDiscretizer to handle NaN scales in truncnorm.rvs gracefully.
    This resolves a known compatibility edge-case across specific scipy/lime versions.
    """

    def _safe_undiscretize(self, feature, values):
        mins = np.array(self.mins[feature])[values]
        maxs = np.array(self.maxs[feature])[values]
        means = np.array(self.means[feature])[values]
        stds = np.maximum(np.array(self.stds[feature])[values], 1e-10)
        minz = (mins - means) / stds
        maxz = (maxs - means) / stds
        min_max_unequal = minz != maxz
        ret = minz.copy().astype(float)
        if np.any(min_max_unequal):
            idx = np.where(min_max_unequal)
            a, b = minz[idx], maxz[idx]
            loc_v, sc = means[idx], stds[idx]
            try:
                ret[idx] = scipy.stats.truncnorm.rvs(
                    a, b, loc=loc_v, scale=sc, random_state=self.random_state
                )
            except Exception:
                ret[idx] = loc_v  # fallback to bin mean
        return ret

    lime.discretize.BaseDiscretizer.get_undiscretize_values = _safe_undiscretize


# Auto-apply patch upon module import
patch_lime_discretizer()


class LIMETabularEngine:
    """
    Encapsulates LimeTabularExplainer for tabular local explanations.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "classification",
        random_state: int = 42,
    ) -> None:
        """
        Initialize LIMETabularEngine.

        Parameters
        ----------
        training_data : np.ndarray
            Training set used by the model (must be numeric and imputed).
        feature_names : Optional[List[str]]
            List of feature names.
        class_names : Optional[List[str]]
            Class label names (defaults to ['Non-Diabetic', 'Diabetic']).
        mode : str
            Explanation mode, 'classification' or 'regression'.
        random_state : int
            Reproducibility seed.
        """
        patch_lime_discretizer()
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
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            random_state=random_state,
        )

    def explain_instance(
        self,
        data_row: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_features: int = 8,
        num_samples: int = 1000,
    ) -> lime.explanation.Explanation:
        """
        Generate a local explanation for a single instance.

        Parameters
        ----------
        data_row : np.ndarray
            1D array of patient feature values.
        predict_fn : Callable
            Probability prediction function, e.g., model.predict_proba or pipeline.predict_proba.
        num_features : int
            Number of top contributing features to include.
        num_samples : int
            Number of perturbed samples generated for local surrogate fitting.

        Returns
        -------
        lime.explanation.Explanation
        """
        return self.explainer.explain_instance(
            data_row=data_row,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

    @staticmethod
    def get_explanation_table(
        explanation: lime.explanation.Explanation,
    ) -> pd.DataFrame:
        """
        Convert LIME explanation into a structured pandas DataFrame.
        """
        items = explanation.as_list()
        df = pd.DataFrame(items, columns=["Feature Condition", "Weight"])
        df["Direction"] = df["Weight"].apply(
            lambda w: "🔴 Risk ↑" if w > 0 else "🟢 Risk ↓"
        )
        df["Weight"] = df["Weight"].round(4)
        return df

    def plot_explanation_bar(
        self,
        explanation: lime.explanation.Explanation,
        patient_idx: Optional[int] = None,
        num_features: int = 8,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> plt.Figure:
        """
        Render a custom styled horizontal bar chart of LIME weights.
        """
        items = explanation.as_list()
        features = [x[0] for x in items]
        weights = [x[1] for x in items]
        colors = ["#E24B4A" if w > 0 else "#1D9E75" for w in weights]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(features))
        ax.barh(y_pos, weights, color=colors, edgecolor="white", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME weight (local linear model coefficient)", fontsize=11)

        title_suffix = f" — Patient #{patient_idx}" if patient_idx is not None else ""
        ax.set_title(
            f"LIME Explanation{title_suffix}\nTop {num_features} features",
            fontsize=12,
            fontweight="bold",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        red_patch = mpatches.Patch(
            color="#E24B4A", label="Supports diabetic prediction"
        )
        green_patch = mpatches.Patch(
            color="#1D9E75", label="Contradicts diabetic prediction"
        )
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
