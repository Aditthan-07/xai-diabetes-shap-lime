"""
Visualization and comparative diagnostic tools for SHAP and LIME.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_most_uncertain_patient(
    model_or_predict_fn: Any,
    X_test: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    target_prob: float = 0.5,
    top_k: int = 1,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Identify patient samples whose predicted probability is closest to the decision threshold.

    Parameters
    ----------
    model_or_predict_fn : Any
        An estimator with a .predict_proba() method, or a callable predict_proba function.
    X_test : np.ndarray
        Test feature matrix.
    y_test : Optional[np.ndarray]
        Ground-truth labels if available.
    target_prob : float
        Decision boundary threshold (default 0.5).
    top_k : int
        Number of most uncertain samples to return.

    Returns
    -------
    Union[Dict[str, Any], List[Dict[str, Any]]]
        Dictionary (if top_k=1) or list of dictionaries containing:
        'index', 'prob_diabetes', 'uncertainty', 'true_label', 'features'.
    """
    if hasattr(model_or_predict_fn, "predict_proba"):
        probs = model_or_predict_fn.predict_proba(X_test)[:, 1]
    elif callable(model_or_predict_fn):
        probs = model_or_predict_fn(X_test)[:, 1]
    else:
        raise ValueError("Provided object must have .predict_proba or be callable.")

    uncertainty = np.abs(probs - target_prob)
    sorted_indices = np.argsort(uncertainty)

    results = []
    for idx in sorted_indices[:top_k]:
        patient_info = {
            "index": int(idx),
            "prob_diabetes": float(probs[idx]),
            "uncertainty": float(uncertainty[idx]),
            "true_label": int(y_test[idx]) if y_test is not None else None,
            "features": X_test[idx],
        }
        results.append(patient_info)

    if top_k == 1:
        return results[0]
    return results


def plot_shap_vs_lime(
    patient_idx: int,
    shap_patient_values: np.ndarray,
    lime_explanation: Any,
    feature_names: List[str],
    top_n: int = 8,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Render a side-by-side comparative visualization of SHAP values and LIME weights.

    Parameters
    ----------
    patient_idx : int
        Index of the patient being diagnosed.
    shap_patient_values : np.ndarray
        1D array of SHAP values for this specific patient (for Diabetic class).
    lime_explanation : Any
        Fitted lime.explanation.Explanation instance.
    feature_names : List[str]
        List of feature column names.
    top_n : int
        Number of features to display in the comparison.
    save_path : Optional[Union[str, Path]]
        Path to save figure.
    show : bool
        Whether to show the plot.

    Returns
    -------
    Tuple[plt.Figure, pd.DataFrame]
        The generated figure and a comparison dataframe.
    """
    shap_df = pd.DataFrame({"Feature": feature_names, "SHAP": shap_patient_values})
    shap_df["abs_SHAP"] = shap_df["SHAP"].abs()
    shap_sorted = shap_df.sort_values("abs_SHAP", ascending=False).head(top_n)
    shap_sorted = shap_sorted.sort_values("SHAP", ascending=True)

    lime_items = lime_explanation.as_list()
    lime_feats = [x[0] for x in lime_items][:top_n]
    lime_vals = [x[1] for x in lime_items][:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"SHAP vs LIME Comparative Attribution — Patient #{patient_idx}",
        fontsize=13,
        fontweight="bold",
    )

    # Subplot 1: SHAP
    colors_s = ["#E24B4A" if v > 0 else "#1D9E75" for v in shap_sorted["SHAP"]]
    ax1.barh(
        shap_sorted["Feature"],
        shap_sorted["SHAP"],
        color=colors_s,
        edgecolor="white",
        height=0.6,
    )
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_title("SHAP Values\n(game-theory, consistent)", fontsize=11)
    ax1.set_xlabel("SHAP value (additive impact)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Subplot 2: LIME
    colors_l = ["#E24B4A" if v > 0 else "#1D9E75" for v in lime_vals]
    ax2.barh(lime_feats, lime_vals, color=colors_l, edgecolor="white", height=0.6)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("LIME Weights\n(local surrogate, approximate)", fontsize=11)
    ax2.set_xlabel("LIME weight (local surrogate coefficient)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, shap_df
