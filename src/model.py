"""
Model training, evaluation, and pipeline serialization module.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def build_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    **rf_kwargs: Any,
) -> Pipeline:
    """
    Construct a leak-free scikit-learn Pipeline with median imputation
    and Random Forest classification.

    Parameters
    ----------
    n_estimators : int
        Number of decision trees in the forest. Default 100.
    max_depth : Optional[int]
        Maximum tree depth. Default None.
    random_state : int
        Random seed for reproducibility. Default 42.
    rf_kwargs : Any
        Additional keyword arguments for RandomForestClassifier.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn pipeline instance.
    """
    imputer = SimpleImputer(strategy="median")
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **rf_kwargs,
    )
    return Pipeline(steps=[("imputer", imputer), ("classifier", classifier)])


def train_pipeline(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Pipeline:
    """
    Fit the pipeline strictly on training data.

    Parameters
    ----------
    pipeline : Pipeline
        The scikit-learn pipeline to train.
    X_train : np.ndarray
        Training feature matrix (may contain NaNs).
    y_train : np.ndarray
        Training target vector.

    Returns
    -------
    Pipeline
        Fitted scikit-learn pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluate the fitted pipeline on test data and compute key performance metrics.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted scikit-learn pipeline.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True target vector.

    Returns
    -------
    Dict[str, Any]
        Dictionary with accuracy, roc_auc, report_dict, and report_text.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_proba))
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Non-Diabetic", "Diabetic"],
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Non-Diabetic", "Diabetic"],
    )

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def save_pipeline(
    pipeline: Pipeline,
    file_path: Union[str, Path],
) -> Path:
    """
    Serialize and save a fitted pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline instance.
    file_path : Union[str, Path]
        Target destination path (e.g., 'models/diabetes_rf_pipeline.joblib').

    Returns
    -------
    Path
        Resolved saved path.
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target)
    return target


def load_pipeline(file_path: Union[str, Path]) -> Pipeline:
    """
    Load a serialized pipeline from disk.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the saved pipeline joblib file.

    Returns
    -------
    Pipeline
        Loaded scikit-learn pipeline.
    """
    target = Path(file_path)
    if not target.exists():
        raise FileNotFoundError(f"Pipeline artifact not found at {target}")
    return joblib.load(target)


def get_pipeline_components(pipeline: Pipeline):
    """
    Helper to extract fitted imputer and classifier steps from the pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted pipeline.

    Returns
    -------
    Tuple[SimpleImputer, RandomForestClassifier]
    """
    return pipeline.named_steps["imputer"], pipeline.named_steps["classifier"]
