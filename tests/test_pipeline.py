"""
Automated unit tests for data ingestion, leak-free pipeline, and explainability engines.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data import (
    ALL_COLUMNS,
    FEATURE_NAMES,
    PHYSIOLOGICAL_ZERO_COLS,
    clean_missing_values,
    get_train_test_split,
    load_raw_data,
)
from src.model import (
    build_pipeline,
    evaluate_pipeline,
    load_pipeline,
    save_pipeline,
    train_pipeline,
)
from src.explainers.patient_explainer import PatientExplainer
from src.explainers.shap_engine import TreeSHAPEngine
from src.explainers.lime_engine import LIMETabularEngine
from src.explainers.visualization import find_most_uncertain_patient


# ── Data Ingestion & Preprocessing Tests ────────────────────────────────────────
def test_load_raw_data():
    """Verify that the raw dataset loads with expected dimensions and columns."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 768
    assert list(df.columns) == ALL_COLUMNS


def test_clean_missing_values():
    """Verify that physiologically implausible zeros are replaced with NaN."""
    df = load_raw_data()
    cleaned = clean_missing_values(df)

    # Columns where zero is invalid must now contain NaNs
    for col in PHYSIOLOGICAL_ZERO_COLS:
        assert cleaned[col].isna().sum() > 0, f"Expected NaNs in {col}"

    # Pregnancy and Outcome should preserve valid zero counts
    assert not cleaned["Pregnancies"].isna().any()
    assert not cleaned["Outcome"].isna().any()


def test_get_train_test_split_shapes_and_stratification():
    """Verify train/test split proportions and stratified balance."""
    X_train, X_test, y_train, y_test = get_train_test_split(
        test_size=0.2, random_state=42
    )

    total_samples = len(X_train) + len(X_test)
    assert total_samples == 768
    assert len(X_test) == 154  # 20% of 768 is ~154
    assert len(X_train) == 614
    assert X_train.shape[1] == len(FEATURE_NAMES)
    assert X_test.shape[1] == len(FEATURE_NAMES)

    # Verify stratified outcome proportion is approximately equal
    train_ratio = np.mean(y_train)
    test_ratio = np.mean(y_test)
    assert abs(train_ratio - test_ratio) < 0.05


# ── Leak-Free Model Pipeline Tests ──────────────────────────────────────────────
def test_pipeline_training_and_inference():
    """Test building, training, and predicting with the leak-free pipeline."""
    X_train, X_test, y_train, y_test = get_train_test_split(
        test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(n_estimators=20, random_state=42)
    trained_pipeline = train_pipeline(pipeline, X_train, y_train)

    # Ensure pipeline handles input containing NaNs seamlessly via SimpleImputer
    predictions = trained_pipeline.predict(X_test)
    probas = trained_pipeline.predict_proba(X_test)

    assert predictions.shape == (len(X_test),)
    assert probas.shape == (len(X_test), 2)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert np.all((probas >= 0.0) & (probas <= 1.0))


def test_evaluate_pipeline_metrics():
    """Test computation of evaluation metrics."""
    X_train, X_test, y_train, y_test = get_train_test_split(
        test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(n_estimators=20, random_state=42)
    pipeline.fit(X_train, y_train)

    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "classification_report" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_pipeline_serialization(tmp_path: Path):
    """Test saving and loading the trained pipeline artifact with joblib."""
    X_train, X_test, y_train, y_test = get_train_test_split(
        test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(n_estimators=10, random_state=42)
    pipeline.fit(X_train, y_train)

    original_preds = pipeline.predict_proba(X_test)

    artifact_file = tmp_path / "model_pipeline.joblib"
    save_pipeline(pipeline, artifact_file)
    assert artifact_file.exists()

    loaded_pipeline = load_pipeline(artifact_file)
    loaded_preds = loaded_pipeline.predict_proba(X_test)

    np.testing.assert_array_almost_equal(original_preds, loaded_preds)


# ── Explainability Engine Tests ────────────────────────────────────────────────
@pytest.fixture
def trained_components():
    """Fixture providing a fitted pipeline and split datasets."""
    X_train, X_test, y_train, y_test = get_train_test_split(
        test_size=0.2, random_state=42
    )
    pipeline = build_pipeline(n_estimators=15, random_state=42)
    pipeline.fit(X_train, y_train)

    imputer = pipeline.named_steps["imputer"]
    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_imp": X_train_imp,
        "X_test_imp": X_test_imp,
    }


def test_tree_shap_engine(trained_components):
    """Test TreeSHAPEngine value computation and data preparation."""
    pipeline = trained_components["pipeline"]
    X_test = trained_components["X_test"]

    shap_engine = TreeSHAPEngine(pipeline, feature_names=FEATURE_NAMES)
    shap_values = shap_engine.compute_shap_values(X_test[:5])

    class1_values = shap_engine.get_class_shap_values(shap_values, class_idx=1)
    assert class1_values.shape == (5, len(FEATURE_NAMES))


def test_lime_tabular_engine(trained_components):
    """Test LIMETabularEngine local explanation generation."""
    pipeline = trained_components["pipeline"]
    X_train_imp = trained_components["X_train_imp"]
    X_test_imp = trained_components["X_test_imp"]

    lime_engine = LIMETabularEngine(
        training_data=X_train_imp,
        feature_names=FEATURE_NAMES,
        random_state=42,
    )

    explanation = lime_engine.explain_instance(
        data_row=X_test_imp[0],
        predict_fn=pipeline.predict_proba,
        num_features=5,
        num_samples=100,
    )

    items = explanation.as_list()
    assert len(items) == 5
    table = lime_engine.get_explanation_table(explanation)
    assert isinstance(table, pd.DataFrame)
    assert "Feature Condition" in table.columns
    assert "Weight" in table.columns


def test_find_most_uncertain_patient(trained_components):
    """Test detection of samples near the 50% probability boundary."""
    pipeline = trained_components["pipeline"]
    X_test = trained_components["X_test"]
    y_test = trained_components["y_test"]

    uncertain_sample = find_most_uncertain_patient(
        pipeline, X_test, y_test=y_test, target_prob=0.5
    )
    assert isinstance(uncertain_sample, dict)
    assert "index" in uncertain_sample
    assert "prob_diabetes" in uncertain_sample
    assert "uncertainty" in uncertain_sample
    assert 0 <= uncertain_sample["index"] < len(X_test)
    assert 0.0 <= uncertain_sample["prob_diabetes"] <= 1.0


def test_patient_explainer_unified_report(trained_components):
    """Test unified multi-method patient explanation generation."""
    pipeline = trained_components["pipeline"]
    X_train_imp = trained_components["X_train_imp"]
    X_test = trained_components["X_test"]

    explainer = PatientExplainer(
        pipeline_or_model=pipeline,
        training_data=X_train_imp,
        feature_names=FEATURE_NAMES,
        random_state=42,
    )

    patient_sample = X_test[0]
    report = explainer.explain_patient(patient_sample, num_features=8, num_samples=100)

    assert "predicted_class" in report
    assert "prob_diabetic" in report
    assert "shap_attributions" in report
    assert "lime_attributions" in report
    assert "top_shap_feature" in report
    assert "top_lime_condition" in report
    assert "top_feature_agreement" in report
    assert isinstance(report["top_feature_agreement"], bool)
    assert len(report["shap_attributions"]) == len(FEATURE_NAMES)
