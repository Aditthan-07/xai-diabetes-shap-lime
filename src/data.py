"""
Data ingestion and preprocessing module for the Pima Indians Diabetes Dataset.
"""

from typing import List, Optional, Tuple, Union
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_NAMES: List[str] = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
]

ALL_COLUMNS: List[str] = FEATURE_NAMES + ['Outcome']

PHYSIOLOGICAL_ZERO_COLS: List[str] = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
]

REMOTE_DATASET_URL: str = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

DEFAULT_LOCAL_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "pima-indians-diabetes.data.csv"
)


def load_raw_data(
    file_path: Optional[Union[str, Path]] = None,
    remote_url: str = REMOTE_DATASET_URL,
) -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes dataset from a local path with remote fallback.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]]
        Path to the local CSV file. Defaults to 'data/pima-indians-diabetes.data.csv'.
    remote_url : str
        URL to fetch the dataset if the local file is unavailable.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with standard column headers.
    """
    target_path = Path(file_path) if file_path else DEFAULT_LOCAL_PATH

    if target_path.exists():
        df = pd.read_csv(target_path, names=ALL_COLUMNS)
    else:
        try:
            df = pd.read_csv(remote_url, names=ALL_COLUMNS)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load dataset locally from '{target_path}' and "
                f"remotely from '{remote_url}': {exc}"
            ) from exc

    return df


def clean_missing_values(
    df: pd.DataFrame,
    zero_replace_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Replace physiologically impossible zeros with NaN in numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw diabetes DataFrame.
    zero_replace_cols : Optional[List[str]]
        Columns where zero indicates a missing observation.

    Returns
    -------
    pd.DataFrame
        DataFrame with zero values replaced by np.nan.
    """
    cols_to_clean = zero_replace_cols or PHYSIOLOGICAL_ZERO_COLS
    df_clean = df.copy()
    for col in cols_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(0, np.nan)
    return df_clean


def get_train_test_split(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    return_df: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
]:
    """
    Ingest, clean, and stratify the diabetes dataset into train and test sets.
    Imputation is intentionally not performed here to avoid data leakage.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Input dataframe. If None, load_raw_data() and clean_missing_values() are called.
    test_size : float
        Proportion of dataset to include in the test split. Default 0.2.
    random_state : int
        Seed for reproducibility. Default 42.
    return_df : bool
        If True, returns pandas DataFrames/Series; if False, returns numpy ndarrays.

    Returns
    -------
    Tuple[X_train, X_test, y_train, y_test]
    """
    if df is None:
        raw_df = load_raw_data()
        cleaned_df = clean_missing_values(raw_df)
    else:
        cleaned_df = clean_missing_values(df)

    X = cleaned_df[FEATURE_NAMES]
    y = cleaned_df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if return_df:
        return X_train, X_test, y_train, y_test

    return X_train.values, X_test.values, y_train.values, y_test.values
