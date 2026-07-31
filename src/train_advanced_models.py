"""
train_advanced_models.py

Week 7 - Advanced Gradient-Boosting Models

This script:

1. Loads the Week 6 feature-engineered datasets
2. Cleans and validates model inputs
3. Creates a chronological training and validation split
4. Trains a reference scikit-learn model
5. Trains XGBoost and LightGBM regressors
6. Uses early stopping for the advanced models
7. Evaluates all models on the untouched time-based test set
8. Saves metrics, predictions, feature importance, and trained models
9. Selects and saves the strongest advanced model

Required input files:
    data/train_features.csv
    data/test_features.csv

Outputs:
    models/reference_model_week7.pkl
    models/xgboost_model.pkl
    models/lightgbm_model.pkl
    models/best_advanced_model.pkl

    reports/advanced_model_metrics.csv
    reports/advanced_model_predictions.csv
    reports/advanced_feature_importance.csv
    reports/advanced_model_comparison.png
    reports/advanced_feature_importance.png
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightgbm import (
    LGBMRegressor,
    early_stopping as lgb_early_stopping,
    log_evaluation as lgb_log_evaluation,
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# --------------------------------------------------
# Project Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train_features.csv"
TEST_FILE = DATA_DIR / "test_features.csv"

TARGET_COLUMN = "ClosePrice"

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20

REFERENCE_MODEL_FILE = MODELS_DIR / "reference_model_week7.pkl"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_model.pkl"
LIGHTGBM_MODEL_FILE = MODELS_DIR / "lightgbm_model.pkl"
BEST_MODEL_FILE = MODELS_DIR / "best_advanced_model.pkl"

METRICS_FILE = REPORTS_DIR / "advanced_model_metrics.csv"
PREDICTIONS_FILE = REPORTS_DIR / "advanced_model_predictions.csv"
IMPORTANCE_FILE = REPORTS_DIR / "advanced_feature_importance.csv"
MODEL_METADATA_FILE = REPORTS_DIR / "best_advanced_model_metadata.json"

COMPARISON_PLOT_FILE = (
    REPORTS_DIR / "advanced_model_comparison.png"
)

IMPORTANCE_PLOT_FILE = (
    REPORTS_DIR / "advanced_feature_importance.png"
)


# --------------------------------------------------
# Folder Creation
# --------------------------------------------------

def create_output_directories() -> None:
    """
    Create output directories when they do not exist.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# --------------------------------------------------
# Data Loading
# --------------------------------------------------

def load_feature_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Week 6 feature-engineered train and test datasets.

    Returns:
        Tuple containing training and testing DataFrames.
    """

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training feature file not found: {TRAIN_FILE}\n"
            "Run src/features.py before running Week 7."
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing feature file not found: {TEST_FILE}\n"
            "Run src/features.py before running Week 7."
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("Feature-engineered datasets loaded successfully.")
    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")

    return train_df, test_df


# --------------------------------------------------
# Dataset Validation
# --------------------------------------------------

def validate_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Validate basic dataset structure.

    Args:
        train_df: Training dataset.
        test_df: Testing dataset.
    """

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} is missing from training data."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} is missing from testing data."
        )

    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if test_df.empty:
        raise ValueError("Testing dataset is empty.")

    if train_df.columns.duplicated().any():
        duplicated = train_df.columns[
            train_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Training data contains duplicate columns: {duplicated}"
        )

    if test_df.columns.duplicated().any():
        duplicated = test_df.columns[
            test_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Testing data contains duplicate columns: {duplicated}"
        )


# --------------------------------------------------
# Model Matrix Preparation
# --------------------------------------------------

def convert_features_to_numeric(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert predictor columns to numeric values.

    Week 6 should already produce numeric model matrices. This function
    provides an additional safeguard for CSV dtype inconsistencies.

    Non-convertible values become NaN and are filled using training
    medians.

    Args:
        train_features: Training predictors.
        test_features: Testing predictors.

    Returns:
        Numeric and aligned training and testing predictors.
    """

    train_numeric = train_features.copy()
    test_numeric = test_features.copy()

    train_numeric, test_numeric = train_numeric.align(
        test_numeric,
        join="left",
        axis=1,
        fill_value=0,
    )

    for column in train_numeric.columns:
        train_numeric[column] = pd.to_numeric(
            train_numeric[column],
            errors="coerce",
        )

        test_numeric[column] = pd.to_numeric(
            test_numeric[column],
            errors="coerce",
        )

        training_median = train_numeric[column].median()

        if pd.isna(training_median):
            training_median = 0.0

        train_numeric[column] = (
            train_numeric[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(training_median)
            .astype(float)
        )

        test_numeric[column] = (
            test_numeric[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(training_median)
            .astype(float)
        )

    return train_numeric, test_numeric


def prepare_model_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Separate predictors and target and create numeric model matrices.

    Args:
        train_df: Training dataset.
        test_df: Testing dataset.

    Returns:
        X_train, y_train, X_test, y_test.
    """

    X_train_raw = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    X_test_raw = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = pd.to_numeric(
        train_df[TARGET_COLUMN],
        errors="coerce",
    )

    y_test = pd.to_numeric(
        test_df[TARGET_COLUMN],
        errors="coerce",
    )

    if y_train.isna().any():
        missing_count = int(y_train.isna().sum())

        raise ValueError(
            f"Training target contains {missing_count} missing values."
        )

    if y_test.isna().any():
        missing_count = int(y_test.isna().sum())

        raise ValueError(
            f"Testing target contains {missing_count} missing values."
        )

    X_train, X_test = convert_features_to_numeric(
        train_features=X_train_raw,
        test_features=X_test_raw,
    )

    if np.isinf(
        X_train.to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "Training predictors contain infinite values."
        )

    if np.isinf(
        X_test.to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "Testing predictors contain infinite values."
        )

    print("\nModel matrices prepared.")
    print(f"Training predictors: {X_train.shape}")
    print(f"Testing predictors: {X_test.shape}")
    print(f"Training target: {y_train.shape}")
    print(f"Testing target: {y_test.shape}")

    return X_train, y_train, X_test, y_test


# --------------------------------------------------
# Chronological Validation Split
# --------------------------------------------------

def create_validation_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Create a chronological validation split.

    The last portion of the training dataset becomes the validation set.
    No random shuffling is used.

    Args:
        X_train: Full training predictors.
        y_train: Full training target.
        validation_fraction: Fraction reserved for validation.

    Returns:
        X_fit, X_validation, y_fit, y_validation.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    split_index = int(
        len(X_train) * (1 - validation_fraction)
    )

    if split_index <= 0 or split_index >= len(X_train):
        raise ValueError(
            "The validation split produced an empty subset."
        )

    X_fit = X_train.iloc[:split_index].copy()
    X_validation = X_train.iloc[split_index:].copy()

    y_fit = y_train.iloc[:split_index].copy()
    y_validation = y_train.iloc[split_index:].copy()

    print("\nChronological validation split created.")
    print(f"Model fitting observations: {len(X_fit):,}")
    print(
        f"Validation observations: "
        f"{len(X_validation):,}"
    )

    return (
        X_fit,
        X_validation,
        y_fit,
        y_validation,
    )


# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

def calculate_mape(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Observations with an actual target of zero are excluded.
    """

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    valid_mask = (
        np.isfinite(actual_array)
        & np.isfinite(predicted_array)
        & (actual_array != 0)
    )

    if not np.any(valid_mask):
        return float("nan")

    percentage_errors = np.abs(
        (
            actual_array[valid_mask]
            - predicted_array[valid_mask]
        )
        / actual_array[valid_mask]
    )

    return float(
        np.mean(percentage_errors) * 100
    )


def calculate_mdape(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Median Absolute Percentage Error.
    """

    actual_array = np.asarray(
        actual,
        dtype=float,
    )

    predicted_array = np.asarray(
        predicted,
        dtype=float,
    )

    valid_mask = (
        np.isfinite(actual_array)
        & np.isfinite(predicted_array)
        & (actual_array != 0)
    )

    if not np.any(valid_mask):
        return float("nan")

    percentage_errors = np.abs(
        (
            actual_array[valid_mask]
            - predicted_array[valid_mask]
        )
        / actual_array[valid_mask]
    )

    return float(
        np.median(percentage_errors) * 100
    )


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """
    Calculate all regression metrics.

    Returns:
        Dictionary containing MAE, RMSE, MAPE, MdAPE, and R².
    """

    return {
        "MAE": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "MAPE": calculate_mape(
            actual,
            predicted,
        ),
        "MdAPE": calculate_mdape(
            actual,
            predicted,
        ),
        "R2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
    }


# --------------------------------------------------
# Model Definitions
# --------------------------------------------------

def create_reference_model() -> GradientBoostingRegressor:
    """
    Create a Week 6-style reference model.
    """

    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        loss="squared_error",
        random_state=RANDOM_STATE,
    )


def create_xgboost_model() -> XGBRegressor:
    """
    Create an XGBoost regression model.
    """

    return XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        early_stopping_rounds=75,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_lightgbm_model() -> LGBMRegressor:
    """
    Create a LightGBM regression model.
    """

    return LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


# --------------------------------------------------
# Model Training
# --------------------------------------------------

def train_reference_model(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
) -> tuple[GradientBoostingRegressor, float]:
    """
    Train the reference model.

    Returns:
        Trained model and training duration in seconds.
    """

    model = create_reference_model()

    start_time = perf_counter()

    model.fit(
        X_fit,
        y_fit,
    )

    training_seconds = (
        perf_counter() - start_time
    )

    return model, training_seconds


def train_xgboost_model(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[XGBRegressor, float]:
    """
    Train XGBoost with early stopping.

    Returns:
        Trained model and training duration in seconds.
    """

    model = create_xgboost_model()

    start_time = perf_counter()

    model.fit(
        X_fit,
        y_fit,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    training_seconds = (
        perf_counter() - start_time
    )

    return model, training_seconds


def train_lightgbm_model(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[LGBMRegressor, float]:
    """
    Train LightGBM with early stopping.

    Returns:
        Trained model and training duration in seconds.
    """

    model = create_lightgbm_model()

    callbacks = [
        lgb_early_stopping(
            stopping_rounds=75,
            first_metric_only=True,
            verbose=False,
        ),
        lgb_log_evaluation(
            period=0,
        ),
    ]

    start_time = perf_counter()

    model.fit(
        X_fit,
        y_fit,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    training_seconds = (
        perf_counter() - start_time
    )

    return model, training_seconds


# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------

def evaluate_trained_model(
    model_name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    training_seconds: float,
) -> tuple[dict[str, float | str], np.ndarray]:
    """
    Evaluate a trained model on the untouched test dataset.

    Returns:
        Metrics row and test predictions.
    """

    prediction_start = perf_counter()

    predictions = model.predict(X_test)

    prediction_seconds = (
        perf_counter() - prediction_start
    )

    metrics = calculate_metrics(
        actual=y_test,
        predicted=predictions,
    )

    metrics_row = {
        "Model": model_name,
        **metrics,
        "TrainingSeconds": float(training_seconds),
        "PredictionSeconds": float(prediction_seconds),
    }

    return metrics_row, predictions


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

def extract_feature_importance(
    model_name: str,
    model: object,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Extract model feature importance values.

    Args:
        model_name: Display name of model.
        model: Trained model.
        feature_names: Predictor column names.

    Returns:
        Feature-importance DataFrame.
    """

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return pd.DataFrame(
            columns=[
                "Model",
                "Feature",
                "Importance",
                "Rank",
            ]
        )

    importance_values = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    importance_df = pd.DataFrame(
        {
            "Model": model_name,
            "Feature": feature_names,
            "Importance": importance_values,
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df["Rank"] = (
        importance_df.index + 1
    )

    return importance_df


# --------------------------------------------------
# Model Saving
# --------------------------------------------------

def save_models(
    reference_model: object,
    xgboost_model: object,
    lightgbm_model: object,
) -> None:
    """
    Save all trained models.
    """

    joblib.dump(
        reference_model,
        REFERENCE_MODEL_FILE,
    )

    joblib.dump(
        xgboost_model,
        XGBOOST_MODEL_FILE,
    )

    joblib.dump(
        lightgbm_model,
        LIGHTGBM_MODEL_FILE,
    )

    print("\nIndividual models saved:")
    print(f"- {REFERENCE_MODEL_FILE}")
    print(f"- {XGBOOST_MODEL_FILE}")
    print(f"- {LIGHTGBM_MODEL_FILE}")


def save_best_model(
    metrics_df: pd.DataFrame,
    trained_models: dict[str, object],
    feature_names: list[str],
) -> str:
    """
    Select the best model using the lowest RMSE.

    R² is used as a secondary sorting criterion.

    Returns:
        Name of the selected model.
    """

    ranked_metrics = metrics_df.sort_values(
        by=["RMSE", "R2"],
        ascending=[True, False],
    ).reset_index(drop=True)

    best_model_name = str(
        ranked_metrics.loc[0, "Model"]
    )

    best_model = trained_models[
        best_model_name
    ]

    artifact = {
        "model_name": best_model_name,
        "model": best_model,
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN,
    }

    joblib.dump(
        artifact,
        BEST_MODEL_FILE,
    )

    metadata = {
        "model_name": best_model_name,
        "target_column": TARGET_COLUMN,
        "feature_count": len(feature_names),
        "selection_metric": "RMSE",
        "test_metrics": {
            key: float(ranked_metrics.loc[0, key])
            for key in [
                "MAE",
                "RMSE",
                "MAPE",
                "MdAPE",
                "R2",
            ]
        },
    }

    with open(
        MODEL_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    print(f"\nBest model: {best_model_name}")
    print(f"Best model artifact saved to: {BEST_MODEL_FILE}")
    print(f"Metadata saved to: {MODEL_METADATA_FILE}")

    return best_model_name


# --------------------------------------------------
# Report Saving
# --------------------------------------------------

def save_metrics_report(
    metrics_df: pd.DataFrame,
) -> None:
    """
    Save model evaluation metrics.
    """

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    print(f"\nMetrics saved to: {METRICS_FILE}")


def save_predictions_report(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
) -> None:
    """
    Save actual values and predictions from every model.
    """

    prediction_df = pd.DataFrame(
        {
            "ActualClosePrice": (
                y_test.reset_index(drop=True)
            ),
        }
    )

    for model_name, model_predictions in predictions.items():
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        prediction_column = (
            f"{safe_name}_prediction"
        )

        error_column = (
            f"{safe_name}_absolute_error"
        )

        prediction_df[prediction_column] = (
            model_predictions
        )

        prediction_df[error_column] = np.abs(
            prediction_df["ActualClosePrice"]
            - prediction_df[prediction_column]
        )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_FILE}"
    )


def save_importance_report(
    importance_df: pd.DataFrame,
) -> None:
    """
    Save feature importance values.
    """

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    print(
        f"Feature importance saved to: "
        f"{IMPORTANCE_FILE}"
    )


# --------------------------------------------------
# Visualizations
# --------------------------------------------------

def save_model_comparison_plot(
    metrics_df: pd.DataFrame,
) -> None:
    """
    Save an RMSE comparison chart.
    """

    plot_df = metrics_df.sort_values(
        by="RMSE",
        ascending=True,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        plot_df["Model"],
        plot_df["RMSE"],
    )

    plt.xlabel("Model")
    plt.ylabel("Root Mean Squared Error ($)")
    plt.title(
        "Week 7 Advanced Model Comparison"
    )

    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Model comparison plot saved to: "
        f"{COMPARISON_PLOT_FILE}"
    )


def save_feature_importance_plot(
    importance_df: pd.DataFrame,
    best_model_name: str,
    top_n: int = 20,
) -> None:
    """
    Save a feature importance chart for the selected model.
    """

    model_importance = importance_df[
        importance_df["Model"]
        == best_model_name
    ].copy()

    if model_importance.empty:
        print(
            "No feature importance was available "
            "for the selected model."
        )

        return

    top_features = (
        model_importance
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            by="Importance",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Top {top_n} Features: "
        f"{best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        IMPORTANCE_PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Feature importance plot saved to: "
        f"{IMPORTANCE_PLOT_FILE}"
    )


# --------------------------------------------------
# Console Summary
# --------------------------------------------------

def print_metrics_summary(
    metrics_df: pd.DataFrame,
) -> None:
    """
    Print a formatted model-comparison summary.
    """

    display_columns = [
        "Model",
        "MAE",
        "RMSE",
        "MAPE",
        "MdAPE",
        "R2",
        "TrainingSeconds",
    ]

    print("\nModel comparison")
    print("-" * 100)

    print(
        metrics_df[
            display_columns
        ]
        .sort_values(
            by="RMSE",
            ascending=True,
        )
        .to_string(
            index=False,
            formatters={
                "MAE": lambda value: (
                    f"${value:,.2f}"
                ),
                "RMSE": lambda value: (
                    f"${value:,.2f}"
                ),
                "MAPE": lambda value: (
                    f"{value:.2f}%"
                ),
                "MdAPE": lambda value: (
                    f"{value:.2f}%"
                ),
                "R2": lambda value: (
                    f"{value:.4f}"
                ),
                "TrainingSeconds": lambda value: (
                    f"{value:.2f}"
                ),
            },
        )
    )


# --------------------------------------------------
# Complete Pipeline
# --------------------------------------------------

def main() -> None:
    """
    Run the complete Week 7 advanced-model pipeline.
    """

    print("=" * 70)
    print("California Property Price Prediction")
    print("Week 7: Advanced Gradient-Boosting Models")
    print("=" * 70)

    create_output_directories()

    train_df, test_df = load_feature_data()

    validate_datasets(
        train_df=train_df,
        test_df=test_df,
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
    ) = prepare_model_data(
        train_df=train_df,
        test_df=test_df,
    )

    (
        X_fit,
        X_validation,
        y_fit,
        y_validation,
    ) = create_validation_split(
        X_train=X_train,
        y_train=y_train,
    )

    print("\nTraining reference model...")

    (
        reference_model,
        reference_training_seconds,
    ) = train_reference_model(
        X_fit=X_fit,
        y_fit=y_fit,
    )

    print("Training XGBoost...")

    (
        xgboost_model,
        xgboost_training_seconds,
    ) = train_xgboost_model(
        X_fit=X_fit,
        y_fit=y_fit,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    print("Training LightGBM...")

    (
        lightgbm_model,
        lightgbm_training_seconds,
    ) = train_lightgbm_model(
        X_fit=X_fit,
        y_fit=y_fit,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    trained_models = {
        "Gradient Boosting Reference": reference_model,
        "XGBoost": xgboost_model,
        "LightGBM": lightgbm_model,
    }

    training_times = {
        "Gradient Boosting Reference": (
            reference_training_seconds
        ),
        "XGBoost": xgboost_training_seconds,
        "LightGBM": lightgbm_training_seconds,
    }

    metrics_rows = []
    prediction_results = {}
    importance_frames = []

    for model_name, model in trained_models.items():
        metrics_row, model_predictions = (
            evaluate_trained_model(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
                training_seconds=(
                    training_times[model_name]
                ),
            )
        )

        metrics_rows.append(
            metrics_row
        )

        prediction_results[
            model_name
        ] = model_predictions

        model_importance = (
            extract_feature_importance(
                model_name=model_name,
                model=model,
                feature_names=(
                    X_train.columns.tolist()
                ),
            )
        )

        importance_frames.append(
            model_importance
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    ).sort_values(
        by=["RMSE", "R2"],
        ascending=[True, False],
    ).reset_index(drop=True)

    importance_df = pd.concat(
        importance_frames,
        ignore_index=True,
    )

    print_metrics_summary(
        metrics_df
    )

    save_models(
        reference_model=reference_model,
        xgboost_model=xgboost_model,
        lightgbm_model=lightgbm_model,
    )

    best_model_name = save_best_model(
        metrics_df=metrics_df,
        trained_models=trained_models,
        feature_names=X_train.columns.tolist(),
    )

    save_metrics_report(
        metrics_df=metrics_df,
    )

    save_predictions_report(
        y_test=y_test,
        predictions=prediction_results,
    )

    save_importance_report(
        importance_df=importance_df,
    )

    save_model_comparison_plot(
        metrics_df=metrics_df,
    )

    save_feature_importance_plot(
        importance_df=importance_df,
        best_model_name=best_model_name,
    )

    print(
        "\nWeek 7 advanced-model training "
        "completed successfully."
    )


if __name__ == "__main__":
    main()