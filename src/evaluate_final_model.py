"""
evaluate_final_model.py

Week 8 - Final Model Evaluation

Leakage-safe workflow:
1. Load feature-engineered training and testing datasets.
2. Use training data only to clean predictor values.
3. Create a chronological validation split from the training data.
4. Train candidate models using the fitting subset.
5. Select the best model using validation RMSE only.
6. Refit the selected model using the complete training dataset.
7. Evaluate the final model once on the untouched test dataset.
8. Calculate overall and price-band metrics.
9. Save the final model, predictions, metrics, and figures.

Inputs:
    data/train_features.csv
    data/test_features.csv

Outputs:
    models/final_evaluated_model.pkl

    reports/metrics_summary.csv
    reports/price_band_metrics.csv
    reports/final_test_predictions.csv
    reports/final_model_metadata.json
    reports/actual_vs_predicted.png
    reports/residual_distribution.png
    reports/price_band_mape.png
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

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

VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42

FINAL_MODEL_FILE = MODELS_DIR / "final_evaluated_model.pkl"

METRICS_FILE = REPORTS_DIR / "metrics_summary.csv"
PRICE_BAND_FILE = REPORTS_DIR / "price_band_metrics.csv"
PREDICTIONS_FILE = REPORTS_DIR / "final_test_predictions.csv"
METADATA_FILE = REPORTS_DIR / "final_model_metadata.json"

ACTUAL_PREDICTED_FIGURE = (
    REPORTS_DIR / "actual_vs_predicted.png"
)

RESIDUAL_FIGURE = (
    REPORTS_DIR / "residual_distribution.png"
)

PRICE_BAND_FIGURE = (
    REPORTS_DIR / "price_band_mape.png"
)


# --------------------------------------------------
# Output Folders
# --------------------------------------------------

def create_output_directories() -> None:
    """Create model and report directories."""

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Week 6 feature-engineered datasets."""

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Run src/features.py before running Week 8."
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_FILE}\n"
            "Run src/features.py before running Week 8."
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("Datasets loaded successfully.")
    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")

    return train_df, test_df


# --------------------------------------------------
# Dataset Validation
# --------------------------------------------------

def validate_dataset_structure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Validate basic dataset structure."""

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
        duplicates = train_df.columns[
            train_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Training data contains duplicate columns: "
            f"{duplicates}"
        )

    if test_df.columns.duplicated().any():
        duplicates = test_df.columns[
            test_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Testing data contains duplicate columns: "
            f"{duplicates}"
        )


# --------------------------------------------------
# Model Matrix Preparation
# --------------------------------------------------

def prepare_model_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    dict[str, float],
]:
    """
    Prepare numeric predictor matrices without test leakage.

    All missing-value replacement values are calculated from the
    training dataset only.

    Returns:
        X_train
        y_train
        X_test
        y_test
        training_medians
    """

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    y_train = pd.to_numeric(
        train_df[TARGET_COLUMN],
        errors="coerce",
    )

    y_test = pd.to_numeric(
        test_df[TARGET_COLUMN],
        errors="coerce",
    )

    if y_train.isna().any():
        raise ValueError(
            "Training target contains missing or invalid values."
        )

    if y_test.isna().any():
        raise ValueError(
            "Testing target contains missing or invalid values."
        )

    # Align test predictors to the training feature definition.
    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )

    training_medians: dict[str, float] = {}

    for column in X_train.columns:
        X_train[column] = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )

        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce",
        )

        X_train[column] = X_train[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X_test[column] = X_test[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # Calculate the value using training data only.
        training_median = X_train[column].median()

        if pd.isna(training_median):
            training_median = 0.0

        training_medians[column] = float(
            training_median
        )

        X_train[column] = X_train[column].fillna(
            training_median
        ).astype(float)

        X_test[column] = X_test[column].fillna(
            training_median
        ).astype(float)

    train_array = X_train.to_numpy(dtype=float)
    test_array = X_test.to_numpy(dtype=float)

    if np.isnan(train_array).any():
        raise ValueError(
            "Training predictors still contain missing values."
        )

    if np.isnan(test_array).any():
        raise ValueError(
            "Testing predictors still contain missing values."
        )

    if np.isinf(train_array).any():
        raise ValueError(
            "Training predictors contain infinite values."
        )

    if np.isinf(test_array).any():
        raise ValueError(
            "Testing predictors contain infinite values."
        )

    print("\nModel matrices prepared.")
    print(f"Training predictors: {X_train.shape}")
    print(f"Testing predictors: {X_test.shape}")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        training_medians,
    )


# --------------------------------------------------
# Chronological Validation Split
# --------------------------------------------------

def create_validation_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Reserve the final portion of the training data for validation.

    No shuffling is used. The test dataset remains untouched.
    """

    split_index = int(
        len(X_train)
        * (1 - VALIDATION_FRACTION)
    )

    if split_index <= 0 or split_index >= len(X_train):
        raise ValueError(
            "Chronological validation split produced "
            "an empty subset."
        )

    X_fit = X_train.iloc[:split_index].copy()
    X_validation = X_train.iloc[split_index:].copy()

    y_fit = y_train.iloc[:split_index].copy()
    y_validation = y_train.iloc[split_index:].copy()

    print("\nChronological validation split")
    print("-" * 45)
    print(f"Fitting observations: {len(X_fit):,}")
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
# Metrics
# --------------------------------------------------

def calculate_percentage_errors(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Calculate valid absolute percentage errors."""

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
        return np.array(
            [],
            dtype=float,
        )

    return np.abs(
        (
            actual_array[valid_mask]
            - predicted_array[valid_mask]
        )
        / actual_array[valid_mask]
    )


def calculate_mape(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate Mean Absolute Percentage Error."""

    errors = calculate_percentage_errors(
        actual,
        predicted,
    )

    if len(errors) == 0:
        return float("nan")

    return float(
        np.mean(errors) * 100
    )


def calculate_mdape(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate Median Absolute Percentage Error."""

    errors = calculate_percentage_errors(
        actual,
        predicted,
    )

    if len(errors) == 0:
        return float("nan")

    return float(
        np.median(errors) * 100
    )


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate regression evaluation metrics."""

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
# Candidate Models
# --------------------------------------------------

def create_reference_model() -> GradientBoostingRegressor:
    """Create the reference Gradient Boosting model."""

    return GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        loss="squared_error",
        random_state=RANDOM_STATE,
    )


def create_xgboost_model(
    n_estimators: int = 2000,
    use_early_stopping: bool = True,
) -> XGBRegressor:
    """Create an XGBoost regression model."""

    parameters: dict[str, Any] = {
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    if use_early_stopping:
        parameters["early_stopping_rounds"] = 75

    return XGBRegressor(**parameters)


def create_lightgbm_model(
    n_estimators: int = 2000,
) -> LGBMRegressor:
    """Create a LightGBM regression model."""

    return LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
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
# Validation Model Training
# --------------------------------------------------

def train_validation_candidates(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    dict[str, int | None],
]:
    """
    Train candidate models and evaluate them on validation data only.

    The test dataset is not used in this function.
    """

    trained_models: dict[str, object] = {}
    validation_rows: list[dict[str, float | str]] = {}
    best_iterations: dict[str, int | None] = {}

    # Fix list initialization for typing and runtime.
    validation_results: list[
        dict[str, float | str]
    ] = []

    # ------------------------------
    # Gradient Boosting
    # ------------------------------

    print("\nTraining validation candidate: Gradient Boosting")

    reference_model = create_reference_model()

    start_time = perf_counter()

    reference_model.fit(
        X_fit,
        y_fit,
    )

    training_seconds = perf_counter() - start_time

    reference_predictions = reference_model.predict(
        X_validation
    )

    reference_metrics = calculate_metrics(
        y_validation,
        reference_predictions,
    )

    validation_results.append(
        {
            "Model": "Gradient Boosting",
            **reference_metrics,
            "TrainingSeconds": training_seconds,
        }
    )

    trained_models["Gradient Boosting"] = (
        reference_model
    )

    best_iterations["Gradient Boosting"] = None

    # ------------------------------
    # XGBoost
    # ------------------------------

    print("Training validation candidate: XGBoost")

    xgboost_model = create_xgboost_model()

    start_time = perf_counter()

    xgboost_model.fit(
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

    training_seconds = perf_counter() - start_time

    xgboost_predictions = xgboost_model.predict(
        X_validation
    )

    xgboost_metrics = calculate_metrics(
        y_validation,
        xgboost_predictions,
    )

    validation_results.append(
        {
            "Model": "XGBoost",
            **xgboost_metrics,
            "TrainingSeconds": training_seconds,
        }
    )

    trained_models["XGBoost"] = xgboost_model

    xgb_best_iteration = getattr(
        xgboost_model,
        "best_iteration",
        None,
    )

    if xgb_best_iteration is not None:
        xgb_best_iteration = int(
            xgb_best_iteration
        ) + 1

    best_iterations["XGBoost"] = (
        xgb_best_iteration
    )

    # ------------------------------
    # LightGBM
    # ------------------------------

    print("Training validation candidate: LightGBM")

    lightgbm_model = create_lightgbm_model()

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

    lightgbm_model.fit(
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

    training_seconds = perf_counter() - start_time

    lightgbm_predictions = lightgbm_model.predict(
        X_validation
    )

    lightgbm_metrics = calculate_metrics(
        y_validation,
        lightgbm_predictions,
    )

    validation_results.append(
        {
            "Model": "LightGBM",
            **lightgbm_metrics,
            "TrainingSeconds": training_seconds,
        }
    )

    trained_models["LightGBM"] = lightgbm_model

    lgb_best_iteration = getattr(
        lightgbm_model,
        "best_iteration_",
        None,
    )

    if lgb_best_iteration is not None:
        lgb_best_iteration = int(
            lgb_best_iteration
        )

    best_iterations["LightGBM"] = (
        lgb_best_iteration
    )

    validation_df = pd.DataFrame(
        validation_results
    ).sort_values(
        by=["RMSE", "R2"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return (
        trained_models,
        validation_df,
        best_iterations,
    )


# --------------------------------------------------
# Final Model Training
# --------------------------------------------------

def train_final_model(
    selected_model_name: str,
    best_iterations: dict[str, int | None],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, float]:
    """
    Refit the selected model on the complete training dataset.

    The number of boosting iterations is determined from the
    validation stage, not from the test set.
    """

    if selected_model_name == "Gradient Boosting":
        final_model = create_reference_model()

    elif selected_model_name == "XGBoost":
        selected_iterations = (
            best_iterations.get("XGBoost")
            or 500
        )

        final_model = create_xgboost_model(
            n_estimators=selected_iterations,
            use_early_stopping=False,
        )

    elif selected_model_name == "LightGBM":
        selected_iterations = (
            best_iterations.get("LightGBM")
            or 500
        )

        final_model = create_lightgbm_model(
            n_estimators=selected_iterations,
        )

    else:
        raise ValueError(
            f"Unsupported model: {selected_model_name}"
        )

    start_time = perf_counter()

    final_model.fit(
        X_train,
        y_train,
    )

    training_seconds = perf_counter() - start_time

    return final_model, training_seconds


# --------------------------------------------------
# Price-Band Evaluation
# --------------------------------------------------

def assign_price_bands(
    prices: pd.Series,
) -> pd.Series:
    """
    Assign fixed property-price bands.

    Fixed thresholds avoid defining evaluation groups using
    information learned from the test distribution.
    """

    bins = [
        -np.inf,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        np.inf,
    ]

    labels = [
        "Under $500K",
        "$500K-$1M",
        "$1M-$2M",
        "$2M-$5M",
        "$5M+",
    ]

    return pd.cut(
        prices,
        bins=bins,
        labels=labels,
        right=False,
    )


def evaluate_price_bands(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate metrics for each actual-price band."""

    results: list[dict[str, float | str | int]] = []

    for price_band, band_df in prediction_df.groupby(
        "PriceBand",
        observed=False,
    ):
        if band_df.empty:
            continue

        metrics = calculate_metrics(
            band_df["ActualClosePrice"],
            band_df["PredictedClosePrice"].to_numpy(),
        )

        results.append(
            {
                "PriceBand": str(price_band),
                "ObservationCount": len(band_df),
                **metrics,
            }
        )

    return pd.DataFrame(results)


# --------------------------------------------------
# Save Outputs
# --------------------------------------------------

def save_final_model(
    final_model: object,
    selected_model_name: str,
    feature_names: list[str],
    training_medians: dict[str, float],
) -> None:
    """Save model and preprocessing metadata together."""

    artifact = {
        "model_name": selected_model_name,
        "model": final_model,
        "feature_names": feature_names,
        "training_medians": training_medians,
        "target_column": TARGET_COLUMN,
    }

    joblib.dump(
        artifact,
        FINAL_MODEL_FILE,
    )

    print(f"\nFinal model saved to: {FINAL_MODEL_FILE}")


def save_reports(
    metrics_summary: pd.DataFrame,
    price_band_metrics: pd.DataFrame,
    prediction_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Save metrics, predictions, and metadata."""

    metrics_summary.to_csv(
        METRICS_FILE,
        index=False,
    )

    price_band_metrics.to_csv(
        PRICE_BAND_FILE,
        index=False,
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Price-band metrics saved to: {PRICE_BAND_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


# --------------------------------------------------
# Figures
# --------------------------------------------------

def save_evaluation_figures(
    prediction_df: pd.DataFrame,
    price_band_metrics: pd.DataFrame,
    model_name: str,
) -> None:
    """Save the Week 8 evaluation figures."""

    actual = prediction_df["ActualClosePrice"]
    predicted = prediction_df["PredictedClosePrice"]

    minimum_price = min(
        actual.min(),
        predicted.min(),
    )

    maximum_price = max(
        actual.max(),
        predicted.max(),
    )

    # Actual vs predicted
    plt.figure(figsize=(8, 7))

    plt.scatter(
        actual,
        predicted,
        alpha=0.35,
    )

    plt.plot(
        [minimum_price, maximum_price],
        [minimum_price, maximum_price],
        linestyle="--",
    )

    plt.xlabel("Actual Close Price ($)")
    plt.ylabel("Predicted Close Price ($)")
    plt.title(
        f"Actual vs. Predicted Prices\n{model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        ACTUAL_PREDICTED_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Residual distribution
    plt.figure(figsize=(8, 6))

    plt.hist(
        prediction_df["Residual"],
        bins=50,
    )

    plt.xlabel(
        "Residual: Actual Price - Predicted Price ($)"
    )

    plt.ylabel("Frequency")

    plt.title(
        f"Residual Distribution\n{model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # Price-band MAPE
    plt.figure(figsize=(9, 6))

    plt.bar(
        price_band_metrics["PriceBand"],
        price_band_metrics["MAPE"],
    )

    plt.xlabel("Actual Property Price Band")
    plt.ylabel("MAPE (%)")
    plt.title(
        "Prediction Error by Property Price Band"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        PRICE_BAND_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nFigures saved:")
    print(f"- {ACTUAL_PREDICTED_FIGURE}")
    print(f"- {RESIDUAL_FIGURE}")
    print(f"- {PRICE_BAND_FIGURE}")


# --------------------------------------------------
# Main Program
# --------------------------------------------------

def main() -> None:
    """Run the complete leakage-safe Week 8 workflow."""

    print("=" * 70)
    print("California Property Price Prediction")
    print("Week 8: Final Model Evaluation")
    print("=" * 70)

    create_output_directories()

    train_df, test_df = load_data()

    validate_dataset_structure(
        train_df,
        test_df,
    )

    (
        X_train,
        y_train,
        X_test,
        y_test,
        training_medians,
    ) = prepare_model_data(
        train_df,
        test_df,
    )

    (
        X_fit,
        X_validation,
        y_fit,
        y_validation,
    ) = create_validation_split(
        X_train,
        y_train,
    )

    (
        validation_models,
        validation_metrics,
        best_iterations,
    ) = train_validation_candidates(
        X_fit,
        y_fit,
        X_validation,
        y_validation,
    )

    print("\nValidation results")
    print("-" * 95)

    print(
        validation_metrics.to_string(
            index=False,
            formatters={
                "MAE": lambda value: f"${value:,.2f}",
                "RMSE": lambda value: f"${value:,.2f}",
                "MAPE": lambda value: f"{value:.2f}%",
                "MdAPE": lambda value: f"{value:.2f}%",
                "R2": lambda value: f"{value:.4f}",
                "TrainingSeconds": lambda value: (
                    f"{value:.2f}"
                ),
            },
        )
    )

    # Selection is based only on validation RMSE.
    selected_model_name = str(
        validation_metrics.loc[0, "Model"]
    )

    print(
        f"\nSelected model using validation data: "
        f"{selected_model_name}"
    )

    final_model, final_training_seconds = train_final_model(
        selected_model_name,
        best_iterations,
        X_train,
        y_train,
    )

    # The test set is used for the first time here.
    test_start = perf_counter()

    test_predictions = final_model.predict(
        X_test
    )

    prediction_seconds = perf_counter() - test_start

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    metrics_summary = pd.DataFrame(
        [
            {
                "EvaluationSet": "Untouched Test Set",
                "SelectedModel": selected_model_name,
                **test_metrics,
                "TrainingSeconds": final_training_seconds,
                "PredictionSeconds": prediction_seconds,
                "TrainingObservations": len(X_train),
                "TestObservations": len(X_test),
                "FeatureCount": X_train.shape[1],
            }
        ]
    )

    prediction_df = pd.DataFrame(
        {
            "ActualClosePrice": (
                y_test.reset_index(drop=True)
            ),
            "PredictedClosePrice": (
                test_predictions
            ),
        }
    )

    prediction_df["Residual"] = (
        prediction_df["ActualClosePrice"]
        - prediction_df["PredictedClosePrice"]
    )

    prediction_df["AbsoluteError"] = (
        prediction_df["Residual"].abs()
    )

    prediction_df["AbsolutePercentageError"] = np.where(
        prediction_df["ActualClosePrice"] != 0,
        (
            prediction_df["AbsoluteError"]
            / prediction_df["ActualClosePrice"]
        )
        * 100,
        np.nan,
    )

    prediction_df["PriceBand"] = assign_price_bands(
        prediction_df["ActualClosePrice"]
    )

    price_band_metrics = evaluate_price_bands(
        prediction_df
    )

    print("\nFinal untouched-test results")
    print("-" * 50)
    print(f"Model: {selected_model_name}")
    print(f"MAE: ${test_metrics['MAE']:,.2f}")
    print(f"RMSE: ${test_metrics['RMSE']:,.2f}")
    print(f"MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"MdAPE: {test_metrics['MdAPE']:.2f}%")
    print(f"R²: {test_metrics['R2']:.4f}")

    print("\nPrice-band performance")
    print("-" * 95)

    print(
        price_band_metrics.to_string(
            index=False,
            formatters={
                "MAE": lambda value: f"${value:,.2f}",
                "RMSE": lambda value: f"${value:,.2f}",
                "MAPE": lambda value: f"{value:.2f}%",
                "MdAPE": lambda value: f"{value:.2f}%",
                "R2": lambda value: f"{value:.4f}",
            },
        )
    )

    metadata: dict[str, Any] = {
        "selected_model": selected_model_name,
        "model_selection_set": (
            "Chronological validation subset from training data"
        ),
        "final_evaluation_set": (
            "Untouched chronological test set"
        ),
        "validation_fraction": VALIDATION_FRACTION,
        "feature_count": X_train.shape[1],
        "training_observations": len(X_train),
        "test_observations": len(X_test),
        "best_iterations": best_iterations,
        "test_metrics": test_metrics,
    }

    save_final_model(
        final_model,
        selected_model_name,
        X_train.columns.tolist(),
        training_medians,
    )

    save_reports(
        metrics_summary,
        price_band_metrics,
        prediction_df,
        metadata,
    )

    save_evaluation_figures(
        prediction_df,
        price_band_metrics,
        selected_model_name,
    )

    print(
        "\nWeek 8 leakage-safe evaluation "
        "completed successfully."
    )


if __name__ == "__main__":
    main()