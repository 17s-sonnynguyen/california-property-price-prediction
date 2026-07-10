"""
train_baseline.py

Week 4 - Baseline Model

This script:
1. Loads the processed training and testing datasets
2. Separates features and target values
3. Trains a Linear Regression model
4. Generates predictions
5. Evaluates the model
6. Saves evaluation results
7. Saves the trained model
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train_processed.csv"
TEST_FILE = DATA_DIR / "test_processed.csv"

MODEL_FILE = MODEL_DIR / "linear_regression.pkl"
METRICS_FILE = REPORTS_DIR / "baseline_metrics.csv"
PREDICTIONS_FILE = REPORTS_DIR / "baseline_predictions.csv"


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the processed training and testing datasets.

    Returns:
        tuple:
            train_df: Processed training dataset
            test_df: Processed testing dataset
    """

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Run src/preprocessing.py before training the model."
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_FILE}\n"
            "Run src/preprocessing.py before training the model."
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("Data loaded successfully.")
    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")

    return train_df, test_df


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separate the target variable from the predictor variables.

    Args:
        train_df: Processed training dataset
        test_df: Processed testing dataset

    Returns:
        tuple:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
    """

    target_column = "ClosePrice"

    if target_column not in train_df.columns:
        raise ValueError(
            f"{target_column} is missing from the training dataset."
        )

    if target_column not in test_df.columns:
        raise ValueError(
            f"{target_column} is missing from the testing dataset."
        )

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Ensure train and test contain the same columns
    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )

    print("\nFeatures prepared successfully.")
    print(f"Number of training features: {X_train.shape[1]}")
    print(f"Number of testing features: {X_test.shape[1]}")

    return X_train, y_train, X_test, y_test


def validate_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Validate that the datasets are ready for model training.
    """

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing feature data is empty.")

    if y_train.empty or y_test.empty:
        raise ValueError("Training or testing target data is empty.")

    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()

    if train_missing > 0:
        raise ValueError(
            f"Training features contain {train_missing} missing values."
        )

    if test_missing > 0:
        raise ValueError(
            f"Testing features contain {test_missing} missing values."
        )

    if y_train.isnull().any():
        raise ValueError("Training target contains missing values.")

    if y_test.isnull().any():
        raise ValueError("Testing target contains missing values.")

    non_numeric_train = X_train.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    non_numeric_test = X_test.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    if non_numeric_train:
        raise TypeError(
            "Training data contains non-numeric columns: "
            f"{non_numeric_train}"
        )

    if non_numeric_test:
        raise TypeError(
            "Testing data contains non-numeric columns: "
            f"{non_numeric_test}"
        )

    print("\nData validation completed successfully.")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    """
    Train the baseline Linear Regression model.

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Trained LinearRegression model
    """

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("\nLinear Regression model trained successfully.")

    return model


def calculate_mape(
    actual: pd.Series,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Rows where the actual value is zero are excluded to prevent
    division-by-zero errors.

    Args:
        actual: Actual property prices
        predicted: Predicted property prices

    Returns:
        MAPE percentage
    """

    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)

    non_zero_mask = actual_array != 0

    if not np.any(non_zero_mask):
        return float("nan")

    percentage_errors = np.abs(
        (
            actual_array[non_zero_mask]
            - predicted_array[non_zero_mask]
        )
        / actual_array[non_zero_mask]
    )

    return float(np.mean(percentage_errors) * 100)


def calculate_mdape(
    actual: pd.Series,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Median Absolute Percentage Error.

    Args:
        actual: Actual property prices
        predicted: Predicted property prices

    Returns:
        MdAPE percentage
    """

    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)

    non_zero_mask = actual_array != 0

    if not np.any(non_zero_mask):
        return float("nan")

    percentage_errors = np.abs(
        (
            actual_array[non_zero_mask]
            - predicted_array[non_zero_mask]
        )
        / actual_array[non_zero_mask]
    )

    return float(np.median(percentage_errors) * 100)


def evaluate_model(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate the baseline model.

    Args:
        y_test: Actual property prices
        predictions: Predicted property prices

    Returns:
        Dictionary containing model evaluation metrics
    """

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    mape = calculate_mape(
        actual=y_test,
        predicted=predictions,
    )

    mdape = calculate_mdape(
        actual=y_test,
        predicted=predictions,
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "MdAPE": float(mdape),
        "R2": float(r2),
    }

    return metrics


def display_metrics(metrics: dict[str, float]) -> None:
    """
    Print evaluation metrics in a readable format.
    """

    print("\nBaseline Model Results")
    print("-" * 35)
    print(f"MAE:   ${metrics['MAE']:,.2f}")
    print(f"RMSE:  ${metrics['RMSE']:,.2f}")
    print(f"MAPE:   {metrics['MAPE']:.2f}%")
    print(f"MdAPE:  {metrics['MdAPE']:.2f}%")
    print(f"R²:     {metrics['R2']:.4f}")


def save_model(model: LinearRegression) -> None:
    """
    Save the trained Linear Regression model.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(f"\nModel saved to: {MODEL_FILE}")


def save_metrics(metrics: dict[str, float]) -> None:
    """
    Save model evaluation metrics as a CSV file.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "Linear Regression",
                **metrics,
            }
        ]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    print(f"Metrics saved to: {METRICS_FILE}")


def save_predictions(
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    """
    Save actual and predicted property prices.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df = pd.DataFrame(
        {
            "ActualClosePrice": y_test.reset_index(drop=True),
            "PredictedClosePrice": predictions,
        }
    )

    predictions_df["Residual"] = (
        predictions_df["ActualClosePrice"]
        - predictions_df["PredictedClosePrice"]
    )

    predictions_df["AbsoluteError"] = (
        predictions_df["Residual"].abs()
    )

    predictions_df["AbsolutePercentageError"] = np.where(
        predictions_df["ActualClosePrice"] != 0,
        (
            predictions_df["AbsoluteError"]
            / predictions_df["ActualClosePrice"]
        )
        * 100,
        np.nan,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(f"Predictions saved to: {PREDICTIONS_FILE}")


def display_largest_coefficients(
    model: LinearRegression,
    feature_names: pd.Index,
    number_of_features: int = 15,
) -> None:
    """
    Display the features with the largest absolute coefficients.

    Note:
        Coefficient size should be interpreted carefully because
        the features are measured using different scales.
    """

    coefficients_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_,
        }
    )

    coefficients_df["AbsoluteCoefficient"] = (
        coefficients_df["Coefficient"].abs()
    )

    coefficients_df = coefficients_df.sort_values(
        by="AbsoluteCoefficient",
        ascending=False,
    )

    print(
        f"\nTop {number_of_features} Features "
        "by Absolute Coefficient"
    )
    print("-" * 60)

    print(
        coefficients_df[
            ["Feature", "Coefficient"]
        ].head(number_of_features).to_string(index=False)
    )


# --------------------------------------------------
# Main Program
# --------------------------------------------------

def main() -> None:
    """
    Run the complete baseline model training workflow.
    """

    print("=" * 60)
    print("California Property Price Prediction")
    print("Week 4: Baseline Linear Regression")
    print("=" * 60)

    train_df, test_df = load_data()

    X_train, y_train, X_test, y_test = prepare_features(
        train_df=train_df,
        test_df=test_df,
    )

    validate_data(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    predictions = model.predict(X_test)

    metrics = evaluate_model(
        y_test=y_test,
        predictions=predictions,
    )

    display_metrics(metrics)

    display_largest_coefficients(
        model=model,
        feature_names=X_train.columns,
    )

    save_model(model)

    save_metrics(metrics)

    save_predictions(
        y_test=y_test,
        predictions=predictions,
    )

    print("\nBaseline model workflow completed successfully.")


if __name__ == "__main__":
    main()