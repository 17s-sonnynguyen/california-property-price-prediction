"""
compare_models.py

Week 5 - Model Comparison

This script:
1. Loads the processed training and testing datasets
2. Separates predictor variables and the target variable
3. Trains multiple regression models
4. Evaluates each model using common regression metrics
5. Saves the comparison results
6. Saves each trained model
7. Creates and saves a model comparison chart
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeRegressor


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = DATA_DIR / "train_processed.csv"
TEST_FILE = DATA_DIR / "test_processed.csv"

METRICS_FILE = REPORTS_DIR / "comparison_metrics.csv"
CHART_FILE = REPORTS_DIR / "model_comparison.png"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the processed training and testing datasets.

    Returns:
        A tuple containing the training and testing DataFrames.
    """

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Run src/preprocessing.py before comparing models."
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_FILE}\n"
            "Run src/preprocessing.py before comparing models."
        )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print("Data loaded successfully.")
    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")

    return train_df, test_df


# --------------------------------------------------
# Prepare Features
# --------------------------------------------------

def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separate predictor variables from the target variable.

    Args:
        train_df: Processed training dataset.
        test_df: Processed testing dataset.

    Returns:
        X_train, y_train, X_test, and y_test.
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

    # Ensure both datasets contain the same feature columns
    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )

    print("\nFeatures prepared successfully.")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, y_train, X_test, y_test


# --------------------------------------------------
# Validate Data
# --------------------------------------------------

def validate_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Validate that the datasets are suitable for model training.
    """

    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing feature data is empty.")

    if y_train.empty or y_test.empty:
        raise ValueError("Training or testing target data is empty.")

    train_missing = int(X_train.isnull().sum().sum())
    test_missing = int(X_test.isnull().sum().sum())

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

    print("Data validation completed successfully.")


# --------------------------------------------------
# Define Models
# --------------------------------------------------

def create_models() -> dict[str, object]:
    """
    Create the regression models used for comparison.

    Returns:
        A dictionary mapping model names to model objects.
    """

    return {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42,
        ),
    }


# --------------------------------------------------
# Evaluation Metrics
# --------------------------------------------------

def calculate_mape(
    actual: pd.Series,
    predicted: np.ndarray,
) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Actual values equal to zero are excluded to prevent
    division-by-zero errors.
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
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Train and evaluate one regression model.

    Returns:
        A dictionary of metrics and the model predictions.
    """

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

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

    return metrics, predictions


# --------------------------------------------------
# Model Filename Helper
# --------------------------------------------------

def create_model_filename(model_name: str) -> str:
    """
    Convert a model name into a safe filename.
    """

    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + ".pkl"
    )


# --------------------------------------------------
# Train and Compare Models
# --------------------------------------------------

def compare_models(
    models: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Train, evaluate, and save all models.

    Returns:
        A DataFrame containing model comparison metrics.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    total_models = len(models)

    for model_number, (model_name, model) in enumerate(
        models.items(),
        start=1,
    ):
        print("\n" + "=" * 60)
        print(
            f"Training model {model_number} of "
            f"{total_models}: {model_name}"
        )
        print("=" * 60)

        metrics, _ = evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(
            {
                "Model": model_name,
                **metrics,
            }
        )

        model_filename = create_model_filename(model_name)
        model_path = MODELS_DIR / model_filename

        joblib.dump(
            model,
            model_path,
        )

        print(f"MAE:   ${metrics['MAE']:,.2f}")
        print(f"RMSE:  ${metrics['RMSE']:,.2f}")
        print(f"MAPE:   {metrics['MAPE']:.2f}%")
        print(f"MdAPE:  {metrics['MdAPE']:.2f}%")
        print(f"R²:     {metrics['R2']:.4f}")
        print(f"Saved model: {model_path}")

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by="R2",
        ascending=False,
    ).reset_index(drop=True)

    return comparison_df


# --------------------------------------------------
# Display Results
# --------------------------------------------------

def display_results(
    comparison_df: pd.DataFrame,
) -> None:
    """
    Print the model comparison table.
    """

    display_df = comparison_df.copy()

    display_df["MAE"] = display_df["MAE"].map(
        lambda value: f"${value:,.2f}"
    )

    display_df["RMSE"] = display_df["RMSE"].map(
        lambda value: f"${value:,.2f}"
    )

    display_df["MAPE"] = display_df["MAPE"].map(
        lambda value: f"{value:.2f}%"
    )

    display_df["MdAPE"] = display_df["MdAPE"].map(
        lambda value: f"{value:.2f}%"
    )

    display_df["R2"] = display_df["R2"].map(
        lambda value: f"{value:.4f}"
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    print(
        display_df.to_string(
            index=False,
        )
    )


# --------------------------------------------------
# Save Results
# --------------------------------------------------

def save_results(
    comparison_df: pd.DataFrame,
) -> None:
    """
    Save the comparison metrics to a CSV file.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    print(f"\nComparison metrics saved to: {METRICS_FILE}")


# --------------------------------------------------
# Create Comparison Chart
# --------------------------------------------------

def create_comparison_chart(
    comparison_df: pd.DataFrame,
) -> None:
    """
    Create and save an R-squared comparison chart.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Model"],
        comparison_df["R2"],
    )

    plt.xlabel("Model")
    plt.ylabel("R² Score")
    plt.title("Regression Model Comparison")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.ylim(
        min(0, comparison_df["R2"].min() - 0.05),
        min(1, comparison_df["R2"].max() + 0.05),
    )

    plt.tight_layout()

    plt.savefig(
        CHART_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Comparison chart saved to: {CHART_FILE}")


# --------------------------------------------------
# Identify Best Model
# --------------------------------------------------

def display_best_model(
    comparison_df: pd.DataFrame,
) -> None:
    """
    Display the highest-performing model based on R-squared.
    """

    best_model = comparison_df.iloc[0]

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model: {best_model['Model']}")
    print(f"MAE: ${best_model['MAE']:,.2f}")
    print(f"RMSE: ${best_model['RMSE']:,.2f}")
    print(f"MAPE: {best_model['MAPE']:.2f}%")
    print(f"MdAPE: {best_model['MdAPE']:.2f}%")
    print(f"R²: {best_model['R2']:.4f}")


# --------------------------------------------------
# Main Program
# --------------------------------------------------

def main() -> None:
    """
    Run the complete Week 5 model comparison workflow.
    """

    print("=" * 60)
    print("California Property Price Prediction")
    print("Week 5: Regression Model Comparison")
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

    models = create_models()

    comparison_df = compare_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    display_results(comparison_df)

    save_results(comparison_df)

    create_comparison_chart(comparison_df)

    display_best_model(comparison_df)

    print("\nWeek 5 model comparison completed successfully.")


if __name__ == "__main__":
    main()