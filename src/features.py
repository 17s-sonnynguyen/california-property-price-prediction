"""
features.py

Week 6 - Feature Engineering

This script:
1. Loads processed training and testing datasets
2. Creates real-estate-related engineered features
3. Safely handles division by zero, missing values, and infinity
4. Uses training-set statistics to clean both datasets
5. Aligns training and testing columns
6. Validates the final outputs
7. Saves feature-engineered datasets

Outputs:
    data/train_features.csv
    data/test_features.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Project Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_INPUT_FILE = DATA_DIR / "train_processed.csv"
TEST_INPUT_FILE = DATA_DIR / "test_processed.csv"

TRAIN_OUTPUT_FILE = DATA_DIR / "train_features.csv"
TEST_OUTPUT_FILE = DATA_DIR / "test_features.csv"

TARGET_COLUMN = "ClosePrice"

REQUIRED_FEATURE_COLUMNS = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "ListPrice",
    "PropertyAge",
]


# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the processed training and testing datasets.

    Returns:
        Tuple containing the training and testing DataFrames.
    """

    if not TRAIN_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_INPUT_FILE}\n"
            "Run src/preprocessing.py before running features.py."
        )

    if not TEST_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_INPUT_FILE}\n"
            "Run src/preprocessing.py before running features.py."
        )

    train_df = pd.read_csv(TRAIN_INPUT_FILE)
    test_df = pd.read_csv(TEST_INPUT_FILE)

    print("Processed datasets loaded successfully.")
    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")

    return train_df, test_df


# --------------------------------------------------
# Required Column Validation
# --------------------------------------------------

def validate_required_columns(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Validate that all required columns are present.

    Args:
        df: Dataset to validate.
        dataset_name: Human-readable dataset label.
    """

    required_columns = [
        TARGET_COLUMN,
        *REQUIRED_FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


# --------------------------------------------------
# Numeric Conversion
# --------------------------------------------------

def convert_required_columns_to_numeric(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert required feature columns and the target to numeric values.

    Non-convertible values are changed to NaN and handled later.

    Args:
        df: Input dataset.

    Returns:
        Dataset with required columns converted to numeric.
    """

    converted_df = df.copy()

    numeric_columns = [
        TARGET_COLUMN,
        *REQUIRED_FEATURE_COLUMNS,
    ]

    for column in numeric_columns:
        converted_df[column] = pd.to_numeric(
            converted_df[column],
            errors="coerce",
        )

    return converted_df


# --------------------------------------------------
# Safe Division
# --------------------------------------------------

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Perform division safely.

    Division by zero and infinite results are converted to NaN.
    These values are filled later using training-set medians.

    Args:
        numerator: Numerator values.
        denominator: Denominator values.

    Returns:
        Division result as a float Series.
    """

    numerator_numeric = pd.to_numeric(
        numerator,
        errors="coerce",
    ).astype(float)

    denominator_numeric = pd.to_numeric(
        denominator,
        errors="coerce",
    ).astype(float)

    denominator_numeric = denominator_numeric.replace(
        0,
        np.nan,
    )

    result = numerator_numeric / denominator_numeric

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result.astype(float)


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create engineered features for property-price prediction.

    ClosePrice is not used to construct any predictor features.

    Args:
        df: Processed property dataset.

    Returns:
        Feature-engineered dataset.
    """

    featured_df = convert_required_columns_to_numeric(df)

    # Living area per bedroom
    featured_df["LivingAreaPerBedroom"] = safe_divide(
        featured_df["LivingArea"],
        featured_df["BedroomsTotal"],
    )

    # Living area per bathroom
    featured_df["LivingAreaPerBathroom"] = safe_divide(
        featured_df["LivingArea"],
        featured_df["BathroomsTotalInteger"],
    )

    # Bedroom-to-bathroom relationship
    featured_df["BedBathRatio"] = safe_divide(
        featured_df["BedroomsTotal"],
        featured_df["BathroomsTotalInteger"],
    )

    # Listing price per square foot
    featured_df["ListPricePerSquareFoot"] = safe_divide(
        featured_df["ListPrice"],
        featured_df["LivingArea"],
    )

    # Lot size relative to living area
    featured_df["LotToLivingAreaRatio"] = safe_divide(
        featured_df["LotSizeSquareFeet"],
        featured_df["LivingArea"],
    )

    # Approximate number of major rooms
    featured_df["TotalRoomsApprox"] = (
        featured_df["BedroomsTotal"]
        + featured_df["BathroomsTotalInteger"]
    )

    # Approximate outdoor space
    featured_df["OutdoorSpaceApprox"] = (
        featured_df["LotSizeSquareFeet"]
        - featured_df["LivingArea"]
    )

    # Log transformations
    featured_df["LogLivingArea"] = np.log1p(
        featured_df["LivingArea"].clip(lower=0)
    )

    featured_df["LogLotSize"] = np.log1p(
        featured_df["LotSizeSquareFeet"].clip(lower=0)
    )

    featured_df["LogListPrice"] = np.log1p(
        featured_df["ListPrice"].clip(lower=0)
    )

    # Property age categories
    featured_df["IsNewProperty"] = (
        featured_df["PropertyAge"] <= 5
    ).astype("int64")

    featured_df["IsRecentProperty"] = (
        featured_df["PropertyAge"] <= 15
    ).astype("int64")

    featured_df["IsOlderProperty"] = (
        featured_df["PropertyAge"] >= 50
    ).astype("int64")

    # Nonlinear property age
    featured_df["PropertyAgeSquared"] = (
        featured_df["PropertyAge"] ** 2
    )

    # Replace any unexpected infinite values
    featured_df = featured_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return featured_df


# --------------------------------------------------
# Missing Value Treatment
# --------------------------------------------------

def fill_engineered_missing_values(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill feature missing values using training-set statistics.

    Numeric columns are filled with training medians.
    Non-numeric columns are filled with the training mode or "Unknown".

    The test-set distribution is not used to calculate fill values.

    Args:
        train_df: Feature-engineered training dataset.
        test_df: Feature-engineered testing dataset.

    Returns:
        Cleaned training and testing datasets.
    """

    train_clean = train_df.copy()
    test_clean = test_df.copy()

    train_clean = train_clean.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    test_clean = test_clean.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_columns = [
        column
        for column in train_clean.columns
        if column != TARGET_COLUMN
    ]

    for column in feature_columns:
        if column not in test_clean.columns:
            continue

        if pd.api.types.is_numeric_dtype(train_clean[column]):
            training_values = pd.to_numeric(
                train_clean[column],
                errors="coerce",
            )

            testing_values = pd.to_numeric(
                test_clean[column],
                errors="coerce",
            )

            training_median = training_values.median()

            if pd.isna(training_median):
                training_median = 0.0

            train_clean[column] = training_values.fillna(
                training_median
            ).astype(float)

            test_clean[column] = testing_values.fillna(
                training_median
            ).astype(float)

        else:
            training_mode = train_clean[column].mode(
                dropna=True
            )

            if training_mode.empty:
                fill_value = "Unknown"
            else:
                fill_value = training_mode.iloc[0]

            train_clean[column] = (
                train_clean[column]
                .fillna(fill_value)
                .astype(str)
            )

            test_clean[column] = (
                test_clean[column]
                .fillna(fill_value)
                .astype(str)
            )

    return train_clean, test_clean


# --------------------------------------------------
# Align Train and Test
# --------------------------------------------------

def align_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure train and test contain identical predictor columns.

    The target is temporarily removed, the predictor columns are aligned,
    and then the target is added back.

    Args:
        train_df: Training dataset.
        test_df: Testing dataset.

    Returns:
        Aligned training and testing datasets.
    """

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} is missing from the training dataset."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} is missing from the testing dataset."
        )

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = pd.to_numeric(
        train_df[TARGET_COLUMN],
        errors="coerce",
    )

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = pd.to_numeric(
        test_df[TARGET_COLUMN],
        errors="coerce",
    )

    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0,
    )

    aligned_train = X_train.copy()
    aligned_test = X_test.copy()

    aligned_train.insert(
        0,
        TARGET_COLUMN,
        y_train.to_numpy(),
    )

    aligned_test.insert(
        0,
        TARGET_COLUMN,
        y_test.to_numpy(),
    )

    return aligned_train, aligned_test


# --------------------------------------------------
# Numeric Array Helper
# --------------------------------------------------

def numeric_array_for_validation(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Convert numeric columns into a float NumPy array.

    This avoids np.isinf errors caused by nullable Pandas dtypes,
    object arrays, and mixed numeric extension types.

    Args:
        df: Dataset containing numeric and non-numeric columns.

    Returns:
        Float NumPy array containing only numeric columns.
    """

    numeric_df = df.select_dtypes(
        include=["number", "bool"],
    ).copy()

    if numeric_df.empty:
        return np.empty(
            shape=(len(df), 0),
            dtype=float,
        )

    for column in numeric_df.columns:
        numeric_df[column] = pd.to_numeric(
            numeric_df[column],
            errors="coerce",
        )

    return numeric_df.to_numpy(
        dtype=float,
        na_value=np.nan,
    )


# --------------------------------------------------
# Final Output Validation
# --------------------------------------------------

def validate_output(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Validate the feature-engineered datasets.

    Checks:
    - Matching columns
    - No missing predictors
    - No missing targets
    - No infinite numeric values
    - No duplicate column names

    Args:
        train_df: Final training dataset.
        test_df: Final testing dataset.
    """

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(
            "Training and testing columns do not match."
        )

    if train_df.columns.duplicated().any():
        duplicate_columns = train_df.columns[
            train_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Training data contains duplicate columns: "
            f"{duplicate_columns}"
        )

    if test_df.columns.duplicated().any():
        duplicate_columns = test_df.columns[
            test_df.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Testing data contains duplicate columns: "
            f"{duplicate_columns}"
        )

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} is missing from training data."
        )

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} is missing from testing data."
        )

    train_features = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    test_features = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    train_missing = int(
        train_features.isna().sum().sum()
    )

    test_missing = int(
        test_features.isna().sum().sum()
    )

    if train_missing > 0:
        raise ValueError(
            f"Training predictors contain "
            f"{train_missing} missing values."
        )

    if test_missing > 0:
        raise ValueError(
            f"Testing predictors contain "
            f"{test_missing} missing values."
        )

    train_target_missing = int(
        train_df[TARGET_COLUMN].isna().sum()
    )

    test_target_missing = int(
        test_df[TARGET_COLUMN].isna().sum()
    )

    if train_target_missing > 0:
        raise ValueError(
            f"Training target contains "
            f"{train_target_missing} missing values."
        )

    if test_target_missing > 0:
        raise ValueError(
            f"Testing target contains "
            f"{test_target_missing} missing values."
        )

    train_numeric_array = numeric_array_for_validation(
        train_df
    )

    test_numeric_array = numeric_array_for_validation(
        test_df
    )

    if np.isinf(train_numeric_array).any():
        raise ValueError(
            "Training data contains infinite numeric values."
        )

    if np.isinf(test_numeric_array).any():
        raise ValueError(
            "Testing data contains infinite numeric values."
        )

    print("\nFinal feature datasets passed validation.")


# --------------------------------------------------
# Save Data
# --------------------------------------------------

def save_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save feature-engineered datasets as CSV files.

    Args:
        train_df: Final training dataset.
        test_df: Final testing dataset.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_OUTPUT_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_OUTPUT_FILE,
        index=False,
    )

    print(f"\nTraining features saved to: {TRAIN_OUTPUT_FILE}")
    print(f"Testing features saved to: {TEST_OUTPUT_FILE}")


# --------------------------------------------------
# Feature Summary
# --------------------------------------------------

def display_feature_summary(
    original_train: pd.DataFrame,
    engineered_train: pd.DataFrame,
    engineered_test: pd.DataFrame,
) -> None:
    """
    Display a summary of newly created features.

    Args:
        original_train: Original processed training dataset.
        engineered_train: Final engineered training dataset.
        engineered_test: Final engineered testing dataset.
    """

    original_features = {
        column
        for column in original_train.columns
        if column != TARGET_COLUMN
    }

    engineered_features = {
        column
        for column in engineered_train.columns
        if column != TARGET_COLUMN
    }

    added_features = sorted(
        engineered_features - original_features
    )

    print("\nFeature engineering summary")
    print("-" * 50)

    print(
        f"Original predictor count: "
        f"{len(original_features)}"
    )

    print(
        f"Final predictor count: "
        f"{len(engineered_features)}"
    )

    print(
        f"Engineered features added: "
        f"{len(added_features)}"
    )

    print(
        f"Final training shape: "
        f"{engineered_train.shape}"
    )

    print(
        f"Final testing shape: "
        f"{engineered_test.shape}"
    )

    print("\nNew features:")

    for feature_number, feature_name in enumerate(
        added_features,
        start=1,
    ):
        print(
            f"{feature_number:>2}. {feature_name}"
        )


# --------------------------------------------------
# Complete Pipeline
# --------------------------------------------------

def build_feature_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full feature-engineering transformation.

    This function is useful for both scripts and notebooks.

    Args:
        train_df: Processed training dataset.
        test_df: Processed testing dataset.

    Returns:
        Final feature-engineered train and test datasets.
    """

    validate_required_columns(
        train_df,
        dataset_name="Training dataset",
    )

    validate_required_columns(
        test_df,
        dataset_name="Testing dataset",
    )

    train_engineered = create_features(
        train_df
    )

    test_engineered = create_features(
        test_df
    )

    train_engineered, test_engineered = (
        fill_engineered_missing_values(
            train_df=train_engineered,
            test_df=test_engineered,
        )
    )

    train_engineered, test_engineered = align_datasets(
        train_df=train_engineered,
        test_df=test_engineered,
    )

    validate_output(
        train_df=train_engineered,
        test_df=test_engineered,
    )

    return train_engineered, test_engineered


# --------------------------------------------------
# Main Program
# --------------------------------------------------

def main() -> None:
    """
    Run the complete Week 6 feature-engineering pipeline.
    """

    print("=" * 65)
    print("California Property Price Prediction")
    print("Week 6: Feature Engineering")
    print("=" * 65)

    train_df, test_df = load_data()

    train_engineered, test_engineered = (
        build_feature_datasets(
            train_df=train_df,
            test_df=test_df,
        )
    )

    display_feature_summary(
        original_train=train_df,
        engineered_train=train_engineered,
        engineered_test=test_engineered,
    )

    save_data(
        train_df=train_engineered,
        test_df=test_engineered,
    )

    print(
        "\nWeek 6 feature engineering "
        "completed successfully."
    )


if __name__ == "__main__":
    main()