"""
preprocessing.py

Week 3 - Data Preprocessing

Purpose:
- Load all CRMLS sold-property datasets
- Filter residential single-family homes
- Handle missing values
- Engineer features
- Encode categorical variables
- Create train/test split
- Save processed datasets
"""

import pandas as pd


# --------------------------------------------------
# Load Data
# --------------------------------------------------

FILES = [
    "data/CRMLSSold202512.csv",
    "data/CRMLSSold202601.csv",
    "data/CRMLSSold202602.csv",
    "data/CRMLSSold202603.csv",
    "data/CRMLSSold202604.csv",
    "data/CRMLSSold202605.csv",
]

dfs = [pd.read_csv(file) for file in FILES]

df = pd.concat(dfs, ignore_index=True)

print(f"Original Shape: {df.shape}")


# --------------------------------------------------
# Filter Residential Single Family Homes
# --------------------------------------------------

df = df[
    (df["PropertyType"] == "Residential")
    & (df["PropertySubType"] == "SingleFamilyResidence")
]

print(f"Filtered Shape: {df.shape}")


# --------------------------------------------------
# Select Features
# --------------------------------------------------

FEATURES = [
    "ClosePrice",
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "YearBuilt",
    "City",
    "CountyOrParish",
    "ListPrice",
    "CloseDate",
]

df = df[FEATURES]


# --------------------------------------------------
# Missing Value Analysis
# --------------------------------------------------

missing_pct = (
    df.isnull()
    .mean()
    .sort_values(ascending=False)
    * 100
)

print("\nMissing Value Percentage:")
print(missing_pct)


# --------------------------------------------------
# Remove Columns With Excessive Missing Values
# --------------------------------------------------

THRESHOLD = 70

keep_cols = missing_pct[
    missing_pct < THRESHOLD
].index

df = df[keep_cols]


# --------------------------------------------------
# Date Processing
# --------------------------------------------------

df["CloseDate"] = pd.to_datetime(df["CloseDate"])


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

df["PropertyAge"] = (
    df["CloseDate"].dt.year
    - df["YearBuilt"]
)

df = df.drop(
    columns=["YearBuilt"]
)


# --------------------------------------------------
# Numerical Imputation
# --------------------------------------------------

numerical_cols = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "ListPrice",
    "PropertyAge",
]

for col in numerical_cols:
    df[col] = df[col].fillna(
        df[col].median()
    )


# --------------------------------------------------
# Categorical Imputation
# --------------------------------------------------

categorical_cols = [
    "City",
    "CountyOrParish",
]

for col in categorical_cols:
    df[col] = df[col].fillna(
        "Unknown"
    )


# --------------------------------------------------
# One-Hot Encoding
# --------------------------------------------------

df = pd.get_dummies(
    df,
    columns=[
        "City",
        "CountyOrParish",
    ],
    drop_first=True,
)


# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

test_df = df[
    df["CloseDate"] >= "2026-05-01"
]

train_df = df[
    df["CloseDate"] < "2026-05-01"
]


# --------------------------------------------------
# Remove Date Column
# --------------------------------------------------

train_df = train_df.drop(
    columns=["CloseDate"]
)

test_df = test_df.drop(
    columns=["CloseDate"]
)


# --------------------------------------------------
# Save Processed Data
# --------------------------------------------------

train_df.to_csv(
    "data/train_processed.csv",
    index=False,
)

test_df.to_csv(
    "data/test_processed.csv",
    index=False,
)

print("\nPreprocessing Complete")
print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {test_df.shape}")

print("\nSaved:")
print("data/train_processed.csv")
print("data/test_processed.csv")