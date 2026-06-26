"""
clean_data.py
─────────────────────────────────────────────
Task 1: Data Cleaning and Preparation
─────────────────────────────────────────────
PURPOSE : Loads raw Amazon survey CSV, applies
          all cleaning steps, and saves the
          cleaned version to data/processed/.
AUTHOR  : Mrudula
─────────────────────────────────────────────
"""

import pandas as pd
import numpy as np


# ── STEP 1: Load Raw Data ────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV file into a pandas DataFrame."""
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── STEP 2: Fix Column Names ─────────────────────────────
def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from all column names.
    Example: 'Rating_Accuracy ' → 'Rating_Accuracy'
    """
    df.columns = df.columns.str.strip()
    print("✅ Column names cleaned")
    return df


# ── STEP 3: Rename Duplicate Column ──────────────────────
def rename_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset has TWO columns named
    'Personalized_Recommendation_Frequency':
      - Column 5  : text version  (Sometimes/Never/Always)
      - Column 17 : numeric version (1/2/3/4/5)

    We rename the numeric duplicate to 'Rec_Freq_Numeric'
    so pandas can handle both without confusion.
    """
    cols = list(df.columns)
    seen = {}
    for i, col in enumerate(cols):
        if col == "Personalized_Recommendation_Frequency":
            if col in seen:
                cols[i] = "Rec_Freq_Numeric"
            else:
                seen[col] = i
    df.columns = cols
    print("✅ Duplicate column renamed → 'Rec_Freq_Numeric'")
    return df


# ── STEP 4: Remove Duplicate Rows ────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate survey responses.
    Same person submitting form twice = noise in data.
    """
    before = len(df)
    df = df.drop_duplicates()
    print(f"✅ Duplicate rows removed: {before - len(df)}")
    return df


# ── STEP 5: Standardize Categorical Columns ───────────────
def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize text columns to consistent Title Case.

    WHY: Raw data has inconsistencies like:
      'male', 'Male', 'MALE' → all become 'Male'
      'once a week', 'Once A Week' → 'Once A Week'

    This prevents same value being counted separately
    due to case differences.
    """
    cat_cols = [
        "Gender",
        "Purchase_Frequency",
        "Browsing_Frequency",
        "Product_Search_Method",
        "Search_Result_Exploration",
        "Add_to_Cart_Browsing",
        "Cart_Completion_Frequency",
        "Cart_Abandonment_Factors",
        "Saveforlater_Frequency",
        "Review_Left",
        "Review_Reliability",
        "Review_Helpfulness",
        "Recommendation_Helpfulness",
        "Personalized_Recommendation_Frequency"
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    print("✅ Categorical columns standardized to Title Case")
    return df


# ── STEP 6: Handle Missing Values ────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values intelligently:
    - Product_Search_Method → fill with MODE
      (most common value — more realistic than 'Unknown')
    - All others → fill with 'Unknown'
    """
    # Fill Product_Search_Method with mode
    if df["Product_Search_Method"].isnull().any():
        mode_val = df["Product_Search_Method"].mode()[0]
        df["Product_Search_Method"].fillna(mode_val, inplace=True)
        print(f"✅ Product_Search_Method filled with: '{mode_val}'")

    # Replace string 'Nan' artefacts from CSV
    df.replace("Nan", np.nan, inplace=True)

    # Fill remaining nulls
    df.fillna("Unknown", inplace=True)
    print("✅ Remaining missing values filled with 'Unknown'")
    return df


# ── STEP 7: Convert Numeric Columns ──────────────────────
def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert rating columns to proper numeric types.

    WHY: These scale ratings (1-5) may have been read
    as strings from CSV. We need them as numbers for
    statistical analysis and KMeans clustering.
    """
    numeric_cols = [
        "Customer_Reviews_Importance",
        "Shopping_Satisfaction",
        "Rating_Accuracy",
        "Rec_Freq_Numeric"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
    print("✅ Numeric columns converted")
    return df


# ── STEP 8: Save Cleaned Data ────────────────────────────
def save_cleaned_data(df: pd.DataFrame,
                      output_path: str) -> None:
    """Save the fully cleaned DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved → {output_path}")


# ── MASTER PIPELINE FUNCTION ─────────────────────────────
def run_cleaning_pipeline(raw_path: str,
                          output_path: str) -> pd.DataFrame:
    """
    Runs ALL cleaning steps in the correct order.
    This is the only function called from main.py.

    Args:
        raw_path    : path to raw Amazon.csv
        output_path : where to save Amazon_cleaned.csv

    Returns:
        Fully cleaned DataFrame ready for analysis
    """
    print("\n" + "="*50)
    print("  TASK 1: DATA CLEANING PIPELINE")
    print("="*50)

    df = load_data(raw_path)
    df = fix_column_names(df)
    df = rename_duplicate_columns(df)
    df = remove_duplicates(df)
    df = standardize_categoricals(df)
    df = handle_missing_values(df)
    df = convert_numeric_columns(df)
    save_cleaned_data(df, output_path)

    print(f"\n✅ Cleaning complete! Final shape: {df.shape}")
    return df