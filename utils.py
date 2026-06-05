import pandas as pd
from config import COLUMN_MAPPINGS
 
# SMART COLUMN DETECTION
def detect_columns(df):

    mapping = {}

    for std_col, variations in COLUMN_MAPPINGS.items():

        for col in df.columns:

            col_clean = col.lower().replace("_", "").replace(" ", "")

            for v in variations:
                v_clean = v.lower().replace("_", "").replace(" ", "")

                if col_clean == v_clean:
                    mapping[std_col] = col

    return mapping

# RENAME COLUMNS TO STANDARD
def rename_columns(df, mapping):

    df = df.copy()

    for std_col, actual_col in mapping.items():
        df.rename(columns={actual_col: std_col}, inplace=True)

    return df


# CLEAN DATA
def clean_data(df):

    df = df.copy()

    df.columns = df.columns.str.strip()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    return df
# FULL PREPROCESS PIPELINE
def preprocess_data(df):

    mapping = detect_columns(df)

    df = rename_columns(df, mapping)

    df = clean_data(df)

    required = ["date", "product_name", "quantity_sold", "price", "stock_available"]

    missing = [col for col in required if col not in df.columns]

    return df, mapping, missing

# DATA SUMMARY 
def data_summary(df):

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns)
    }
