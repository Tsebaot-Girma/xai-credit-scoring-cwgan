# scripts/feature_engineering.py

import numpy as np
import pandas as pd

# -----------------------------
# 1. Handle Missing Values (Report Only)
# -----------------------------
def handle_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if missing.empty:
        print("✅ No missing values detected.")
        return None
    
    print("===== MISSING VALUES REPORT =====")
    print(f"{'Column':<40s} {'Missing':>10s}  {'Percent':>10s}")
    print("-" * 65)
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"{col:<40s} {count:>10,}  {pct:>9.1f}%")
    print("-" * 65)
    print(f"{'Total':<40s} {missing.sum():>10,}")
    
    return missing

# -----------------------------
# 2. Auto detect categorical
# -----------------------------
def auto_detect_categorical(df, num_cols, cat_cols, target_col='Class', unique_threshold=10):
    
    new_num_cols = []
    new_cat_cols = cat_cols.copy()
    for col in num_cols:
        if col == target_col:
            continue
        n_unique = df[col].nunique()
        if n_unique <= unique_threshold:
            print(f"Moving '{col}' to categorical (unique values: {n_unique} ≤ {unique_threshold})")
            new_cat_cols.append(col)
            # Convert to category dtype so one‑hot encoding will catch it
            df[col] = df[col].astype('category')
        else:
            new_num_cols.append(col)
    return new_num_cols, new_cat_cols

# ----------------------------------------------------------------------
# 3. Feature Engineering Helpers
# ----------------------------------------------------------------------
def create_ratio(df, num1, num2, new_col):
    """Add a new column = num1 / num2, handling division by zero."""
    df[new_col] = df[num1] / df[num2].replace(0, np.nan)
    return df

def bin_column(df, col, bins, labels=None, new_col=None):
    """
    Bin a numerical column into categories.
    bins: list of bin edges.
    labels: list of labels.
    new_col: name of new column; if None, use f"{col}_binned".
    """
    if new_col is None:
        new_col = f"{col}_binned"
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
    return df

# -----------------------------
# 4. Encode Categorical Variables
# -----------------------------
def encode_categorical(df):
    
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # One‑hot encode (drop_first=False to keep all categories)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    return df_encoded, categorical_cols