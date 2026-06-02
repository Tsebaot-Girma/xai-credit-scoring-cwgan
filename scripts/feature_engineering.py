import numpy as np
import pandas as pd


def handle_missing_values(df):
    """Print a missing-value report and return the missing counts."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values detected.")
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


def fit_missing_imputer(df, numerical_cols, categorical_cols):
    """Fit train-only median/mode imputers."""
    num_values = {}
    cat_values = {}
    for col in numerical_cols:
        num_values[col] = float(pd.to_numeric(df[col], errors="coerce").median())
    for col in categorical_cols:
        mode = df[col].mode(dropna=True)
        cat_values[col] = "missing" if mode.empty else mode.iloc[0]
    return {"numerical": num_values, "categorical": cat_values}


def apply_missing_imputer(df, imputer):
    df = df.copy()
    for col, value in imputer.get("numerical", {}).items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(value)
    for col, value in imputer.get("categorical", {}).items():
        df[col] = df[col].fillna(value)
    return df


def fit_outlier_bounds(df, numerical_cols, factor=1.5):
    bounds = {}
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - factor * iqr, q3 + factor * iqr)
    return bounds


def apply_outlier_clipping(df, bounds):
    df = df.copy()
    for col, (lower, upper) in bounds.items():
        df[col] = df[col].clip(lower, upper)
    return df


def auto_detect_categorical(df, num_cols, cat_cols, target_col="Class", unique_threshold=10):
    new_num_cols = []
    new_cat_cols = cat_cols.copy()
    for col in num_cols:
        if col == target_col:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique < unique_threshold:
            print(f"Moving '{col}' to categorical (unique values: {n_unique} < {unique_threshold})")
            new_cat_cols.append(col)
            df[col] = df[col].astype("category")
        else:
            new_num_cols.append(col)
    return new_num_cols, new_cat_cols


def create_ratio(df, num1, num2, new_col):
    df = df.copy()
    df[new_col] = df[num1] / df[num2].replace(0, np.nan)
    return df


def bin_column(df, col, bins, labels=None, new_col=None):
    df = df.copy()
    if new_col is None:
        new_col = f"{col}_binned"
    df[new_col] = pd.cut(df[col], bins=bins, labels=labels)
    return df


def encode_categorical(df, drop_first=False):
    """
    One-hot encode object/category columns.

    Use drop_first=False for GAN training because the generator outputs one
    complete categorical simplex per original categorical variable.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df_encoded, categorical_cols
