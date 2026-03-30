# scripts/eda_functions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

# ----------------------------------------------------------------------
# 1. Data Loading & Basic Cleaning
# ----------------------------------------------------------------------
def load_data(file_path, clean_cols=True, drop_unnamed=True):
    """
    Load CSV, optionally clean column names and drop unnamed columns.
    Returns the DataFrame.
    """
    try:
        df = pd.read_csv(file_path)

        if drop_unnamed:
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if clean_cols:
            df.columns = df.columns.str.strip()

        return df
    except Exception as e:
        raise Exception(f"Error loading file: {e}")

# ----------------------------------------------------------------------
# 2. Dataset Overview
# ----------------------------------------------------------------------
def overview(df):
    """Print a comprehensive dataset overview."""
    print("===== DATASET OVERVIEW =====")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    print("Data Types:\n", df.dtypes, "\n")
    print("Missing Values:\n", df.isnull().sum(), "\n")
    print("Summary Statistics:\n", df.describe(include='all'), "\n")
    print("First 5 Rows:\n", df.head())

# ----------------------------------------------------------------------
# 3. Feature Type Detection
# ----------------------------------------------------------------------
def get_numerical_features(df, exclude=None):
    """Return list of numerical columns (int64, float64). Optionally exclude some."""
    cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols

def get_categorical_features(df, exclude=None):
    """Return list of categorical columns (object, category). Optionally exclude some."""
    cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols

def get_feature_types(df, target=None):
    """Return numerical and categorical feature lists, optionally excluding target."""
    num = get_numerical_features(df, exclude=[target] if target else None)
    cat = get_categorical_features(df, exclude=[target] if target else None)
    return num, cat

# ----------------------------------------------------------------------
# 4. Target Variable Processing
# ----------------------------------------------------------------------
def map_target(df, target_col, mapping):
    """
    Map target values according to a dictionary.
    Returns the modified DataFrame.
    """
    df[target_col] = df[target_col].map(mapping)
    return df

def plot_target_distribution(df, target_col, labels=None, palette=None):
    """
    Plot distribution of target variable.
    - target_col: column name
    - labels: list of labels for the categories (e.g., ['Good', 'Bad'])
    - palette: dictionary mapping original values to colors
    """
    data = df[target_col]
    plt.figure(figsize=(6, 4))
    # Use hue with the same data so palette is applied correctly
    ax = sns.countplot(x=data, hue=data, palette=palette, legend=False)
    plt.title(f"Distribution of {target_col}")
    plt.xlabel(target_col)
    plt.ylabel("Count")

    # Set custom labels if provided
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

    # Add counts and percentages
    total = len(data)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.02*total,
                f'{height}\n({height/total:.1%})', ha='center', fontweight='bold')
    plt.show()

# ----------------------------------------------------------------------
# 5. Univariate Analysis
# ----------------------------------------------------------------------
def plot_numerical(df, cols=None, kde=True, bins=30):
    """Plot histograms with optional KDE for numerical columns."""
    if cols is None:
        cols = get_numerical_features(df)

    for col in cols:
        plt.figure()
        sns.histplot(df[col], kde=kde, bins=bins)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

def plot_categorical(df, cols=None, max_categories=10, horizontal=True):
    """Plot bar charts for categorical columns, limiting to top categories if needed."""
    if cols is None:
        cols = get_categorical_features(df)

    for col in cols:
        counts = df[col].value_counts()
        if len(counts) > max_categories:
            top = counts.nlargest(max_categories).index
            data = df[df[col].isin(top)][col]
            title = f"{col} (Top {max_categories})"
        else:
            data = df[col]
            title = f"Distribution of {col}"

        plt.figure()
        if horizontal:
            sns.countplot(y=data, order=data.value_counts().index)
            plt.title(title)
            plt.xlabel("Count")
            plt.ylabel(col)
        else:
            sns.countplot(x=data, order=data.value_counts().index)
            plt.title(title)
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
        plt.show()

# ----------------------------------------------------------------------
# 6. Bivariate Analysis
# ----------------------------------------------------------------------
def plot_cat_vs_target(df, cat_cols=None, target='Class'):
    """Stacked count plots for categorical features vs target."""
    if cat_cols is None:
        cat_cols = get_categorical_features(df, exclude=[target])

    for col in cat_cols:
        plt.figure()
        sns.countplot(x=col, hue=target, data=df)
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.show()

def plot_num_vs_target(df, num_cols=None, target='Class'):
    """Boxplots for numerical features vs target."""
    if num_cols is None:
        num_cols = get_numerical_features(df, exclude=[target])

    for col in num_cols:
        plt.figure()
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f"{col} vs {target}")
        plt.show()

# ----------------------------------------------------------------------
# 7. Correlation Analysis
# ----------------------------------------------------------------------
def correlation_heatmap(df, cols=None, method='pearson', annot=True, cmap='coolwarm'):
    """Plot correlation matrix for numerical columns."""
    if cols is None:
        cols = get_numerical_features(df)

    if len(cols) < 2:
        print("Need at least two numerical columns.")
        return None

    corr = df[cols].corr(method=method)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=annot, cmap=cmap, linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()
    return corr

# ----------------------------------------------------------------------
# 8. Missing Values
# ----------------------------------------------------------------------
def missing_values_report(df):
    """Print columns with missing values and return a Series."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values.")
    else:
        print("Missing values per column:\n", missing)
    return missing

# ----------------------------------------------------------------------
# 9. Outlier Detection
# ----------------------------------------------------------------------
def plot_outliers_boxplot(df, cols=None):
    """Boxplots for outlier visualization."""
    if cols is None:
        cols = get_numerical_features(df)

    for col in cols:
        plt.figure()
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

def detect_outliers_iqr(df, col):
    """Return rows that are outliers for a column using IQR method."""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)]

def outlier_summary(df, cols=None):
    """Print number of outliers for each numerical column."""
    if cols is None:
        cols = get_numerical_features(df)

    for col in cols:
        outliers = detect_outliers_iqr(df, col)
        print(f"{col}: {len(outliers)} outliers")

# ----------------------------------------------------------------------
# 10. Feature Engineering Helpers
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

# ----------------------------------------------------------------------
# 11. Encoding
# ----------------------------------------------------------------------
def encode_categorical(df, cols=None, drop_first=True, method='onehot'):
    """
    Encode categorical columns.
    method: 'onehot' (one-hot) or 'label' (label encoding - not recommended for tree models)
    """
    if cols is None:
        cols = get_categorical_features(df)

    if method == 'onehot':
        return pd.get_dummies(df, columns=cols, drop_first=drop_first)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        df_encoded = df.copy()
        for col in cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        return df_encoded
    else:
        raise ValueError("method must be 'onehot' or 'label'")

# ----------------------------------------------------------------------
# 12. Full EDA Pipeline
# ----------------------------------------------------------------------
def run_full_eda(df, target='Class', target_mapping=None, 
                 plot_cat_max=10, plot_cat_horizontal=True):
    """
    Execute a complete EDA workflow.
    """
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    overview(df)

    # Target analysis
    if target in df.columns:
        print("\n" + "="*50)
        print("TARGET ANALYSIS")
        print("="*50)
        plot_target_distribution(df, target, mapping=target_mapping)

    # Feature types
    num, cat = get_feature_types(df, target)
    print("\nNumerical features:", num)
    print("Categorical features:", cat)

    # Univariate
    print("\n" + "="*50)
    print("NUMERICAL DISTRIBUTIONS")
    print("="*50)
    plot_numerical(df, num)

    print("\n" + "="*50)
    print("CATEGORICAL DISTRIBUTIONS")
    print("="*50)
    plot_categorical(df, cat, max_categories=plot_cat_max, horizontal=plot_cat_horizontal)

    # Bivariate
    if target in df.columns:
        print("\n" + "="*50)
        print("CATEGORICAL vs TARGET")
        print("="*50)
        plot_cat_vs_target(df, cat, target)

        print("\n" + "="*50)
        print("NUMERICAL vs TARGET")
        print("="*50)
        plot_num_vs_target(df, num, target)

    # Correlation
    print("\n" + "="*50)
    print("CORRELATION MATRIX")
    print("="*50)
    correlation_heatmap(df, num)

    # Missing values
    print("\n" + "="*50)
    print("MISSING VALUES")
    print("="*50)
    missing_values_report(df)

    # Outliers
    print("\n" + "="*50)
    print("OUTLIER DETECTION (Boxplots)")
    print("="*50)
    plot_outliers_boxplot(df, num)

    print("\n" + "="*50)
    print("OUTLIER SUMMARY (IQR)")
    print("="*50)
    outlier_summary(df, num)