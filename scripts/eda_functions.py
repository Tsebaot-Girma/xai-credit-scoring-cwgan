# scripts/eda_functions.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set global style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

# ----------------------------------------------------------------------
# 1. Data Loading & Basic Cleaning
# ----------------------------------------------------------------------
def load_data(file_path, clean_cols=True, drop_unnamed=True):
    """Load CSV, optionally clean column names and drop unnamed columns."""
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
    cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols

def get_categorical_features(df, exclude=None):
    cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols

def get_feature_types(df, target=None):
    num = get_numerical_features(df, exclude=[target] if target else None)
    cat = get_categorical_features(df, exclude=[target] if target else None)
    return num, cat

# ----------------------------------------------------------------------
# 4. Target Variable Processing
# ----------------------------------------------------------------------
def map_target(df, target_col, mapping):
    df[target_col] = df[target_col].map(mapping)
    return df

def plot_target_distribution(df, target_col, save_path=None):
    """Enhanced target distribution with count plot."""
    class_counts = df[target_col].value_counts()
    print("===== TARGET DISTRIBUTION =====")
    print(class_counts)
    print(f"\nImbalance Ratio: {class_counts[0] / class_counts[1]:.2f}:1")
    print(f"Default Rate: {class_counts[1] / len(df) * 100:.2f}%")
    
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=df[target_col], hue=df[target_col],
                       palette={0: 'green', 1: 'red'}, legend=False)
    ax.set_title('Class Distribution')
    ax.set_xlabel('Class (0 = Non-Default, 1 = Default)')
    ax.set_ylabel('Count')
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height() + 50,
                f'{int(p.get_height())}\n({p.get_height()/len(df)*100:.1f}%)',
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ----------------------------------------------------------------------
# 5. Univariate Analysis
# ----------------------------------------------------------------------
def plot_numerical(df, cols=None, kde=True, bins=30):
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
    if cat_cols is None:
        cat_cols = get_categorical_features(df, exclude=[target])
    for col in cat_cols:
        plt.figure()
        sns.countplot(x=col, hue=target, data=df)
        plt.title(f"{col} vs {target}")
        plt.xticks(rotation=45)
        plt.show()

def plot_num_vs_target(df, num_cols=None, target='Class'):
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
    if cols is None:
        cols = get_numerical_features(df)
    for col in cols:
        plt.figure()
        sns.boxplot(y=df[col])
        plt.title(f"Boxplot of {col}")
        plt.show()

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] < lower) | (df[col] > upper)]

def outlier_summary(df, cols=None):
    if cols is None:
        cols = get_numerical_features(df)
    for col in cols:
        outliers = detect_outliers_iqr(df, col)
        print(f"{col}: {len(outliers)} outliers")

# ======================================================================
# 10. NEW: Default Rate by Demographics
# ======================================================================
def plot_default_rate_demographics(df, target_col='Class', save_path=None):
    """Bar charts showing default rate by gender, education, and marital status."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Gender
    if 'SEX' in df.columns:
        default_by_sex = df.groupby('SEX')[target_col].mean() * 100
        gender_labels = ['Male', 'Female'] if len(default_by_sex) == 2 else [f'G{i}' for i in default_by_sex.index]
        axes[0].bar(gender_labels, default_by_sex.values, color=['#3498db', '#e74c3c'], edgecolor='black')
        axes[0].set_title('Default Rate by Gender', fontsize=12)
        axes[0].set_ylabel('Default Rate (%)')
        axes[0].set_ylim(0, max(default_by_sex.values) * 1.3)
        for i, v in enumerate(default_by_sex.values):
            axes[0].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
    else:
        axes[0].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Default Rate by Gender')
    
    # Education
    if 'EDUCATION' in df.columns:
        default_by_edu = df.groupby('EDUCATION')[target_col].mean() * 100
        edu_labels = [f'Lvl {i}' for i in default_by_edu.index]
        axes[1].bar(edu_labels, default_by_edu.values, color='#f39c12', edgecolor='black')
        axes[1].set_title('Default Rate by Education', fontsize=12)
        axes[1].set_ylabel('Default Rate (%)')
        axes[1].set_ylim(0, max(default_by_edu.values) * 1.3)
        axes[1].tick_params(axis='x', rotation=30)
        for i, v in enumerate(default_by_edu.values):
            axes[1].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Default Rate by Education')
    
    # Marital Status
    if 'MARRIAGE' in df.columns:
        default_by_marriage = df.groupby('MARRIAGE')[target_col].mean() * 100
        marriage_labels = [f'St {i}' for i in default_by_marriage.index]
        axes[2].bar(marriage_labels, default_by_marriage.values, color='#9b59b6', edgecolor='black')
        axes[2].set_title('Default Rate by Marital Status', fontsize=12)
        axes[2].set_ylabel('Default Rate (%)')
        axes[2].set_ylim(0, max(default_by_marriage.values) * 1.3)
        axes[2].tick_params(axis='x', rotation=30)
        for i, v in enumerate(default_by_marriage.values):
            axes[2].text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)
    else:
        axes[2].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Default Rate by Marital Status')
    
    plt.suptitle('Default Rate by Demographic Features', fontsize=14, y=1.03)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n===== DEMOGRAPHIC DEFAULT RATES =====")
    if 'SEX' in df.columns:
        print(f"Gender:\n{default_by_sex}\n")
    if 'EDUCATION' in df.columns:
        print(f"Education:\n{default_by_edu}\n")
    if 'MARRIAGE' in df.columns:
        print(f"Marital Status:\n{default_by_marriage}")

