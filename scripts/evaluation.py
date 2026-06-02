# scripts/evaluation.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, fbeta_score
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import friedmanchisquare, wilcoxon
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'


# ======================================================================
# Part 1: Generative Performance Evaluation (cWGAN‑GP)
# ======================================================================

def evaluate_generative_quality(df_synth_vis, feature_info, df_processed, 
                                target_col='Class', minority_label=1,
                                save_path='../figures'):
    """
    Complete generative evaluation of the cWGAN‑GP using pre‑generated synthetic samples.
    """
    
    num_cols = feature_info['numerical']
    df_minority = df_processed[df_processed[target_col] == minority_label]
    
    # =====================================================================
    # 0. Numerical Range Comparison
    # =====================================================================
    print("\n" + "="*90)
    print("NUMERICAL FEATURE RANGE COMPARISON")
    print("="*90)
    
    # Full dataset
    print(f"\n  {'FULL DATASET (All Applicants)':^80}")
    print(f"  {'Feature':<45s} {'Min':>15s} {'Max':>15s}")
    print("  " + "-"*75)
    for col in num_cols:
        print(f"  {col:<45s} {df_processed[col].min():>15.2f} {df_processed[col].max():>15.2f}")
    
    # Real minority
    print(f"\n  {'REAL MINORITY CLASS (Defaulters)':^80}")
    print(f"  {'Feature':<45s} {'Min':>15s} {'Max':>15s}")
    print("  " + "-"*75)
    for col in num_cols:
        print(f"  {col:<45s} {df_minority[col].min():>15.2f} {df_minority[col].max():>15.2f}")
    
    # Synthetic minority
    print(f"\n  {'SYNTHETIC MINORITY (Generated)':^80}")
    print(f"  {'Feature':<45s} {'Min':>15s} {'Max':>15s}")
    print("  " + "-"*75)
    for col in num_cols:
        print(f"  {col:<45s} {df_synth_vis[col].min():>15.2f} {df_synth_vis[col].max():>15.2f}")

    # =====================================================================
    # 1. Wasserstein distances
    # =====================================================================
    print("\n" + "="*70)
    print("WASSERSTEIN DISTANCES (Numerical Features)")
    print("="*70)
    print(f"  {'Feature':<45s} {'Distance':>12s}")
    print("  " + "-"*57)
    w_distances = {}
    for col in num_cols:
        real_vals = df_processed[df_processed[target_col] == minority_label][col].values
        synth_vals = df_synth_vis[col].values
        w_dist = wasserstein_distance(real_vals, synth_vals)
        w_distances[col] = w_dist
        print(f"  {col:<45s} {w_dist:>12.4f}")

    # =====================================================================
    # 2. Statistical Similarity Metrics
    # =====================================================================
    print("\n" + "="*110)
    print("STATISTICAL SIMILARITY METRICS")
    print("="*110)
    print(f"  {'Feature':<45s} {'KS Stat':>10s} {'KS P-val':>14s} {'W-Dist':>10s} {'Mean Diff':>12s} {'Diff %':>8s}")
    print("  " + "-"*105)
    
    for col in num_cols:
        real_vals = df_processed[df_processed[target_col] == minority_label][col].values
        synth_vals = df_synth_vis[col].values
        
        ks_stat, ks_pval = ks_2samp(real_vals, synth_vals)
        w_dist = wasserstein_distance(real_vals, synth_vals)
        mean_diff = real_vals.mean() - synth_vals.mean()
        mean_diff_pct = (abs(mean_diff) / real_vals.mean()) * 100 if real_vals.mean() != 0 else 0
        
        print(f"  {col:<45s} {ks_stat:>10.4f} {ks_pval:>14.4e} {w_dist:>10.4f} {mean_diff:>12.4f} {mean_diff_pct:>7.2f}%")

    # =====================================================================
    # 3. KDE plots
    # =====================================================================
    print("\nPlotting numerical distributions with mean lines...")
    n_cols_plot = min(3, len(num_cols))
    n_rows_plot = (len(num_cols) + n_cols_plot - 1) // n_cols_plot
    
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6*n_cols_plot, 5*n_rows_plot))
    if n_rows_plot == 1 and n_cols_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        ax = axes[i]
        real_vals = df_processed[df_processed[target_col] == minority_label][col]
        synth_vals = df_synth_vis[col]
        
        sns.kdeplot(real_vals, label='Real Minority', fill=True, alpha=0.5, ax=ax)
        sns.kdeplot(synth_vals, label='Synthetic', fill=True, alpha=0.5, ax=ax)
        
        real_mean = real_vals.mean()
        synth_mean = synth_vals.mean()
        ax.axvline(real_mean, linestyle='--', linewidth=2, label=f'Real Mean: {real_mean:.2f}')
        ax.axvline(synth_mean, color='orange', linestyle='--', linewidth=2, label=f'Synth Mean: {synth_mean:.2f}')
        
        ax.set_title(f'{col}', fontsize=10)
        ax.legend(fontsize=7)
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Numerical Distributions: Real vs Synthetic Minority', fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/cwgan_numerical_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================================================
    # 4. Categorical proportions
    # =====================================================================
    if feature_info.get('categorical'):
        print("Plotting categorical proportions with value labels...")
        def reverse_onehot(df, prefix):
            cols = [c for c in df.columns if c.startswith(prefix + '_')]
            if not cols:
                return None
            cat_idx = df[cols].values.argmax(axis=1)
            mapping = {i: col.split('_')[-1] for i, col in enumerate(cols)}
            return pd.Series(cat_idx).map(mapping)

        for orig_cat in feature_info['categorical'].keys():
            real_cat = reverse_onehot(df_processed[df_processed[target_col] == minority_label], orig_cat)
            synth_cat = reverse_onehot(df_synth_vis, orig_cat)
            if real_cat is not None:
                plt.figure(figsize=(10, 5))
                compare_df = pd.DataFrame({
                    'Real': real_cat.value_counts(normalize=True),
                    'Synthetic': synth_cat.value_counts(normalize=True)
                }).fillna(0)
                
                ax = compare_df.plot(kind='bar', width=0.75, alpha=0.85, ax=plt.gca())
                plt.title(f'{orig_cat} Proportions: Real vs Synthetic', fontsize=14)
                plt.ylabel('Proportion')
                plt.xticks(rotation=45)
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', fontsize=8, padding=2)
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/cwgan_{orig_cat}_proportions.png', dpi=150, bbox_inches='tight')
                plt.show()

    # =====================================================================
    # 5. Correlation matrices
    # =====================================================================
    print("Plotting correlation matrices...")
    real_min = df_processed[df_processed[target_col] == minority_label][num_cols]
    corr_real = real_min.corr()
    corr_synth = df_synth_vis[num_cols].corr()
    diff = corr_real - corr_synth
    
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))
    
    sns.heatmap(corr_real, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=axes[0])
    axes[0].set_title('Correlation Matrix - Real Minority', fontsize=12)
    
    sns.heatmap(corr_synth, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=axes[1])
    axes[1].set_title('Correlation Matrix - Synthetic', fontsize=12)
    
    sns.heatmap(diff, annot=True, cmap='RdBu', center=0, ax=axes[2])
    axes[2].set_title('Correlation Difference (Real - Synthetic)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/cwgan_correlation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # =====================================================================
    # 6. PCA / t‑SNE - Side by Side (Compact)
    # =====================================================================
    print("Generating PCA/t‑SNE projections...")

    # Prepare data
    X_min = np.vstack([df_processed[df_processed[target_col] == minority_label][num_cols].values,
                    df_synth_vis[num_cols].values])
    labels_vis = np.array(['Real'] * (df_processed[target_col] == minority_label).sum() + 
                        ['Synthetic'] * len(df_synth_vis))

    # Compute projections
    pca = PCA(n_components=2, random_state=42).fit_transform(X_min)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_min)

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot both
    for ax, proj, title in [(ax1, pca, 'PCA'), (ax2, tsne, 't-SNE')]:
        sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=labels_vis, 
                    alpha=0.6, ax=ax, palette=['#1f77b4', '#ff7f0e'], s=25)
        ax.set_title(f'{title} Projection', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'{title} 1')
        ax.set_ylabel(f'{title} 2')
        ax.legend(title='Type', loc='best', fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(f'{save_path}', exist_ok=True)
    plt.savefig(f'{save_path}/cwgan_pca_tsne_combined.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Visualization complete! Saved to {save_path}/cwgan_pca_tsne_combined.png")
    # =====================================================================
    # 7. Comprehensive Quality Report
    # =====================================================================
    print("\n" + "="*80)
    print("SYNTHETIC DATA QUALITY REPORT")
    print("="*80)
    
    print("\n1. NUMERICAL FEATURE STATISTICS:")
    print(f"  {'Feature':<45s} {'Real Mean':>10s} {'Synth Mean':>10s} {'Mean Diff%':>10s} {'Real Std':>10s} {'Synth Std':>10s} {'Std Diff%':>10s}")
    print("  " + "-"*105)
    
    mean_scores = []
    for col in num_cols:
        real_vals = df_processed[df_processed[target_col] == minority_label][col]
        synth_vals = df_synth_vis[col]
        
        real_mean = real_vals.mean()
        synth_mean = synth_vals.mean()
        mean_diff_pct = (abs(real_mean - synth_mean) / real_mean) * 100 if real_mean != 0 else 0
        real_std = real_vals.std()
        synth_std = synth_vals.std()
        std_diff_pct = (abs(real_std - synth_std) / real_std) * 100 if real_std != 0 else 0
        
        mean_score = max(0, 100 - mean_diff_pct)
        mean_scores.append(mean_score)
        
        print(f"  {col:<45s} {real_mean:>10.2f} {synth_mean:>10.2f} {mean_diff_pct:>9.1f}% {real_std:>10.2f} {synth_std:>10.2f} {std_diff_pct:>9.1f}%")
    
    # 7.2 Categorical
    if feature_info.get('categorical'):
        print("\n2. CATEGORICAL FEATURE FIDELITY:")
        top_match_count = 0
        total_cats = 0
        for orig_cat in feature_info['categorical'].keys():
            real_cat = reverse_onehot(df_processed[df_processed[target_col] == minority_label], orig_cat)
            synth_cat = reverse_onehot(df_synth_vis, orig_cat)
            if real_cat is not None:
                total_cats += 1
                real_top = real_cat.value_counts(normalize=True).index[0]
                synth_top = synth_cat.value_counts(normalize=True).index[0]
                real_top_pct = real_cat.value_counts(normalize=True).iloc[0] * 100
                synth_top_pct = synth_cat.value_counts(normalize=True).iloc[0] * 100
                match = "✓" if real_top == synth_top else "✗"
                if real_top == synth_top:
                    top_match_count += 1
                print(f"  {orig_cat:30s}: Top match {match} | Real: {real_top} ({real_top_pct:.1f}%) | Synth: {synth_top} ({synth_top_pct:.1f}%)")
        cat_fidelity = (top_match_count / total_cats) * 100 if total_cats > 0 else 100
    else:
        cat_fidelity = 100
        print("\n2. CATEGORICAL FEATURE FIDELITY: No categorical features")

    # 7.3 Correlation
    print("\n3. CORRELATION PRESERVATION:")
    corr_diff_abs = np.abs(corr_real.values - corr_synth.values)
    avg_corr_diff = np.mean(corr_diff_abs)
    max_corr_diff = np.max(corr_diff_abs)
    print(f"  Average absolute correlation difference: {avg_corr_diff:.4f}")
    print(f"  Maximum correlation difference: {max_corr_diff:.4f}")
    
    # 7.4 Overall
    print("\n4. OVERALL QUALITY SCORE:")
    mean_preservation = np.mean(mean_scores)
    corr_preservation = max(0, 100 - avg_corr_diff * 100)
    
    dist_scores = []
    for col in num_cols:
        real_vals = df_processed[df_processed[target_col] == minority_label][col].values
        synth_vals = df_synth_vis[col].values
        w_dist = wasserstein_distance(real_vals, synth_vals)
        feature_range = real_vals.max() - real_vals.min()
        if feature_range > 0:
            normalized_dist = (w_dist / feature_range) * 100
            dist_scores.append(max(0, 100 - normalized_dist))
        else:
            dist_scores.append(100)
    dist_similarity = np.mean(dist_scores)
    
    overall_score = (mean_preservation * 0.3 + corr_preservation * 0.25 + 
                     dist_similarity * 0.25 + cat_fidelity * 0.2)
    
    print(f"  Mean preservation score:     {mean_preservation:.1f}/100")
    print(f"  Correlation preservation:     {corr_preservation:.1f}/100")
    print(f"  Distribution similarity:      {dist_similarity:.1f}/100")
    print(f"  Categorical fidelity:         {cat_fidelity:.1f}/100")
    print(f"  OVERALL QUALITY SCORE:        {overall_score:.1f}/100")
    
    if overall_score >= 80:
        print("  ✅ Excellent quality - Synthetic data closely matches real data")
    elif overall_score >= 60:
        print("  ⚠️ Good quality - Some minor discrepancies")
    else:
        print("  ❌ Poor quality - Consider retraining with different parameters")

    print("\n✅ Generative evaluation complete.")
    
    return {
        'wasserstein_distances': w_distances,
        'overall_score': overall_score,
        'mean_preservation': mean_preservation,
        'corr_preservation': corr_preservation,
        'dist_similarity': dist_similarity,
        'cat_fidelity': cat_fidelity
    }


# ======================================================================
# Part 2: Predictive Performance Evaluation (XGBoost) - UPDATED
# ======================================================================

# ============================================================
# Unified classifier evaluation
# ============================================================
def evaluate_classifier(model, X_test, y_test, model_name, threshold=None):
    """
    Compute metrics, confusion matrix, and ROC curve.
    """

    y_proba = model.predict_proba(X_test)[:, 1]

    # =====================================================
    # Threshold handling
    # =====================================================
    if threshold is None:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)

    # =====================================================
    # Metrics
    # =====================================================
    metrics = {
        'Model': model_name,
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred)
    }

    print("\n" + "="*60)
    print(f"{model_name} Performance")
    print("="*60)

    print(f"Threshold : {threshold:.4f}")

    for k, v in metrics.items():
        if k != 'Model':
            print(f"{k:<12}: {v:.4f}")

    # =====================================================
    # Confusion Matrix
    # =====================================================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

    # =====================================================
    # ROC Curve
    # =====================================================
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(7,5))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f'{model_name} (AUC = {metrics["AUC-ROC"]:.4f})'
    )

    plt.plot([0,1], [0,1], 'k--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title(f'ROC Curve - {model_name}')

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return metrics, y_proba, y_pred

# ============================================================
# Model comparison (FIXED BUG: correct y_test handling)
# ============================================================
def compare_all_models(models_dict, save_path='../figures/model_comparison_xgb.png'):
    """Compare multiple models with optional threshold tuning."""
    
    all_metrics = []
    all_probas = {}
    all_y_tests = {}  # Store y_test for each model
    
    # =========================
    # Evaluate models
    # =========================
    for name, model_tuple in models_dict.items():
        
        if len(model_tuple) == 4:
            model, X_test, y_test, threshold = model_tuple
        else:
            model, X_test, y_test = model_tuple
            threshold = None
        
        metrics, y_proba, _ = evaluate_classifier(
            model, X_test, y_test, name, threshold
        )
        
        all_metrics.append(metrics)
        all_probas[name] = y_proba
        all_y_tests[name] = y_test  # Store y_test for each model
    
    results_df = pd.DataFrame(all_metrics).set_index('Model')
    
    print("\n" + "="*60)
    print("FINAL COMPARISON TABLE")
    print("="*60)
    print(results_df.round(4).to_string())
    
    # =========================
    # Bar Plot
    # =========================
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison (XGBoost)')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # =========================
    # ROC Curve Comparison (FIXED: use each model's own y_test)
    # =========================
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (name, proba) in enumerate(all_probas.items()):
        y_test_model = all_y_tests[name]  # Get correct y_test for this model
        fpr, tpr, _ = roc_curve(y_test_model, proba)
        auc_val = roc_auc_score(y_test_model, proba)
        
        plt.plot(
            fpr, tpr,
            label=f'{name} (AUC = {auc_val:.4f})',
            linewidth=2,
            color=colors[i % len(colors)]
        )
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison (XGBoost)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/roc_curves_comparison_xgb.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    results_df.to_csv('../results/model_comparison_xgb.csv')
    
    return results_df

# ----------------------------------------------------------------------
# Part 3: SHAP Explainability Evaluation
# ----------------------------------------------------------------------  

def shap_evaluation(model, X_sample, feature_names=None, save_path='../figures/shap'):
    """
    Generate SHAP plots for a trained XGBoost model.
    - model: trained XGBoost classifier
    - X_sample: DataFrame or numpy array of test samples (200-500 recommended)
    """
    # Fix base_score if needed
    if hasattr(model, 'base_score') and isinstance(model.base_score, str):
        model.base_score = float(model.base_score.strip("[]"))
    # Patch internal booster config
    try:
        import json
        booster = model.get_booster()
        config_str = booster.save_config()
        config = json.loads(config_str)
        learner = config.get('learner', {})
        learner_model_param = learner.get('learner_model_param', {})
        if 'base_score' in learner_model_param:
            base_val = learner_model_param['base_score']
            if isinstance(base_val, str):
                base_val = float(base_val.strip("[]"))
            learner_model_param['base_score'] = str(base_val)
            booster.load_config(json.dumps(config))
    except Exception as e:
        print(f"Could not patch booster config: {e}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Bar plot
    shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'{save_path}_bar.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Beeswarm plot
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'{save_path}_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Waterfall plot for a high‑risk prediction
    proba = model.predict_proba(X_sample)[:, 1]
    high_risk_idx = np.argmax(proba)
    shap.waterfall_plot(shap.Explanation(values=shap_values[high_risk_idx],
                                         base_values=explainer.expected_value,
                                         data=X_sample.iloc[high_risk_idx],
                                         feature_names=feature_names))
    plt.title('SHAP Waterfall - High Risk Sample')
    plt.tight_layout()
    plt.savefig(f'{save_path}_waterfall.png', dpi=300)
    plt.show()

    return explainer, shap_values


# ======================================================================
# Part 4: Statistical Significance Testing
# ======================================================================

def nested_cv_evaluation(X, y, param_grid, inner_cv=5, outer_cv=10):
    """Nested cross‑validation for unbiased AUC estimates."""
    if hasattr(X, 'values'):
        X = X.values
    if hasattr(y, 'values'):
        y = y.values
    
    outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in outer_cv_splitter.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        gb = GradientBoostingClassifier(random_state=42)
        grid = GridSearchCV(gb, param_grid, cv=inner_cv_splitter, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_tr, y_tr)
        y_proba = grid.best_estimator_.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, y_proba))
    return np.mean(scores), np.std(scores), scores


def friedman_test(scores_list):
    """Perform Friedman test."""
    stat, p = friedmanchisquare(*scores_list)
    print(f"Friedman test statistic: {stat:.4f}, p-value: {p:.4f}")
    return stat, p


def wilcoxon_test(scores_a, scores_b):
    """Paired Wilcoxon signed‑rank test."""
    stat, p = wilcoxon(scores_a, scores_b)
    print(f"Wilcoxon test statistic: {stat:.4f}, p-value: {p:.4f}")
    return stat, p




