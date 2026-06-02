"""
Model Explainability Module using SHAP
Supports both TreeExplainer (XGBoost 1.x) and KernelExplainer (any version).
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Optional
import warnings
import json
warnings.filterwarnings('ignore')

# Set default plot styling
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'
})


def _fix_xgboost_base_score(model: Any) -> None:
    """
    Fix XGBoost base_score string bug in both the model attribute and internal booster config.
    """
    # Fix model attribute
    if hasattr(model, 'base_score') and isinstance(model.base_score, str):
        model.base_score = float(model.base_score.strip("[]"))

    # Patch internal booster configuration
    try:
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
            print("  Booster config patched successfully.")
    except Exception as e:
        print(f"  Could not patch booster config: {e}")


def shap_explain(model: Any, X_train: np.ndarray, X_test: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 max_display: int = 20,
                 save_path: Optional[str] = None,
                 n_samples: Optional[int] = None) -> tuple:  # Add n_samples parameter
    """
    Generate SHAP explanations for the model.
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of test samples to use. If None, uses min(500, X_test.shape[0])
    """
    print("\n" + "="*60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("="*60)

    # --- Try TreeExplainer first ---
    try:
        print("\nAttempting TreeExplainer...")
        _fix_xgboost_base_score(model)

        background_size = min(100, X_train.shape[0])
        background = shap.sample(X_train, background_size, random_state=42)

        explainer = shap.TreeExplainer(model, background)
        print("  TreeExplainer created successfully.")

    except Exception as e:
        print(f"  TreeExplainer failed: {e}")
        print("  Falling back to KernelExplainer...")

        background = shap.sample(X_train, min(100, X_train.shape[0]), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        print("  KernelExplainer created successfully.")

    # --- Compute SHAP values with configurable sample size ---
    if n_samples is None:
        test_sample_size = min(500, X_test.shape[0])  # Default: 500
    else:
        test_sample_size = min(n_samples, X_test.shape[0])  # User-specified, capped at available data
    
    X_test_sample = X_test[:test_sample_size]

    
    print(f"  Computing SHAP values for {test_sample_size} samples...")

    if isinstance(explainer, shap.KernelExplainer):
        shap_values_raw = explainer.shap_values(X_test_sample, nsamples=200)
    else:
        shap_values_raw = explainer.shap_values(X_test_sample)

    # Handle different SHAP output formats
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]  # Multi-class: take positive class
    else:
        shap_values = shap_values_raw

    # If still 3D (samples × features × classes), take class 1
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    print(f"  SHAP values computed. Shape: {shap_values.shape}")

    # --- Bar Plot ---
    print("\n  Generating SHAP bar plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample,
                      feature_names=feature_names,
                      max_display=max_display,
                      plot_type='bar',
                      show=False)
    ax.set_title('SHAP Feature Importance (Mean |SHAP Value|)', fontsize=14, pad=20)
    plt.tight_layout(pad=2.0)
    if save_path:
        plt.savefig(f'{save_path}_bar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

    # --- Beeswarm Plot ---
    print("  Generating SHAP beeswarm plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_sample,
                      feature_names=feature_names,
                      max_display=max_display,
                      show=False)
    ax.set_title('SHAP Beeswarm Plot (Feature Impact on Predictions)', fontsize=14, pad=20)
    plt.tight_layout(pad=2.0)
    if save_path:
        plt.savefig(f'{save_path}_beeswarm.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

    # --- Feature Importance Ranking ---
    if feature_names is not None:
        # Mean absolute SHAP across samples
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = sorted(zip(feature_names, mean_abs_shap),
                                   key=lambda x: x[1], reverse=True)

        print("\n  Top 10 Most Important Features (by mean |SHAP value|):")
        print("  " + "-"*50)
        for i, (feat, importance) in enumerate(feature_importance[:10], 1):
            print(f"  {i:2d}. {feat:40s}: {importance:.4f}")

    print("\n✅ SHAP analysis complete.")
    return shap_values, explainer


def shap_waterfall_plot(model: Any, X_sample: np.ndarray,
                        feature_names: List[str],
                        sample_idx: int = 0,
                        save_path: Optional[str] = None):
    """
    Generate SHAP waterfall plot for a single prediction.
    """
    print(f"\n  Generating SHAP waterfall plot for sample {sample_idx}...")

    # Compute SHAP values for the single sample
    try:
        explainer = shap.TreeExplainer(model)
        shap_values_all = explainer.shap_values(X_sample)
    except Exception:
        background = shap.sample(X_sample, min(50, len(X_sample)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values_all = explainer.shap_values(X_sample[sample_idx:sample_idx+1], nsamples=200)

    # Extract the correct slice for the requested sample
    if isinstance(shap_values_all, list):
        # Multi-class output — take positive class
        shap_vals = shap_values_all[1]
    else:
        shap_vals = shap_values_all

    # Handle 3D shape (samples × features × classes)
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    # Handle 2D shape (samples × features) — take the single sample
    if len(shap_vals.shape) == 2:
        shap_vals = shap_vals[0]  # Take first (only) sample

    # Get the base value
    if isinstance(explainer, shap.KernelExplainer):
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]
        elif isinstance(base_value, np.ndarray) and len(base_value.shape) > 0:
            base_value = float(base_value) if base_value.size == 1 else base_value[1]
    else:
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value) if base_value.size == 1 else base_value

    # Create and display the waterfall plot
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals,
            base_values=base_value,
            data=X_sample[sample_idx],
            feature_names=feature_names
        ),
        max_display=15,
        show=False
    )
    plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=14, pad=20)
    plt.tight_layout(pad=2.0)
    if save_path:
        plt.savefig(f'{save_path}_waterfall_sample_{sample_idx}.png', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()