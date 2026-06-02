#scripts/train_xgboost.py

"""
XGBoost Training Module
- Default training for Imbalanced and SMOTE baselines
- Optuna hyperparameter optimization for cWGAN‑GP model
"""

import xgboost as xgb
import optuna
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def train_xgboost_default(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          scale_pos_weight: bool = True) -> xgb.XGBClassifier:
    """
    Train XGBoost with default parameters.
    Used for Imbalanced baseline and SMOTE baseline.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (for early stopping)
        y_test: Test labels (for early stopping)
        scale_pos_weight: Whether to automatically compute scale_pos_weight

    Returns:
        Trained XGBoost model
    """
    # Compute class weight for imbalance handling
    if scale_pos_weight:
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        spw = n_neg / n_pos if n_pos > 0 else 1
    else:
        spw = 1

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"  Default XGBoost trained (scale_pos_weight={spw:.2f})")
    return model

def train_xgboost_optuna(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         n_trials: int = 50, cv_folds: int = 5,
                         random_state: int = 42) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train XGBoost with Optuna hyperparameter optimization using k‑fold CV.
    Used for the proposed cWGAN‑GP model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_trials: Number of Optuna trials (50 recommended)
        cv_folds: Number of cross‑validation folds (5 recommended)
        random_state: Random seed

    Returns:
        best_model, best_params
    """
    from sklearn.model_selection import StratifiedKFold
    
    print(f"\n  Starting Optuna hyperparameter optimization")
    print(f"  Trials: {n_trials}, CV folds: {cv_folds}")

    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )

    # Define objective with k‑fold CV
    def objective_with_cv(trial: optuna.Trial) -> float:
        """Optuna objective that uses k‑fold CV for robust evaluation."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 0
        }

        # K‑fold cross‑validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        auc_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            auc_scores.append(auc)

        return np.mean(auc_scores)

    # Optimize
    study.optimize(
        objective_with_cv,
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n  Best CV AUC-ROC: {study.best_value:.4f}")
    print("  Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # Train final model on full training set with best parameters
    best_params = study.best_params
    best_params['random_state'] = random_state
    best_params['eval_metric'] = 'logloss'
    best_params['use_label_encoder'] = False
    best_params['verbosity'] = 0

    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    return best_model, best_params