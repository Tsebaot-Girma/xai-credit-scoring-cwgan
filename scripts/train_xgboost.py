# scripts/train_xgboost.py

import xgboost as xgb
import optuna
import numpy as np

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split
)

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    fbeta_score,
    accuracy_score
)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# Default Baseline XGBoost (unchanged)
# ============================================================
def train_xgboost_default(
    X_train,
    y_train,
    scale_pos_weight=True
):

    if scale_pos_weight:
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        spw = n_neg / n_pos if n_pos > 0 else 1
    else:
        spw = 1

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print("\nDefault XGBoost trained")
    print(f"scale_pos_weight = {spw:.2f}")

    return model


# ============================================================
# Fβ threshold optimisation (no precision constraint)
# ============================================================
def find_best_threshold_fbeta(model, X, y, beta=2.0):
    """
    Find threshold that maximizes Fβ score.
    β > 1 gives more weight to recall (β=2 typical for imbalanced credit risk).
    """
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.30, 0.99, 160)
    best_score = -1.0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = fbeta_score(y, y_pred, beta=beta)
        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"\nBest threshold (F{beta}): {best_thresh:.4f} (score={best_score:.4f})")
    return best_thresh, best_score


# ============================================================
# Optuna with Fβ objective (inside CV) + expanded search space
# ============================================================
def train_xgboost_optuna(
    X_train,
    y_train,
    n_trials=100,         # increased from 50
    cv_folds=5,
    beta=2.0              # use F₂ score by default
):

    print("\n" + "=" * 60)
    print(f"Starting Optuna Optimisation (maximising F{beta} inside CV)")
    print("=" * 60)

    def objective(trial):
        # Expanded hyperparameter ranges
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 15),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),  # increased upper bound
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,
            'use_label_encoder': False
        }

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]

            model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            y_proba = model.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0.30, 0.99, 60)

            best_fold_score = -1.0
            for t in thresholds:
                y_pred = (y_proba >= t).astype(int)
                # Use Fβ inside CV (not F1)
                fold_score = fbeta_score(y_val, y_pred, beta=beta)
                if fold_score > best_fold_score:
                    best_fold_score = fold_score

            fold_scores.append(best_fold_score)

        return np.mean(fold_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("OPTUNA RESULTS")
    print("=" * 60)

    print(f"Best CV F{beta} Score : {study.best_value:.4f}")

    print("\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    # Final train / validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # Final model with best parameters
    best_model = xgb.XGBClassifier(
        **study.best_params,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )

    best_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Threshold optimisation on validation set using Fβ
    best_threshold, best_fbeta = find_best_threshold_fbeta(best_model, X_val, y_val, beta=beta)

    print("\nFinal Validation Results")
    print(f"Best Threshold : {best_threshold:.4f}")
    print(f"Validation F{beta} : {best_fbeta:.4f}")

    return best_model, study.best_params, best_threshold

# Add this modified version to your train_xgboost.py file

def train_xgboost_optuna_with_recall_constraint(
    X_train,
    y_train,
    n_trials=100,
    cv_folds=5,
    beta=2.0,
    recall_min=0.70  # minimum recall constraint
):
    """
    Optuna optimization with recall constraint.
    Maximizes (recall + accuracy) / 2 while ensuring recall >= recall_min
    """
    
    print("\n" + "=" * 60)
    print(f"Starting Optuna Optimisation with recall >= {recall_min}")
    print("=" * 60)
    
    def objective(trial):
        # Expanded hyperparameter ranges
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 15),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,
            'use_label_encoder': False
        }
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]
            
            model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            y_proba = model.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0.30, 0.99, 60)
            
            best_fold_score = -1.0
            for t in thresholds:
                y_pred = (y_proba >= t).astype(int)
                recall = recall_score(y_val, y_pred)
                
                # Only consider thresholds meeting recall constraint
                if recall >= recall_min:
                    # Maximize (recall + accuracy) / 2
                    acc = accuracy_score(y_val, y_pred)
                    score = (recall + acc) / 2
                    if score > best_fold_score:
                        best_fold_score = score
            
            # If no threshold meets recall constraint, penalize heavily
            if best_fold_score == -1.0:
                best_fold_score = 0.0
                
            fold_scores.append(best_fold_score)
        
        return np.mean(fold_scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "=" * 60)
    print("OPTUNA RESULTS")
    print("=" * 60)
    
    print(f"Best CV Score : {study.best_value:.4f}")
    
    print("\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    
    # Final train / validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )
    
    # Final model with best parameters
    best_model = xgb.XGBClassifier(
        **study.best_params,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )
    
    best_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Find threshold that meets recall constraint while maximizing (recall+acc)/2
    y_proba = best_model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.30, 0.99, 160)
    
    best_threshold = 0.5
    best_score = -1.0
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        recall = recall_score(y_val, y_pred)
        if recall >= recall_min:
            acc = accuracy_score(y_val, y_pred)
            score = (recall + acc) / 2
            if score > best_score:
                best_score = score
                best_threshold = t
    
    print(f"\nFinal Validation Results")
    print(f"Best Threshold (recall >= {recall_min}): {best_threshold:.4f}")
    print(f"Validation Score (recall+acc)/2: {best_score:.4f}")
    
    # Verify recall constraint
    y_pred_final = (y_proba >= best_threshold).astype(int)
    final_recall = recall_score(y_val, y_pred_final)
    final_acc = accuracy_score(y_val, y_pred_final)
    print(f"Recall at threshold: {final_recall:.4f} >= {recall_min}")
    print(f"Accuracy at threshold: {final_acc:.4f}")
    
    return best_model, study.best_params, best_threshold
def train_xgboost_optuna_with_precision_constraint(
    X_train,
    y_train,
    n_trials=100,
    cv_folds=5,
    beta=2.0,
    precision_min=0.70  # minimum precision constraint (changed from recall_min)
):
    """
    Optuna optimization with precision constraint.
    Maximizes (recall + accuracy) / 2 while ensuring precision >= precision_min
    """
    
    print("\n" + "=" * 60)
    print(f"Starting Optuna Optimisation with precision >= {precision_min}")
    print("=" * 60)
    
    def objective(trial):
        # Expanded hyperparameter ranges
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 15),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 5.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'verbosity': 0,
            'use_label_encoder': False
        }
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]
            
            model = xgb.XGBClassifier(**params, early_stopping_rounds=30)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            y_proba = model.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0.30, 0.99, 60)
            
            best_fold_score = -1.0
            for t in thresholds:
                y_pred = (y_proba >= t).astype(int)
                precision = precision_score(y_val, y_pred, zero_division=0)
                
                # Only consider thresholds meeting precision constraint
                if precision >= precision_min:
                    # Maximize (recall + accuracy) / 2
                    recall = recall_score(y_val, y_pred)
                    acc = accuracy_score(y_val, y_pred)
                    score = (recall + acc) / 2
                    if score > best_fold_score:
                        best_fold_score = score
            
            # If no threshold meets precision constraint, penalize heavily
            if best_fold_score == -1.0:
                best_fold_score = 0.0
                
            fold_scores.append(best_fold_score)
        
        return np.mean(fold_scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "=" * 60)
    print("OPTUNA RESULTS")
    print("=" * 60)
    
    print(f"Best CV Score : {study.best_value:.4f}")
    
    print("\nBest Parameters:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    
    # Final train / validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )
    
    # Final model with best parameters
    best_model = xgb.XGBClassifier(
        **study.best_params,
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )
    
    best_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Find threshold that meets precision constraint while maximizing (recall+acc)/2
    y_proba = best_model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.30, 0.99, 160)
    
    best_threshold = 0.5
    best_score = -1.0
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precision = precision_score(y_val, y_pred, zero_division=0)
        if precision >= precision_min:
            recall = recall_score(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)
            score = (recall + acc) / 2
            if score > best_score:
                best_score = score
                best_threshold = t
    
    print(f"\nFinal Validation Results")
    print(f"Best Threshold (precision >= {precision_min}): {best_threshold:.4f}")
    print(f"Validation Score (recall+acc)/2: {best_score:.4f}")
    
    # Verify precision constraint
    y_pred_final = (y_proba >= best_threshold).astype(int)
    final_precision = precision_score(y_val, y_pred_final, zero_division=0)
    final_recall = recall_score(y_val, y_pred_final)
    final_acc = accuracy_score(y_val, y_pred_final)
    print(f"Precision at threshold: {final_precision:.4f} >= {precision_min}")
    print(f"Recall at threshold: {final_recall:.4f}")
    print(f"Accuracy at threshold: {final_acc:.4f}")
    
    return best_model, study.best_params, best_threshold