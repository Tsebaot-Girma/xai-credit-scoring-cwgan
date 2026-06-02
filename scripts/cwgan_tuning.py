# scripts/cwgan_tuning.py

import optuna
from optuna.trial import TrialState
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

from cwgan_gp import CWGANGP
from data_balancing import prepare_full_data_for_gan, generate_synthetic_samples
from gan_util import build_feature_info, split_num_cat, combine_num_cat


def objective_cwgan(trial, df_train, feature_info, target_col, minority_label, 
                    X_val, y_val, num_cols, max_epochs=300):
    """
    Optuna objective for cWGAN-GP hyperparameter tuning.
    """
    
    # ============================================================
    # Hyperparameter search space
    # ============================================================
    
    # Architecture
    latent_dim = trial.suggest_categorical('latent_dim', [128, 256])
    gen_hidden_1 = trial.suggest_int('gen_hidden_1', 128, 512, step=64)
    gen_hidden_2 = trial.suggest_int('gen_hidden_2', 128, 512, step=64)
    critic_hidden_1 = trial.suggest_int('critic_hidden_1', 128, 512, step=64)
    critic_hidden_2 = trial.suggest_int('critic_hidden_2', 128, 512, step=64)
    
    # Learning rates (CRITICAL for stability)
    learning_rate = trial.suggest_float('learning_rate', 5e-5, 2e-4, log=True)
    critic_lr_ratio = trial.suggest_float('critic_lr_ratio', 0.2, 0.6, step=0.1)
    critic_learning_rate = learning_rate * critic_lr_ratio
    
    # Training parameters
    n_critic = trial.suggest_int('n_critic', 1, 4)
    gp_weight = trial.suggest_float('gp_weight', 5.0, 20.0, step=2.0)
    aux_weight = trial.suggest_float('aux_weight', 0.5, 2.0, step=0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    gumbel_temperature = trial.suggest_float('gumbel_temperature', 0.3, 0.7, step=0.1)
    use_cross_layers = trial.suggest_categorical('use_cross_layers', [True, False])
    
    # ============================================================
    # Train cWGAN-GP with limited epochs
    # ============================================================
    
    try:
        # Prepare dataset - FIXED: Use the full df_train directly
        dataset, scaler, n_num, n_cat_dims, _, _ = prepare_full_data_for_gan(
            df_train, feature_info, target_col, minority_label, batch_size
        )
        
        # Initialize GAN
        gan = CWGANGP(
            n_num=n_num, n_cat_dims=n_cat_dims,
            latent_dim=latent_dim, cond_dim=1,
            gen_hidden=[gen_hidden_1, gen_hidden_2],
            critic_hidden=[critic_hidden_1, critic_hidden_2],
            gp_weight=gp_weight, aux_weight=aux_weight,
            learning_rate=learning_rate,
            critic_learning_rate=critic_learning_rate,
            gen_learning_rate=learning_rate,
            gumbel_temperature=gumbel_temperature,
            use_cross_layers=use_cross_layers,
            use_lr_scheduling=False
        )
        
        # Train (limited epochs for tuning speed)
        gan.train(
            dataset, 
            epochs=max_epochs, 
            n_critic=n_critic, 
            verbose=False,
            early_stopping_patience=50,
            restore_best=True
        )
        
        # Get best generator loss (quick quality proxy)
        best_g_loss = gan.best_g_loss if gan.best_g_loss else min(gan.g_losses)
        
        # ============================================================
        # Generate synthetic samples for evaluation
        # ============================================================
        
        # Calculate how many synthetic samples to generate
        train_majority = (df_train[target_col] != minority_label).sum()
        train_minority = (df_train[target_col] == minority_label).sum()
        needed = train_majority - train_minority
        
        if needed <= 0:
            # Already balanced, use 500 samples
            n_synthetic = min(500, len(df_train))
        else:
            n_synthetic = min(needed, 1000)
        
        # Generate synthetic minority samples
        feature_ranges = {
            col: (df_train[col].min(), df_train[col].max()) 
            for col in feature_info['numerical']
        }
        
        cond = np.ones((n_synthetic, 1), dtype=np.float32)
        num_gen_scaled, cat_gen = gan.generate(cond, batch_size=64)
        
        if num_gen_scaled is not None:
            num_gen = scaler.inverse_transform(num_gen_scaled)
            # Clip to realistic ranges
            for i, col in enumerate(feature_info['numerical']):
                min_val, max_val = feature_ranges[col]
                num_gen[:, i] = np.clip(num_gen[:, i], min_val, max_val)
        else:
            num_gen = None
        
        # Combine into DataFrame - FIXED: use correct column order
        df_synth = combine_num_cat(num_gen, cat_gen, feature_info, df_train.columns.tolist())
        df_synth[target_col] = minority_label
        
        # ============================================================
        # Evaluate using classifier on validation set
        # ============================================================
        
        # Combine original training data with synthetic
        X_combined = pd.concat([
            df_train.drop(columns=[target_col]),
            df_synth.drop(columns=[target_col])
        ], ignore_index=True)
        y_combined = pd.concat([
            df_train[target_col],
            pd.Series([minority_label] * len(df_synth))
        ], ignore_index=True)
        
        # Train a simple classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        clf.fit(X_combined, y_combined)
        
        # Evaluate on validation set
        y_pred_proba = clf.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        # Also track Wasserstein distance for feature distributions
        w_distances = []
        real_min = df_train[df_train[target_col] == minority_label][num_cols]
        synth_samples = df_synth[num_cols]
        
        from scipy.stats import wasserstein_distance
        for col in num_cols:
            w_dist = wasserstein_distance(real_min[col].values, synth_samples[col].values)
            w_distances.append(w_dist)
        avg_w_dist = np.mean(w_distances)
        
        # Store trial metadata
        trial.set_user_attr('best_g_loss', float(best_g_loss))
        trial.set_user_attr('auc_score', float(auc_score))
        trial.set_user_attr('avg_w_dist', float(avg_w_dist))
        
        # Objective: maximize AUC (higher is better)
        return auc_score
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def tune_cwgan_hyperparameters(df_train, feature_info, target_col, minority_label,
                               num_cols, n_trials=30, test_size=0.2, max_epochs=300):
    """
    Run Optuna hyperparameter tuning for cWGAN-GP.
    """
    
    print("\n" + "="*60)
    print("OPTUNA HYPERPARAMETER TUNING FOR cWGAN-GP")
    print("="*60)
    print(f"Number of trials: {n_trials}")
    print(f"Max epochs per trial: {max_epochs}")
    
    # Create validation set for evaluation
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # FIXED: Recombine with correct column names
    df_train_tune = pd.concat([X_train, y_train], axis=1)
    df_train_tune.columns = list(X_train.columns) + [target_col]
    
    print(f"Training set size: {len(df_train_tune)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',  # Maximize AUC
        study_name='cwgan_tuning',
        storage=None,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective_cwgan(
            trial, df_train_tune, feature_info, target_col, minority_label,
            X_val.values, y_val.values, num_cols, max_epochs
        ),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # ============================================================
    # Display results
    # ============================================================
    
    print("\n" + "="*60)
    print("TUNING RESULTS")
    print("="*60)
    
    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best AUC score: {best_trial.value:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    print("\nBest trial metrics:")
    # FIXED: Handle missing values gracefully
    best_g_loss = best_trial.user_attrs.get('best_g_loss')
    auc_score = best_trial.user_attrs.get('auc_score')
    avg_w_dist = best_trial.user_attrs.get('avg_w_dist')
    
    if best_g_loss is not None:
        print(f"  Best Generator Loss: {best_g_loss:.4f}")
    else:
        print(f"  Best Generator Loss: N/A")
    
    if auc_score is not None:
        print(f"  Validation AUC: {auc_score:.4f}")
    else:
        print(f"  Validation AUC: N/A")
    
    if avg_w_dist is not None:
        print(f"  Avg Wasserstein Distance: {avg_w_dist:.4f}")
    else:
        print(f"  Avg Wasserstein Distance: N/A")
    
    # ============================================================
    # Visualize results
    # ============================================================
    
    try:
        # Optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        
        # Parameter importances
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
        
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # Save study
    os.makedirs('../models', exist_ok=True)
    joblib.dump(study, '../models/cwgan_optuna_study.pkl')
    print("\n✅ Study saved to ../models/cwgan_optuna_study.pkl")
    
    return study


# ============================================================
# Quick tuning function (fewer parameters, faster)
# ============================================================

def quick_tune_cwgan(df_train, feature_info, target_col, minority_label,
                     num_cols, n_trials=15, max_epochs=200):
    """
    Faster tuning focusing on the most critical hyperparameters.
    """
    
    def objective_quick(trial, df_train, feature_info, target_col, 
                        minority_label, X_val, y_val, num_cols):
        
        # Focus on most important parameters
        critic_lr = trial.suggest_float('critic_lr', 1e-5, 8e-5, log=True)
        gen_lr = trial.suggest_float('gen_lr', 5e-5, 2e-4, log=True)
        n_critic = trial.suggest_int('n_critic', 1, 3)
        gp_weight = trial.suggest_float('gp_weight', 5.0, 15.0, step=2.0)
        latent_dim = trial.suggest_categorical('latent_dim', [128, 256])
        
        try:
            # Train quickly
            dataset, scaler, n_num, n_cat_dims, _, _ = prepare_full_data_for_gan(
                df_train, feature_info, target_col, minority_label, 64
            )
            
            gan = CWGANGP(
                n_num=n_num, n_cat_dims=n_cat_dims,
                latent_dim=latent_dim, cond_dim=1,
                gp_weight=gp_weight,
                learning_rate=gen_lr,
                critic_learning_rate=critic_lr,
                gen_learning_rate=gen_lr
            )
            
            gan.train(dataset, epochs=max_epochs, n_critic=n_critic, 
                      verbose=False, early_stopping_patience=30)
            
            # Quick quality check using best generator loss
            best_g_loss = min(gan.g_losses)
            
            # Return negative because we want to maximize (more negative = better)
            return -best_g_loss
            
        except Exception as e:
            print(f"Quick trial failed: {e}")
            return float('inf')
    
    # Create validation split
    X = df_train.drop(columns=[target_col])
    y = df_train[target_col]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # FIXED: Recombine correctly
    df_train_tune = pd.concat([X_train, y_train], axis=1)
    df_train_tune.columns = list(X_train.columns) + [target_col]
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_quick(
            trial, df_train_tune, feature_info, target_col, minority_label,
            X_val.values, y_val.values, num_cols
        ),
        n_trials=n_trials
    )
    
    print("\nQuick tuning results:")
    print(f"Best value: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    
    return study