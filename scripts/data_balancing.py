# scripts/data_balancing.py

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import EditedNearestNeighbours
from gan_util import split_num_cat, combine_num_cat
from cwgan_gp import CWGANGP


def prepare_minority_data_for_gan(df, feature_info, target_col='Class', minority_label=1):
    """Extract minority class, scale numericals with MinMaxScaler, return data and scaler."""
    df_min = df[df[target_col] == minority_label].copy()
    num_raw, cat_list, _ = split_num_cat(df_min, feature_info)
    scaler = MinMaxScaler()
    if num_raw.shape[1] > 0:
        num_scaled = scaler.fit_transform(num_raw)
    else:
        num_scaled = num_raw
    return (num_scaled, cat_list), scaler, df_min


def prepare_full_data_for_gan(df, feature_info, target_col='Class', minority_label=1, batch_size=64):
    """
    Prepare full dataset (both classes) for cWGAN-GP training with conditioning.
    Returns a tf.data.Dataset yielding (real_data_tuple, cond, labels).
    """
    num_raw, cat_list, _ = split_num_cat(df, feature_info)
    scaler = MinMaxScaler()
    if num_raw.shape[1] > 0:
        num_scaled = scaler.fit_transform(num_raw)
    else:
        num_scaled = num_raw

    n_num = num_scaled.shape[1]
    n_cat_dims = [arr.shape[1] for arr in cat_list]

    cond = (df[target_col] == minority_label).astype(np.float32).values.reshape(-1, 1)
    labels = df[target_col].astype(np.float32).values.reshape(-1, 1)

    data_components = []
    if n_num > 0:
        data_components.append(num_scaled.astype(np.float32))
    for arr in cat_list:
        data_components.append(arr.astype(np.float32))

    dataset = tf.data.Dataset.from_tensor_slices(
        (tuple(data_components), cond, labels)
    )
    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset, scaler, n_num, n_cat_dims, num_scaled, cat_list


def create_tf_dataset(num_scaled, cat_list, cond, batch_size=64, shuffle=True):
    """Create tf.data.Dataset from scaled arrays."""
    data_components = []
    if num_scaled.shape[1] > 0:
        data_components.append(num_scaled.astype(np.float32))
    for cat_arr in cat_list:
        data_components.append(cat_arr.astype(np.float32))
    dataset = tf.data.Dataset.from_tensor_slices((tuple(data_components), cond.astype(np.float32)))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(cond))
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def train_cwgan_on_minority(df_processed, feature_info, target_col='Class', minority_label=1,
                           latent_dim=256, epochs=3000, batch_size=64, n_critic=5, gp_weight=10.0,
                           gen_hidden=[256, 256], critic_hidden=[256, 256],
                           learning_rate=1e-4, gumbel_temperature=0.5,
                           verbose=True, save_path='models/cwgan/'):
    """Train a cWGAN-GP on the minority class and save model and scaler."""
    (num_scaled, cat_list), scaler, df_min = prepare_minority_data_for_gan(
        df_processed, feature_info, target_col, minority_label
    )

    cond = np.ones((len(df_min), 1), dtype=np.float32)
    n_num = num_scaled.shape[1] if num_scaled.shape[0] > 0 else 0
    n_cat_dims = [arr.shape[1] for arr in cat_list]

    dataset = create_tf_dataset(num_scaled, cat_list, cond, batch_size=batch_size, shuffle=True)

    gan = CWGANGP(
        n_num=n_num, n_cat_dims=n_cat_dims,
        latent_dim=latent_dim, cond_dim=1,
        gen_hidden=gen_hidden, critic_hidden=critic_hidden,
        gp_weight=gp_weight, learning_rate=learning_rate,
        gumbel_temperature=gumbel_temperature
    )

    gan.train(dataset, epochs=epochs, n_critic=n_critic, verbose=verbose)

    gan.save(save_path)
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
    print(f"Model saved to {save_path}")
    return gan, scaler


def train_cwgan_full(df_train, feature_info, target_col='Class', minority_label=1,
                     latent_dim=256, epochs=3000, batch_size=64, n_critic=5,
                     gp_weight=10.0, aux_weight=1.0, 
                     learning_rate=1e-4, critic_learning_rate=None, gen_learning_rate=None,
                     gumbel_temperature=0.5, use_cross_layers=True,
                     early_stopping_patience=2000, verbose=True, save_path='models/cwgan_full/',
                     use_lr_scheduling=False, restore_best=True):
    """
    Train cWGAN-GP on full dataset with conditioning and auxiliary classifier.
    
    Args:
        critic_learning_rate: Separate LR for critic (if None, uses learning_rate)
        gen_learning_rate: Separate LR for generator (if None, uses learning_rate)
        use_lr_scheduling: Whether to use exponential learning rate decay
        restore_best: Whether to restore best model at early stopping
    """
    dataset, scaler, n_num, n_cat_dims, _, _ = prepare_full_data_for_gan(
        df_train, feature_info, target_col, minority_label, batch_size
    )

    gan = CWGANGP(
        n_num=n_num, n_cat_dims=n_cat_dims,
        latent_dim=latent_dim, cond_dim=1,
        gen_hidden=[256, 256], critic_hidden=[256, 256],
        gp_weight=gp_weight, aux_weight=aux_weight,
        learning_rate=learning_rate,
        critic_learning_rate=critic_learning_rate,
        gen_learning_rate=gen_learning_rate,
        gumbel_temperature=gumbel_temperature,
        use_cross_layers=use_cross_layers,
        use_lr_scheduling=use_lr_scheduling
    )

    gan.train(dataset, epochs=epochs, n_critic=n_critic, verbose=verbose,
              early_stopping_patience=early_stopping_patience, restore_best=restore_best)

    gan.save(save_path)
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
    print(f"Model saved to {save_path}")
    return gan, scaler


def generate_synthetic_samples(gan, scaler, n_samples, feature_info, columns_order,
                               minority_label=1, target_col='Class', feature_ranges=None):
    """Generate n_samples synthetic minority samples."""
    cond = np.ones((n_samples, 1), dtype=np.float32)
    num_gen_scaled, cat_gen = gan.generate(cond, batch_size=64)

    if num_gen_scaled is not None:
        num_gen = scaler.inverse_transform(num_gen_scaled)
        if feature_ranges is not None:
            for i, col in enumerate(feature_info['numerical']):
                min_val, max_val = feature_ranges[col]
                num_gen[:, i] = np.clip(num_gen[:, i], min_val, max_val)
        integer_cols = ['Duration', 'CreditAmount', 'Age']
        for col in integer_cols:
            if col in feature_info['numerical']:
                idx = feature_info['numerical'].index(col)
                num_gen[:, idx] = np.round(num_gen[:, idx]).astype(int)
    else:
        num_gen = None

    df_synth = combine_num_cat(num_gen, cat_gen, feature_info, columns_order)
    for cat_name, cols in feature_info['categorical'].items():
        for col in cols:
            if col in df_synth.columns:
                df_synth[col] = df_synth[col].astype(int)
    for col in feature_info['numerical']:
        if col in df_synth.columns and col not in integer_cols:
            df_synth[col] = df_synth[col].astype(float)

    df_synth[target_col] = minority_label
    return df_synth


def generate_and_clean_synthetic_samples(gan, scaler, feature_info, columns_order,
                                         df_train_imb, target_col='Class', minority_label=1,
                                         factor=2.0, n_neighbors=3):
    """
    Generate factor * needed synthetic samples, combine with imbalanced training set,
    then apply Edited Nearest Neighbours (ENN) to remove ambiguous/overlapping samples.
    """
    majority_count = (df_train_imb[target_col] != minority_label).sum()
    minority_count = (df_train_imb[target_col] == minority_label).sum()
    needed = majority_count - minority_count
    
    if needed <= 0:
        print("Dataset already balanced. No generation needed.")
        return df_train_imb
    
    n_generate = int(needed * factor)
    print(f"Generating {n_generate} synthetic samples ({factor}x needed = {needed})")
    
    feature_ranges = {
        col: (df_train_imb[col].min(), df_train_imb[col].max()) 
        for col in feature_info['numerical']
    }
    
    df_synth_excess = generate_synthetic_samples(
        gan, scaler, n_generate, feature_info, columns_order,
        minority_label=minority_label, target_col=target_col,
        feature_ranges=feature_ranges
    )
    
    df_combined = pd.concat([df_train_imb, df_synth_excess], ignore_index=True)
    print(f"Combined shape: {df_combined.shape} (original + excess synthetic)")
    
    X_temp = df_combined.drop(columns=[target_col])
    y_temp = df_combined[target_col]
    
    enn = EditedNearestNeighbours(
        sampling_strategy='all',
        n_neighbors=n_neighbors,
        kind_sel='all'
    )
    
    X_clean, y_clean = enn.fit_resample(X_temp, y_temp)
    df_balanced = pd.concat([X_clean, y_clean], axis=1)
    
    final_majority = (df_balanced[target_col] != minority_label).sum()
    final_minority = (df_balanced[target_col] == minority_label).sum()
    print(f"\nAfter ENN cleaning:")
    print(f"  Majority: {final_majority}, Minority: {final_minority}")
    print(f"  Final balance ratio: {final_majority/final_minority:.2f}:1")
    
    return df_balanced