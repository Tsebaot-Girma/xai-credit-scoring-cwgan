import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from cwgan_gp_pytorch import ConditionalWGAN_GP
from gan_util import get_data_dim, split_num_cat


def prepare_full_data_for_gan_pytorch(
    df: pd.DataFrame,
    feature_info: Dict,
    target_col: str = "Class",
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare the full training set for cWGAN-GP.

    The GAN must see both majority and minority rows and receive the class label
    as a condition. Numerical columns are MinMax scaled; one-hot categorical
    columns are passed through unchanged.
    """
    num_raw, cat_list, _ = split_num_cat(df, feature_info)
    scaler = MinMaxScaler()
    if num_raw.shape[1] > 0:
        num_scaled = scaler.fit_transform(num_raw)
    else:
        num_scaled = np.empty((len(df), 0), dtype=np.float32)

    parts = [num_scaled.astype(np.float32)]
    parts.extend([arr.astype(np.float32) for arr in cat_list])
    X_combined = np.hstack(parts).astype(np.float32)
    y = df[target_col].astype(int).values
    return X_combined, y, scaler


def train_cwgan_full_pytorch(
    df_train: pd.DataFrame,
    feature_info: Dict,
    target_col: str = "Class",
    latent_dim: int = 64,
    epochs: int = 300,
    batch_size: int = 128,
    n_critic: int = 5,
    gp_weight: float = 10.0,
    aux_weight: float = 1.0,
    learning_rate: float = 1e-4,
    temperature_start: float = 1.0,
    temperature_end: float = 0.5,
    temperature_anneal_epochs: int = 100,
    early_stopping_patience: int = 50,
    hidden_dim: int = 256,
    cross_layers: int = 2,
    use_woe: bool = False,
    verbose: bool = True,
    save_path: str = "models/cwgan_full_pytorch/",
    random_state: int = 42,
) -> Tuple[ConditionalWGAN_GP, MinMaxScaler]:
    X_data, y_labels, scaler = prepare_full_data_for_gan_pytorch(df_train, feature_info, target_col)
    n_num, _, n_cat_dims = get_data_dim(feature_info)

    gan = ConditionalWGAN_GP(
        numerical_dim=n_num,
        categorical_dims=n_cat_dims,
        latent_dim=latent_dim,
        n_classes=2,
        hidden_dim=hidden_dim,
        cross_layers=cross_layers,
        lambda_gp=gp_weight,
        aux_weight=aux_weight,
        lr=learning_rate,
        n_critic=n_critic,
        use_woe=use_woe,
        seed=random_state,
    )
    gan.train(
        X_train=X_data,
        y_train=y_labels,
        epochs=epochs,
        batch_size=batch_size,
        n_critic=n_critic,
        lr_g=learning_rate,
        lr_d=learning_rate,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        temperature_anneal_epochs=temperature_anneal_epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
        random_state=random_state,
    )
    gan.save(save_path)
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))
    joblib.dump(feature_info, os.path.join(save_path, "feature_info.pkl"))
    return gan, scaler


def _postprocess_categorical_blocks(
    synthetic_data: np.ndarray,
    feature_info: Dict,
    n_num: int,
) -> np.ndarray:
    out = synthetic_data.copy()
    start = n_num
    for cols in feature_info["categorical"].values():
        width = len(cols)
        block = out[:, start:start + width]
        hard = np.zeros_like(block)
        hard[np.arange(len(block)), np.argmax(block, axis=1)] = 1.0
        out[:, start:start + width] = hard
        start += width
    return out


def generate_synthetic_samples_pytorch(
    gan: ConditionalWGAN_GP,
    scaler: MinMaxScaler,
    n_samples: int,
    feature_info: Dict,
    columns_order: List[str],
    minority_label: int = 1,
    target_col: str = "Class",
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    integer_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    synthetic_data = gan.generate_samples(class_label=minority_label, n_samples=n_samples, temperature=0.1)
    n_num = len(feature_info["numerical"])
    synthetic_data = _postprocess_categorical_blocks(synthetic_data, feature_info, n_num)

    if n_num > 0:
        synthetic_num = synthetic_data[:, :n_num]
        synthetic_num_original = scaler.inverse_transform(synthetic_num)

        if feature_ranges is not None:
            for i, col in enumerate(feature_info["numerical"]):
                min_val, max_val = feature_ranges[col]
                synthetic_num_original[:, i] = np.clip(synthetic_num_original[:, i], min_val, max_val)

        integer_cols = integer_cols or []
        for col in integer_cols:
            if col in feature_info["numerical"]:
                idx = feature_info["numerical"].index(col)
                synthetic_num_original[:, idx] = np.round(synthetic_num_original[:, idx])

        synthetic_data[:, :n_num] = synthetic_num_original

    df_synth = pd.DataFrame(synthetic_data, columns=columns_order)
    for _, cols in feature_info["categorical"].items():
        for col in cols:
            df_synth[col] = df_synth[col].round().astype(int)
    df_synth[target_col] = minority_label
    return df_synth


def balance_dataset_with_cwgan_pytorch(
    df_train: pd.DataFrame,
    gan: ConditionalWGAN_GP,
    scaler: MinMaxScaler,
    feature_info: Dict,
    minority_label: int = 1,
    majority_label: int = 0,
    target_col: str = "Class",
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    integer_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    majority_count = int((df_train[target_col] == majority_label).sum())
    minority_count = int((df_train[target_col] == minority_label).sum())
    needed = max(majority_count - minority_count, 0)
    print(f"Majority: {majority_count}, minority: {minority_count}, synthetic needed: {needed}")

    if needed == 0:
        return df_train.copy(), pd.DataFrame(columns=df_train.columns)

    columns_order = [col for col in df_train.columns if col != target_col]
    df_synth = generate_synthetic_samples_pytorch(
        gan=gan,
        scaler=scaler,
        n_samples=needed,
        feature_info=feature_info,
        columns_order=columns_order,
        minority_label=minority_label,
        target_col=target_col,
        feature_ranges=feature_ranges,
        integer_cols=integer_cols,
    )
    df_balanced = pd.concat([df_train, df_synth], ignore_index=True)
    return df_balanced, df_synth


def load_cwgan_and_balance_pytorch(
    df_train: pd.DataFrame,
    gan_path: str,
    minority_label: int = 1,
    majority_label: int = 0,
    target_col: str = "Class",
    feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    integer_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gan = ConditionalWGAN_GP.load(gan_path)
    scaler = joblib.load(os.path.join(gan_path, "scaler.pkl"))
    feature_info = joblib.load(os.path.join(gan_path, "feature_info.pkl"))
    return balance_dataset_with_cwgan_pytorch(
        df_train=df_train,
        gan=gan,
        scaler=scaler,
        feature_info=feature_info,
        minority_label=minority_label,
        majority_label=majority_label,
        target_col=target_col,
        feature_ranges=feature_ranges,
        integer_cols=integer_cols,
    )
