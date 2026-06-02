import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


EPS = 1e-6


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_train_test(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def detect_column_types(
    X: pd.DataFrame,
    target_col: Optional[str] = None,
    categorical_threshold: int = 10,
) -> Tuple[List[str], List[str]]:
    categorical_cols = []
    numerical_cols = []
    for col in X.columns:
        if col == target_col:
            continue
        unique_count = X[col].nunique(dropna=True)
        if (
            pd.api.types.is_object_dtype(X[col])
            or pd.api.types.is_categorical_dtype(X[col])
            or unique_count < categorical_threshold
        ):
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    return categorical_cols, numerical_cols


def _safe_mode(series: pd.Series):
    mode = series.mode(dropna=True)
    return "missing" if mode.empty else mode.iloc[0]


def _clip_iqr(
    X: pd.DataFrame,
    numerical_cols: List[str],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    X = X.copy()
    fitted_bounds = {} if bounds is None else bounds
    for col in numerical_cols:
        if bounds is None:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            fitted_bounds[col] = (q1 - factor * iqr, q3 + factor * iqr)
        lower, upper = fitted_bounds[col]
        X[col] = X[col].clip(lower, upper)
    return X, fitted_bounds


def _woe(non_events: float, events: float, total_non_events: float, total_events: float) -> float:
    non_event_rate = (non_events + EPS) / (total_non_events + EPS)
    event_rate = (events + EPS) / (total_events + EPS)
    return float(np.log(non_event_rate / event_rate))


@dataclass
class WOETransformer:
    categorical_cols: List[str]
    numerical_cols: List[str]
    n_bins: int = 10
    event_label: int = 1
    mappings_: Dict[str, Dict[str, float]] = field(default_factory=dict)
    bin_edges_: Dict[str, np.ndarray] = field(default_factory=dict)
    medians_: Dict[str, float] = field(default_factory=dict)
    iv_: Dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOETransformer":
        X = X.reset_index(drop=True).copy()
        y = pd.Series(y).astype(int).reset_index(drop=True)
        total_events = float((y == self.event_label).sum())
        total_non_events = float((y != self.event_label).sum())

        for col in self.categorical_cols:
            grouped = X[col].astype("object").fillna("missing").astype(str)
            self._fit_grouped(col, grouped, y, total_non_events, total_events)

        for col in self.numerical_cols:
            series = pd.to_numeric(X[col], errors="coerce")
            self.medians_[col] = float(series.median())
            filled = series.fillna(self.medians_[col])
            try:
                _, edges = pd.qcut(filled, q=self.n_bins, retbins=True, duplicates="drop")
            except ValueError:
                edges = np.array([filled.min(), filled.max()])
            if len(edges) <= 2 or np.isclose(edges[0], edges[-1]):
                edges = np.array([filled.min() - EPS, filled.max() + EPS])
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges_[col] = edges
            grouped = pd.cut(filled, bins=edges, include_lowest=True).astype(str)
            self._fit_grouped(col, grouped, y, total_non_events, total_events)
        return self

    def _fit_grouped(
        self,
        col: str,
        grouped: pd.Series,
        y: pd.Series,
        total_non_events: float,
        total_events: float,
    ) -> None:
        mapping = {}
        iv = 0.0
        for value in grouped.unique():
            mask = grouped == value
            events = float(((y == self.event_label) & mask).sum())
            non_events = float(((y != self.event_label) & mask).sum())
            value_woe = _woe(non_events, events, total_non_events, total_events)
            non_event_dist = (non_events + EPS) / (total_non_events + EPS)
            event_dist = (events + EPS) / (total_events + EPS)
            iv += (non_event_dist - event_dist) * value_woe
            mapping[str(value)] = value_woe
        self.mappings_[col] = mapping
        self.iv_[col] = float(iv)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.categorical_cols:
            series = X[col].astype("object").fillna("missing").astype(str)
            out[col] = series.map(self.mappings_[col]).fillna(0.0).astype(float)
        for col in self.numerical_cols:
            series = pd.to_numeric(X[col], errors="coerce").fillna(self.medians_[col])
            grouped = pd.cut(series, bins=self.bin_edges_[col], include_lowest=True).astype(str)
            out[col] = grouped.map(self.mappings_[col]).fillna(0.0).astype(float)
        return out[self.categorical_cols + self.numerical_cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def compute_woe(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: List[str],
    numerical_cols: List[str],
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float], WOETransformer]:
    transformer = WOETransformer(categorical_cols, numerical_cols, n_bins=n_bins)
    transformed = transformer.fit_transform(X, y)
    return transformed, transformer.iv_, transformer


class CreditDataPreprocessor:
    def __init__(
        self,
        use_woe: bool = False,
        categorical_threshold: int = 10,
        n_bins: int = 10,
        clip_outliers: bool = True,
        drop_first: bool = False,
        seed: int = 42,
    ):
        self.use_woe = use_woe
        self.categorical_threshold = categorical_threshold
        self.n_bins = n_bins
        self.clip_outliers = clip_outliers
        self.drop_first = drop_first
        self.seed = seed
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self.num_impute_: Dict[str, float] = {}
        self.cat_impute_: Dict[str, object] = {}
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.scaler_: Optional[MinMaxScaler] = None
        self.encoder_: Optional[OneHotEncoder] = None
        self.woe_: Optional[WOETransformer] = None
        self.iv_: Dict[str, float] = {}
        self.feature_names_: List[str] = []
        self.categorical_dims_: List[int] = []
        self.categorical_feature_names_: List[str] = []

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        set_seed(self.seed)
        self.categorical_cols, self.numerical_cols = detect_column_types(
            X_train, categorical_threshold=self.categorical_threshold
        )
        X = self._fit_impute(X_train)
        if self.clip_outliers:
            X, self.outlier_bounds_ = _clip_iqr(X, self.numerical_cols)

        if self.use_woe:
            X_out, self.iv_, self.woe_ = compute_woe(
                X, y_train, self.categorical_cols, self.numerical_cols, self.n_bins
            )
            self.feature_names_ = list(X_out.columns)
            return X_out

        self.scaler_ = MinMaxScaler()
        parts = []
        if self.numerical_cols:
            X_num = pd.DataFrame(
                self.scaler_.fit_transform(X[self.numerical_cols]),
                columns=self.numerical_cols,
                index=X.index,
            )
            parts.append(X_num)

        drop = "first" if self.drop_first else None
        try:
            self.encoder_ = OneHotEncoder(drop=drop, handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder_ = OneHotEncoder(drop=drop, handle_unknown="ignore", sparse=False)

        if self.categorical_cols:
            encoded = self.encoder_.fit_transform(X[self.categorical_cols].astype(str))
            self.categorical_feature_names_ = list(self.encoder_.get_feature_names_out(self.categorical_cols))
            self.categorical_dims_ = [
                len(cats) - (1 if self.drop_first else 0) for cats in self.encoder_.categories_
            ]
            parts.append(pd.DataFrame(encoded, columns=self.categorical_feature_names_, index=X.index))

        X_out = pd.concat(parts, axis=1)
        self.feature_names_ = list(X_out.columns)
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._apply_impute(X)
        if self.clip_outliers and self.numerical_cols:
            X, _ = _clip_iqr(X, self.numerical_cols, bounds=self.outlier_bounds_)

        if self.use_woe:
            return self.woe_.transform(X)[self.feature_names_]

        parts = []
        if self.numerical_cols:
            parts.append(pd.DataFrame(
                self.scaler_.transform(X[self.numerical_cols]),
                columns=self.numerical_cols,
                index=X.index,
            ))
        if self.categorical_cols:
            encoded = self.encoder_.transform(X[self.categorical_cols].astype(str))
            parts.append(pd.DataFrame(encoded, columns=self.categorical_feature_names_, index=X.index))
        return pd.concat(parts, axis=1)[self.feature_names_]

    def _fit_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.numerical_cols:
            self.num_impute_[col] = float(pd.to_numeric(X[col], errors="coerce").median())
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.num_impute_[col])
        for col in self.categorical_cols:
            self.cat_impute_[col] = _safe_mode(X[col])
            X[col] = X[col].fillna(self.cat_impute_[col]).astype(str)
        return X

    def _apply_impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.numerical_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(self.num_impute_[col])
        for col in self.categorical_cols:
            X[col] = X[col].fillna(self.cat_impute_[col]).astype(str)
        return X

    def print_summary(self) -> None:
        print(f"Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        print(f"Numerical columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        print(f"Numerical imputations: {self.num_impute_}")
        print(f"Categorical imputations: {self.cat_impute_}")
        if self.use_woe:
            print("Information Value:")
            for name, value in sorted(self.iv_.items(), key=lambda item: item[1], reverse=True):
                print(f"  {name}: {value:.6f}")
