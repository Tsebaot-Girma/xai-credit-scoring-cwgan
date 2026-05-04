# scripts/gan_util.py

import numpy as np
import pandas as pd
from collections import defaultdict

def build_feature_info(df, original_cat_cols, target_col='Class'):
    feature_cols = [c for c in df.columns if c != target_col]
    cat_onehot_cols = []
    for orig_cat in original_cat_cols:
        matching = [c for c in feature_cols if c.startswith(orig_cat + '_')]
        cat_onehot_cols.extend(matching)
    numerical_cols = [c for c in feature_cols if c not in cat_onehot_cols]
    categorical_map = {}
    for orig_cat in original_cat_cols:
        matching = [c for c in feature_cols if c.startswith(orig_cat + '_')]
        if matching:
            categorical_map[orig_cat] = matching
    return {'numerical': numerical_cols, 'categorical': categorical_map}

def get_data_dim(feature_info):
    n_num = len(feature_info['numerical'])
    n_cat_groups = len(feature_info['categorical'])
    n_cat_dims = [len(cols) for cols in feature_info['categorical'].values()]
    return n_num, n_cat_groups, n_cat_dims

def split_num_cat(data, feature_info):
    num_data = data[feature_info['numerical']].values.astype(np.float32)
    cat_data_list = []
    cat_column_groups = []
    for orig_cat, cols in feature_info['categorical'].items():
        cat_data_list.append(data[cols].values.astype(np.float32))
        cat_column_groups.append(cols)
    return num_data, cat_data_list, cat_column_groups

def combine_num_cat(num_array, cat_arrays, feature_info, columns_order):
    data_dict = {}
    if num_array is not None:
        for i, col in enumerate(feature_info['numerical']):
            data_dict[col] = num_array[:, i]
    for (orig_cat, cols), cat_arr in zip(feature_info['categorical'].items(), cat_arrays):
        for j, col in enumerate(cols):
            data_dict[col] = cat_arr[:, j]
    df = pd.DataFrame(data_dict)[columns_order]
    return df