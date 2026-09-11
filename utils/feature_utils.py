import numpy as np
import pandas as pd

from typing import List, Dict, Tuple, Union, Optional

from config.feature_config import FeatureConfig



# --- Dataframe to array conversion ---
def df_to_sequence_array(
    df:               pd.DataFrame,
    case_id_field:    str,
    feature_config:   FeatureConfig,
    prefix_len:       Optional[int] = None,
    sort_field:       Optional[str] = None,
) -> np.ndarray:
    
    """
    Convert event log DataFrame into a 3D array:
        (num_cases, seq_len, n_features)

    Assumes:
    - all cases have identical sequence length, or longer than prefix_len to be truncated
    - rows are fully aligned per case
    """

    feature_order = list(feature_config.feature_order)

    if sort_field is not None:
        df = df.sort_values([case_id_field, sort_field])
    
    grouped = df.groupby(case_id_field, sort=False)
    sequences = []
    expected_len = None
    
    for case_id, group in grouped:

        if prefix_len is not None:
            if len(group) < prefix_len:
                continue

            group = group.iloc[:prefix_len]

        seq = group[feature_order].to_numpy()

        # --- Sanity check ---
        if expected_len is None:
            expected_len = len(seq)
        elif len(seq) != expected_len:
            raise ValueError(
                f"Inconsistent sequence length in case {case_id}: "
                f"{len(seq)} != {expected_len}"
            )

        sequences.append(seq)

    return np.stack(sequences, axis=0)



# --- Compute indices of features according to global feature order ---
def compute_vary_indices(
    feature_config:   FeatureConfig
) -> Dict[str, np.ndarray]: 

    # --- Feature indices ---
    activity_idx = feature_config.feature_to_idx[feature_config.activity_feature]
    
    vary_static_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.static_to_vary], dtype=int)
    vary_dynamic_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.dynamic_to_vary], dtype=int)
    
    vary_cont_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.continuous_to_vary], dtype=int)
    vary_cat_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.categorical_to_vary], dtype=int)

    vary_static_cont_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.static_continuous_to_vary], dtype=int)
    vary_static_cat_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.static_categorical_to_vary], dtype=int)

    vary_dynamic_cont_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.dynamic_continuous_to_vary], dtype=int)
    vary_dynamic_cat_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.dynamic_categorical_to_vary], dtype=int)

    return {
        "activity_idx": activity_idx,
        "vary_static_idx": vary_static_idx,
        "vary_dynamic_idx": vary_dynamic_idx,
        "vary_cont_idx": vary_cont_idx,
        "vary_cat_idx": vary_cat_idx,
        "vary_static_cont_idx": vary_static_cont_idx,
        "vary_static_cat_idx": vary_static_cat_idx,
        "vary_dynamic_cont_idx": vary_dynamic_cont_idx,
        "vary_dynamic_cat_idx": vary_dynamic_cat_idx
    }


def compute_all_indices(
    feature_config:   FeatureConfig
) -> Dict[str, np.ndarray]: 

    # --- Feature indices ---
    activity_idx = feature_config.feature_to_idx[feature_config.activity_feature]
    
    static_cont_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.static_continuous_features], dtype=int)
    static_cat_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.static_categorical_features], dtype=int)

    dynamic_cont_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.dynamic_continuous_features], dtype=int)
    dynamic_cat_idx = np.array([feature_config.feature_to_idx[f] for f in feature_config.dynamic_categorical_features], dtype=int)

    return {
        "activity_idx": activity_idx,
        "static_cont_idx": static_cont_idx,
        "static_cat_idx": static_cat_idx,
        "dynamic_cont_idx": dynamic_cont_idx,
        "dynamic_cat_idx": dynamic_cat_idx
    }


def compute_split_vary_indices(
    feature_config: FeatureConfig,
) -> Dict[str, np.ndarray]:

    # --- Global feature indices ---
    static_dyn_indices = compute_all_indices(feature_config)

    activity_idx = static_dyn_indices["activity_idx"]
    static_cont_idx = static_dyn_indices["static_cont_idx"]
    static_cat_idx = static_dyn_indices["static_cat_idx"]
    dynamic_cont_idx = static_dyn_indices["dynamic_cont_idx"]
    dynamic_cat_idx = static_dyn_indices["dynamic_cat_idx"]

    # --- Indices within split representations ---
    vary_split_static_cont_idx = np.array(
        [
            i 
            for i, f in enumerate(feature_config.static_continuous_features)
            if f in feature_config.static_continuous_to_vary
        ],
        dtype=int,
    )

    vary_split_static_cat_idx = np.array(
        [
            i
            for i, f in enumerate(feature_config.static_categorical_features)
            if f in feature_config.static_categorical_to_vary
        ],
        dtype=int,
    )

    vary_split_dynamic_cont_idx = np.array(
        [
            i
            for i, f in enumerate(feature_config.dynamic_continuous_features)
            if f in feature_config.dynamic_continuous_to_vary
        ],
        dtype=int,
    )

    vary_split_dynamic_cat_idx = np.array(
        [
            i
            for i, f in enumerate(feature_config.dynamic_categorical_features)
            if f in feature_config.dynamic_categorical_to_vary
        ],
        dtype=int,
    )

    return {
        "split_vary_static_cont_idx": vary_split_static_cont_idx,
        "split_vary_static_cat_idx": vary_split_static_cat_idx,
        "split_vary_dynamic_cont_idx": vary_split_dynamic_cont_idx,
        "split_vary_dynamic_cat_idx": vary_split_dynamic_cat_idx,
    }


def compute_feature_configs(
    feature_config:   FeatureConfig,
) -> Dict[
    str,
    Union[Dict[int, Tuple[float, float]], Dict[int, List[str]], Dict[int, str], np.ndarray]]:

    # --- Continuous features ---
    valid_ranges_per_idx = {}

    for f in feature_config.continuous_features:
        f_spec = feature_config.feature_specs.get(f)
        if f_spec is None or f_spec.get("permitted_range") is None:
            raise ValueError(
                f"No valid range provided for feature '{f}'"
            )

        idx = feature_config.feature_to_idx[f]
        valid_ranges_per_idx[idx] = f_spec["permitted_range"]

    # --- Categorical features ---
    valid_categories_per_idx = {}

    for f in feature_config.categorical_features:
        f_spec = feature_config.feature_specs.get(f)
        if f_spec is None or f_spec.get("valid_categories") is None:
            raise ValueError(
                f"No valid categories provided for feature '{f}'"
            )

        idx = feature_config.feature_to_idx[f]
        valid_categories_per_idx[idx] = f_spec["valid_categories"]

    # --- MAD arr for continuous ---
    total_features = len(feature_config.feature_order)
    mad_arr = np.ones(total_features, dtype=np.float32)
    mad_dict = feature_config.mad

    for f in feature_config.continuous_features:
        mad_val = mad_dict.get(f)
        if mad_val is None:
            raise ValueError(
                f"No mad for feature '{f}'"
            )
        idx = feature_config.feature_to_idx[f]
        mad_arr[idx] = np.float32(max(mad_val, 1e-8))

    # --- Normalization arr for cotinuous ---
    norm_arr = np.ones(total_features, dtype=np.float32)

    for idx, (lo, hi) in valid_ranges_per_idx.items():
        norm_arr[idx] = np.float32(max(hi - lo, 1e-8))

    # --- Feature strategies for continuous ---
    feature_strategies_per_idx = {}

    for idx, (lo, hi) in valid_ranges_per_idx.items():
        lo = float(lo)
        hi = float(hi)
        
        # --- Strategy Decision Matrix ---
        # If the feature spans more than 2 orders of magnitude, use log space
        if lo > 0 and hi > 0 and (hi / lo) > 100.0:
            feature_strategies_per_idx[idx] = "log"
        else:
            feature_strategies_per_idx[idx] = "linear"
       
    return {
        "valid_ranges_per_idx": valid_ranges_per_idx,
        "valid_categories_per_idx": valid_categories_per_idx,
        "feature_strategies_per_idx": feature_strategies_per_idx,
        "mad_arr": mad_arr,
        "norm_arr": norm_arr,
    }
