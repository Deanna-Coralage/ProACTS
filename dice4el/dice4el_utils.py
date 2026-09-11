import numpy as np
import pandas as pd

import torch

from sklearn.preprocessing import RobustScaler

from typing import Dict, List, Tuple, Optional

from config.feature_config import FeatureConfig



def _compute_scaled_continuous_feature_configs(
    feature_config:  FeatureConfig,
    scaler:          RobustScaler,
) -> Dict[str, np.ndarray]:
    
    """
    Compute permitted continuous-feature bounds in RobustScaler space.
    -----
    RobustScaler applies:
        x_scaled = (x - center_) / scale_
    Therefore:
        scaled_range = (hi - lo) / scale_
    """
    
    scaler_features = tuple(feature_config.continuous_features)

    total_features = len(feature_config.feature_order)

    scaled_lower_arr = np.zeros(total_features, dtype=np.float32)
    scaled_upper_arr = np.zeros(total_features, dtype=np.float32)
    scaled_norm_arr = np.ones(total_features, dtype=np.float32)

    scaler_feature_to_idx = {
        feature: idx
        for idx, feature in enumerate(scaler_features)
    }

    if not hasattr(scaler, "scale_"):
        raise ValueError(
            "The fitted scaler does not expose scale_. "
            "This helper expects a fitted RobustScaler."
        )

    if len(scaler.scale_) != len(scaler_features):
        raise ValueError(
            "scaler.scale_ and scaler_features are misaligned: "
            f"{len(scaler.scale_)=}, {len(scaler_features)=}."
        )

    if hasattr(scaler, "center_") and scaler.center_ is not None:
        scaler_center = np.asarray(scaler.center_, dtype=np.float64)
    else:
        # RobustScaler(with_centering=False)
        scaler_center = np.zeros(len(scaler_features), dtype=np.float64)

    scaler_scale = np.asarray(scaler.scale_, dtype=np.float64)

    for feature in feature_config.continuous_features:
        spec = feature_config.feature_specs.get(feature)

        if spec is None or spec.get("permitted_range") is None:
            raise ValueError(
                f"No permitted range provided for continuous feature "
                f"'{feature}'."
            )

        if feature not in scaler_feature_to_idx:
            raise ValueError(
                f"Continuous feature '{feature}' is missing from "
                "scaler_features."
            )

        global_idx = feature_config.feature_to_idx[feature]
        scaler_idx = scaler_feature_to_idx[feature]

        lo, hi = map(float, spec["permitted_range"])

        if lo > hi:
            raise ValueError(
                f"Invalid permitted range for '{feature}': "
                f"({lo}, {hi})."
            )

        scale = float(scaler_scale[scaler_idx])
        center = float(scaler_center[scaler_idx])

        if not np.isfinite(scale) or abs(scale) < 1e-8:
            raise ValueError(
                f"Invalid RobustScaler scale for '{feature}': {scale}."
            )

        scaled_lo = (lo - center) / scale
        scaled_hi = (hi - center) / scale

        # Defensive ordering in case a custom scaler has negative scale.
        lower = min(scaled_lo, scaled_hi)
        upper = max(scaled_lo, scaled_hi)
        scaled_range = max(upper - lower, 1e-8)

        scaled_lower_arr[global_idx] = np.float32(lower)
        scaled_upper_arr[global_idx] = np.float32(upper)
        scaled_norm_arr[global_idx] = np.float32(scaled_range)

    return {
        "scaled_lower_arr": scaled_lower_arr,
        "scaled_upper_arr": scaled_upper_arr,
        "scaled_norm_arr": scaled_norm_arr,
    }



def _get_split_scaled_continuous_configs(
    feature_config:    FeatureConfig,
    scaled_lower_arr:  np.ndarray,
    scaled_upper_arr:  np.ndarray,
    scaled_norm_arr:   np.ndarray,
) -> Dict[str, np.ndarray]:

    dynamic_features = [
        f for f in feature_config.feature_order
        if f in feature_config.dynamic_continuous_features
    ]
    static_features = [
        f for f in feature_config.feature_order
        if f in feature_config.static_continuous_features
    ]

    def extract(arr, features):
        return np.asarray(
            [
                arr[feature_config.feature_to_idx[f]]
                for f in features
            ],
            dtype=np.float32,
        )

    return {
        "dynamic_cont_lower": extract(scaled_lower_arr, dynamic_features),
        "dynamic_cont_upper": extract(scaled_upper_arr, dynamic_features),
        "static_cont_lower": extract(scaled_lower_arr, static_features),
        "static_cont_upper": extract(scaled_upper_arr, static_features),
        "dynamic_cont_scaled_norm": extract(scaled_norm_arr, dynamic_features),
        "static_cont_scaled_norm": extract(scaled_norm_arr, static_features),
    }



def _scaled_categorical_distance(
    dyn_cat_ohe:               List[torch.Tensor],
    dyn_cat_0:                 List[torch.Tensor],
    static_cat_ohe:            List[torch.Tensor],
    static_cat_0:              List[torch.Tensor],
    mask:                      Optional[torch.Tensor] = None,
    dyn_cat_update_masks:      Optional[Dict[int, torch.Tensor]] = None,
    static_cat_update_masks:   Optional[Dict[int, torch.Tensor]] = None,
) -> torch.Tensor:
    
    """
    Mean categorical distance over mutable categorical feature cells.

    Dynamic categorical features are counted once per allowed,
    valid sequence position.
    Static categorical features are counted once per allowed
    feature and candidate.

    Same one-hot category      -> 0
    Different one-hot category -> 1
    """

    dyn_cat_update_masks = dyn_cat_update_masks or {}
    static_cat_update_masks = static_cat_update_masks or {}

    if dyn_cat_ohe:
        reference = dyn_cat_ohe[0]
    elif static_cat_ohe:
        reference = static_cat_ohe[0]
    else:
        raise ValueError(
            "No categorical tensors were provided."
        )

    distance_sum = reference.new_tensor(0.0)
    valid_count = reference.new_tensor(0.0)

    # --- Dynamic categorical ---
    for feature_idx, (current, original) in enumerate(
        zip(dyn_cat_ohe, dyn_cat_0)
    ):
        update_mask = dyn_cat_update_masks.get(feature_idx)

        # Feature is not mutable.
        if update_mask is None:
            continue

        per_instance = (
            current - original
        ).abs().sum(dim=-1) / 2.0  # (B, T)

        # (B, T, K) -> (B, T)
        valid = (
            update_mask
            .bool()
            .any(dim=-1)
            .to(
                device=per_instance.device,
                dtype=per_instance.dtype,
            )
        )

        # Remove padded positions.
        if mask is not None:
            valid = valid * mask.to(
                device=per_instance.device,
                dtype=per_instance.dtype,
            )

        distance_sum += (
            per_instance * valid
        ).sum()

        valid_count += valid.sum()

    # --- Static categorical ---
    for feature_idx, (current, original) in enumerate(
        zip(static_cat_ohe, static_cat_0)
    ):
        update_mask = static_cat_update_masks.get(feature_idx)

        # Feature is not mutable.
        if update_mask is None:
            continue

        per_instance = (
            current - original
        ).abs().sum(dim=-1) / 2.0  # (B,)

        # (B, K) -> (B,)
        valid = (
            update_mask
            .bool()
            .any(dim=-1)
            .to(
                device=per_instance.device,
                dtype=per_instance.dtype,
            )
        )

        distance_sum += (
            per_instance * valid
        ).sum()

        valid_count += valid.sum()

    if valid_count.item() == 0:
        return reference.new_tensor(0.0)

    return distance_sum / valid_count.clamp_min(1e-8)



def _scaled_continuous_distance(
    current:      torch.Tensor,
    original:     torch.Tensor,
    scaled_norm:  torch.Tensor,
    update_mask:  Optional[torch.Tensor] = None,
    mask:         Optional[torch.Tensor] = None,
    eps:          float = 1e-8,
) -> torch.Tensor:
    
    """
    Log-compressed, range-normalized L1 distance in scaled space.

    The mean is calculated only over mutable continuous cells.
    For dynamic continuous tensors, padded positions are excluded.
    """

    if current.numel() == 0 or update_mask is None:
        return current.new_tensor(0.0)

    if current.shape != original.shape:
        raise ValueError(
            "Current and original tensors must have the same shape: "
            f"{current.shape} != {original.shape}."
        )

    if update_mask.shape != current.shape:
        raise ValueError(
            "update_mask must have the same shape as current: "
            f"{update_mask.shape} != {current.shape}."
        )

    if scaled_norm.ndim != 1:
        raise ValueError(
            "scaled_norm must be one-dimensional, "
            f"got {scaled_norm.shape}."
        )

    if current.shape[-1] != scaled_norm.numel():
        raise ValueError(
            "Continuous-feature alignment mismatch: "
            f"{current.shape[-1]=}, {scaled_norm.numel()=}."
        )

    if current.ndim not in {2, 3}:
        raise ValueError(
            "Continuous tensors must have shape (B, D) or "
            f"(B, T, D), got {current.shape}."
        )

    valid = update_mask.to(
        device=current.device,
        dtype=current.dtype,
    )

    # Dynamic continuous: (B, T, D)
    if current.ndim == 3 and mask is not None:
        if mask.shape != current.shape[:2]:
            raise ValueError(
                "Sequence mask must have shape (B, T): "
                f"{mask.shape} != {current.shape[:2]}."
            )

        valid = valid * mask.to(
            device=current.device,
            dtype=current.dtype,
        ).unsqueeze(-1)

    # Static continuous should not receive a sequence mask.
    if current.ndim == 2 and mask is not None:
        raise ValueError(
            "A sequence mask should only be supplied for dynamic "
            "continuous tensors."
        )

    view_shape = [1] * (current.ndim - 1) + [-1]

    max_scaled_diff = (
        scaled_norm
        .to(
            device=current.device,
            dtype=current.dtype,
        )
        .view(*view_shape)
        .clamp_min(eps)
    )

    scaled_diff = (
        current - original
    ).abs()

    normalized = (
        torch.log1p(scaled_diff)
        / (torch.log1p(max_scaled_diff) + eps)
    ).clamp(0.0, 1.0)

    valid_count = valid.sum()

    if valid_count.item() == 0:
        return current.new_tensor(0.0)

    distance_sum = (
        normalized * valid
    ).sum()

    return (
        distance_sum
        / valid_count.clamp_min(eps)
    )



def _compute_local_split_vary_indices(
    feature_config: FeatureConfig,
) -> Dict[str, np.ndarray]:

    dynamic_cat_features = [f for f in feature_config.feature_order if f in feature_config.dynamic_categorical_features]
    static_cat_features = [f for f in feature_config.feature_order if f in feature_config.static_categorical_features]
    dynamic_cont_features = [f for f in feature_config.feature_order if f in feature_config.dynamic_continuous_features]
    static_cont_features = [f for f in feature_config.feature_order if f in feature_config.static_continuous_features]

    dynamic_cat_to_local = {
        feature: idx
        for idx, feature in enumerate(dynamic_cat_features)
    }

    static_cat_to_local = {
        feature: idx
        for idx, feature in enumerate(static_cat_features)
    }

    dynamic_cont_to_local = {
        feature: idx
        for idx, feature in enumerate(dynamic_cont_features)
    }

    static_cont_to_local = {
        feature: idx
        for idx, feature in enumerate(static_cont_features)
    }

    return {
        "split_vary_dynamic_cat_idx": np.asarray(
            [dynamic_cat_to_local[f] for f in feature_config.categorical_to_vary if f in dynamic_cat_to_local],
            dtype=np.int64,
        ),
        "split_vary_static_cat_idx": np.asarray(
            [static_cat_to_local[f] for f in feature_config.categorical_to_vary if f in static_cat_to_local],
            dtype=np.int64,
        ),
        "split_vary_dynamic_cont_idx": np.asarray(
            [dynamic_cont_to_local[f] for f in feature_config.continuous_to_vary if f in dynamic_cont_to_local],
            dtype=np.int64,
        ),
        "split_vary_static_cont_idx": np.asarray(
            [static_cont_to_local[f] for f in feature_config.continuous_to_vary if f in static_cont_to_local],
            dtype=np.int64,
        ),
    }



GradientMask = Tuple[
        torch.Tensor,  # variable
        torch.Tensor,  # update mask
        torch.Tensor,  # original values
    ]
