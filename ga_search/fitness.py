import numpy as np

import warnings

import copy

from typing import List, Dict, Set, Optional

from process.constraint import ProcessConstraint



# --- Utils ---
def _precompute_change_mask(
    n:                  int,
    seq_len:            int,
    n_features:         int,
    positions:          List[int],
    vary_static_idx:    np.array,
    vary_dynamic_idx:   np.array,
) -> np.ndarray: 

    # --- Mask ---
    change_mask = np.zeros((n, seq_len, n_features), dtype=bool)  # Static and dynamic, cat and cont changed features together

    # --- Static features: mark as changed at position 0 ---
    change_mask[:, 0, vary_static_idx] = True

    # --- Dynamic features: only selected positions ---
    for p in positions:
        change_mask[:, p, vary_dynamic_idx] = True

    return change_mask



# --- Fitness components ---
def _calculate_distance_for_continuous_features(
    trace:                  np.ndarray,                        # (seq_len, n_features)
    cf_pop:                 np.ndarray,                        # (n, seq_len, n_features)
    vary_cont_idx:          np.ndarray,                        # (n_cont,)
    mad_arr:                np.ndarray,
    norm_arr:               np.ndarray,
    change_mask:            np.ndarray,                        # (n, seq_len, n_features)
) -> np.ndarray:                                               # (n,) 

    """
    Range-normalized L1 proximity over continuous features.
    Only considers positions marked in change_mask.
    """

    N = cf_pop.shape[0]

    if vary_cont_idx is None or len(vary_cont_idx) == 0:
        return np.zeros(N, dtype=np.float64)

    cf_vals = cf_pop[:, :, vary_cont_idx].astype(np.float32)
    x_vals = trace[:, vary_cont_idx][None, :, :].astype(np.float32)
    mask = change_mask[:, :, vary_cont_idx].astype(np.float32)

    # --- Compute feature ranges ---
    mad = mad_arr[vary_cont_idx]
    norm = norm_arr[vary_cont_idx]

    # --- Masked normalized L1 difference ---
    raw_diff = np.abs(cf_vals - x_vals) / mad[None, None, :]
    raw_max = norm / mad

    diff = np.log1p(raw_diff) / np.log1p(raw_max[None, None, :])
    diff = np.clip(diff, 0.0, 1.0)
    
    diff = diff * mask

    diff_sum = diff.sum(axis=(1, 2))
    valid_count = mask.sum(axis=(1, 2)) + 1e-8
    
    return (diff_sum / valid_count).astype(np.float64)


def _calculate_distance_for_categorical_features(
    trace:          np.ndarray,       # (seq_len, n_features) 
    cf_pop:         np.ndarray,       # (n, seq_len, n_features)   
    vary_cat_idx:   np.ndarray,       # (n_cat,) 
    change_mask:    np.ndarray,       # (n, seq_len, n_features)
) -> np.ndarray:                      # (n,) 
    
    """
    Vectorized mismatch proximity for categorical features.
    proximity_i = mean over changed categorical feature values of:
                  1(cf[pos, col] != x[pos, col]) -> Hamming distance
    """
    
    N = cf_pop.shape[0]

    if vary_cat_idx is None or len(vary_cat_idx) == 0:
        return np.zeros(N, dtype=np.float64)

    cf_vals = cf_pop[:, :, vary_cat_idx]
    x_vals = trace[:, vary_cat_idx][None, :, :]
    mask = change_mask[:, :, vary_cat_idx]

    # --- Mismatch = 1 if different, else 0 ---
    mismatch = (cf_vals != x_vals).astype(np.float32)
    mismatch = mismatch * mask

    # --- Per-batch across cat aggregation ---
    mismatch_sum = mismatch.sum(axis=(1, 2))      # (n,)
    valid_count = mask.sum(axis=(1, 2)) + 1e-8    # (n,)

    return (mismatch_sum / valid_count).astype(np.float64)
    

def _calculate_sparsity(
    trace:         np.ndarray, 
    cf_pop:        np.ndarray,  
    change_mask:   np.ndarray 
) -> np.ndarray:   # (n,) 

    """
    Computes sparsity = number of changed elements per sample.
    """
    
    changed = np.not_equal(cf_pop, trace[None, :, :])
    effective_changes = changed & change_mask

    mismatch_sum = effective_changes.sum(axis=(1, 2))
    valid_count = change_mask.sum(axis=(1, 2)) + 1e-8

    sparsity = mismatch_sum / valid_count
    
    return sparsity.astype(np.float64)
    

def _calculate_margin_loss(
    logits_store:         List[Dict[int, Dict[str, float]]],
    desired_constraints:  ProcessConstraint,
    confidence_ratio:     float = 2.0,
) -> np.ndarray:

    N = len(logits_store)
    margin_losses = np.zeros(N, dtype=np.float64)

    desired_pos = desired_constraints.get_constrained_positions()
    if not desired_pos:
        return margin_losses

    margin = np.log(confidence_ratio)

    for n in range(N):
        desired_state = copy.deepcopy(desired_constraints)

        total_loss = 0.0
        checked = 0

        for t in desired_pos:
            if t not in logits_store[n]:
                continue

            logits_dict = logits_store[n][t]

            # Find active desired segment covering this position
            matched_key = None
            for key in list(desired_state.segments.keys()):
                start, end = key
                if start <= t <= end:
                    matched_key = key
                    break

            if matched_key is None:
                continue

            desired_classes = set(desired_state.segments[matched_key])

            labels = list(logits_dict.keys())
            logits = np.array([logits_dict[label] for label in labels], dtype=np.float64)
            label_to_idx = {label: idx for idx, label in enumerate(labels)}

            desired_indices = [
                label_to_idx[c]
                for c in desired_classes
                if c in label_to_idx
            ]

            if not desired_indices:
                total_loss += 1.0
                checked += 1
                continue

            desired_logit = logits[desired_indices].max()

            tmp = logits.copy()
            tmp[desired_indices] = -np.inf
            best_other = tmp.max()

            if np.isfinite(best_other):
                gap = desired_logit - best_other
                violation = max(0.0, margin - gap)
                total_loss += violation / (violation + margin + 1e-8)

            checked += 1

            best_desired_idx = desired_indices[int(np.argmax(logits[desired_indices]))]
            best_desired_class = labels[best_desired_idx]
            
            desired_state.check_constrained_position(t, best_desired_class)

        if checked > 0:
            margin_losses[n] = total_loss / checked

    return margin_losses.astype(np.float64)



# --- Process-aware fitness ---
def _calculate_process_violation(
    trace:                        np.ndarray,                    # (seq_len, n_features)
    simulated_cfs:                np.ndarray,                    # (n, seq_len, n_features)
    desired_constraints:          ProcessConstraint,
    flexible_constraints:         ProcessConstraint,
    activity_idx:                 int,
    penalty_desired_nonflip:      float = 3.0,
    penalty_desired_wrongflip:    float = 5.0,
    penalty_undesired_flip:       float = 2.0,
    penalty_flexible_wrongflip:   float = 2.0,
) -> np.ndarray:                  # (n, )

    """
    Batched process-consistency violation. Lower is better.

    Priority:
        1. Desired position
        2. Genuine flexible position
        3. Baseline simulation-mismatch position
        4. Ordinary position

    Desired:
        - Correct desired flip: no penalty
        - Original activity retained: desired_nonflip penalty
        - Changed to wrong activity: desired_wrongflip penalty

    Flexible:
        - Original or valid flexible flip: no penalty
        - Changed to invalid activity: flexible_wrongflip penalty

    Simulation mismatch:
        - Original or baseline-simulated activity: no penalty
        - Changed to another activity: mismatch_wrongflip penalty

    Ordinary:
        - Any activity change: undesired_flip penalty
    """

    N, T, _ = simulated_cfs.shape

    process_violation = np.zeros(N, dtype=np.float64)

    # --- Object-Level Batch Isolation ---
    batch_desired: List[ProcessConstraint] = [copy.deepcopy(desired_constraints) for _ in range(N)]
    batch_flexible: List[ProcessConstraint] = [copy.deepcopy(flexible_constraints) for _ in range(N)]

    # --- Calculate worst-case penalty bounds ---
    D_pos = set(desired_constraints.get_constrained_positions())
    F_pos = set(flexible_constraints.get_constrained_positions())
    
    # if desired has priority, remove desired from flexible count
    F_only = F_pos - D_pos
    U_pos = set(range(T)) - D_pos - F_pos
    
    D_initial = len(D_pos)
    F_initial = len(F_only)
    U_initial = len(U_pos)

    # --- Chronological Timeline Evaluation ---
    for t in range(T):
        
        original_act = str(trace[t, activity_idx])
        simulated_acts = simulated_cfs[:, t, activity_idx].astype(str)

        # Vectorized check for changes across the entire batch at step t
        has_flipped = (simulated_acts != original_act)

        # Row loop replaced completely by an optimized zip-comprehension pass
        step_results = [
            (
                d_obj.check_constrained_position(t, act),
                f_obj.check_constrained_position(t, act)
            )
            for d_obj, f_obj, act in zip(batch_desired, batch_flexible, simulated_acts)
        ]

        # Vectorized Penalty Accumulation Matrix Pass
        for i, ((is_desired, desired_ok, _), (is_flex, flex_ok, _)) in enumerate(step_results):
            if is_desired:
                if not desired_ok:
                    process_violation[i] += penalty_desired_wrongflip if has_flipped[i] else penalty_desired_nonflip
            elif is_flex:
                if not flex_ok and has_flipped[i]:
                    process_violation[i] += penalty_flexible_wrongflip
            elif has_flipped[i]:
                process_violation[i] += penalty_undesired_flip

    # Worst-Case Bound Normalisation
    max_penalty = (D_initial * penalty_desired_wrongflip) + \
                  (F_initial * penalty_flexible_wrongflip) + \
                  (U_initial * penalty_undesired_flip)

    if max_penalty < 1e-6:
        max_penalty = 1.0

    return (process_violation / max_penalty).astype(np.float64)



# --- Full fitness calculation per each counterfactual ---
def calculate_fitness(
    trace:                        np.ndarray,
    cf_pop:                       np.ndarray,
    simulated_cfs:                np.ndarray,
    positions:                    List[int],
    
    logits_store:                 List[Dict[int, Dict[str, float]]],
    logits_store_na:              List[Dict[int, Dict[str, float]]],
    
    desired_constraints:          ProcessConstraint,
    flexible_constraints:         ProcessConstraint,
    
    activity_idx:                 int,
    vary_static_idx:              np.ndarray,
    vary_dynamic_idx:             np.ndarray,
    vary_cont_idx:                np.ndarray,
    vary_cat_idx:                 np.ndarray,
    mad_arr:                      np.ndarray,
    norm_arr:                     np.ndarray,

    confidence_ratio:             float = 2.0,
    
    penalty_desired_nonflip:      float = 3.0,
    penalty_desired_wrongflip:    float = 5.0,
    penalty_undesired_flip:       float = 2.0,
    penalty_flexible_wrongflip:   float = 2.0,
    
    w_distance:                   float = 1.0,  
    w_sparsity:                   float = 1.0,  
    w_margin:                     float = 1.0,
    w_process_violation:          float = 1.0,
) -> Dict[str, np.ndarray]:
    
    """
    Full GA fitness (lower is better).
    """

    change_mask = _precompute_change_mask(
        n=cf_pop.shape[0],
        seq_len=cf_pop.shape[1],
        n_features=cf_pop.shape[2],
        positions=positions,
        vary_static_idx=vary_static_idx,
        vary_dynamic_idx=vary_dynamic_idx
    )
    
    # --- Components ---
    cont_distance = _calculate_distance_for_continuous_features(
        trace=trace,
        cf_pop=cf_pop,
        vary_cont_idx=vary_cont_idx,
        mad_arr=mad_arr,
        norm_arr=norm_arr,
        change_mask=change_mask
    )
    cat_distance = _calculate_distance_for_categorical_features(
        trace=trace,
        cf_pop=cf_pop,
        vary_cat_idx=vary_cat_idx,
        change_mask=change_mask
    )
    distance = (cont_distance + cat_distance) / 2

    sparsity = _calculate_sparsity(
        trace=trace,
        cf_pop=cf_pop,
        change_mask=change_mask
    )

    margin_loss = _calculate_margin_loss(
        logits_store=logits_store, 
        desired_constraints=desired_constraints,
        confidence_ratio=confidence_ratio,
    )

    margin_loss_na = _calculate_margin_loss(
        logits_store=logits_store_na, 
        desired_constraints=desired_constraints,
        confidence_ratio=confidence_ratio,
    )

    margin_loss_flipped = _calculate_margin_loss(
        logits_store=logits_store, 
        desired_constraints=desired_constraints,
        confidence_ratio=1.000001
    )

    margin_loss_na_flipped = _calculate_margin_loss(
        logits_store=logits_store_na, 
        desired_constraints=desired_constraints,
        confidence_ratio=1.000001
    )
    
    process_violation = _calculate_process_violation(
        trace=trace,
        simulated_cfs=simulated_cfs,
        desired_constraints=desired_constraints,
        flexible_constraints=flexible_constraints,
        activity_idx=activity_idx,
        penalty_desired_nonflip=penalty_desired_nonflip,
        penalty_desired_wrongflip=penalty_desired_wrongflip,
        penalty_undesired_flip=penalty_undesired_flip,
        penalty_flexible_wrongflip=penalty_flexible_wrongflip
    )

    # --- Weighted fitness ---
    fitness = (
        w_distance * distance +
        w_sparsity * sparsity +
        w_margin * margin_loss +
        w_process_violation * process_violation
    ).astype(np.float64)

    return {
        "fitness": fitness,
        "process_violation": process_violation,
        "distance": distance,
        "cat_distance": cat_distance,
        "cont_distance": cont_distance,
        "sparsity": sparsity,
        "margin_loss": margin_loss,
        "margin_loss_na": margin_loss_na,
        "margin_loss_flipped": margin_loss_flipped,
        "margin_loss_na_flipped": margin_loss_na_flipped
    }



# --- Pairwise diversity (for niche sharing when selecting, and diversity calculation) ---
def _calculate_pairwise_distance_matrix_for_continuous_features(
    cf_pop:                 np.ndarray,     # (n, seq_len, n_features)
    vary_cont_idx:          np.ndarray,
    mad_arr:                np.ndarray,
    norm_arr:               np.ndarray,
    change_mask:            np.ndarray,
) -> np.ndarray:                           # (n, n)
    
    """
    Returns pairwise continuous distance matrix between the cfs in the population.
    """

    N = cf_pop.shape[0]

    if N <= 1 or vary_cont_idx is None or len(vary_cont_idx) == 0:
        return np.zeros((N, N), dtype=np.float64)

    x = cf_pop[:, :, vary_cont_idx].astype(np.float32)
    mask = change_mask[:, :, vary_cont_idx].astype(np.float32)

    mad = mad_arr[vary_cont_idx]
    norm = norm_arr[vary_cont_idx]

    raw_diff = np.abs(x[:, None, :, :] - x[None, :, :, :])
    raw_diff = raw_diff / mad[None, None, None, :]
    raw_max = norm / mad

    diff = np.log1p(raw_diff) / np.log1p(raw_max[None, None, None, :])
    diff = np.clip(diff, 0.0, 1.0)
    
    valid = mask[:, None, :, :] * mask[None, :, :, :]
    diff = diff * valid

    diff_sum = diff.sum(axis=(2, 3))
    valid_count = valid.sum(axis=(2, 3)) + 1e-8

    return (diff_sum / valid_count).astype(np.float64)


def _calculate_pairwise_distance_matrix_for_categorical_features(
    cf_pop:         np.ndarray,
    vary_cat_idx:   np.ndarray,
    change_mask:    np.ndarray,
) -> np.ndarray:    # (n, n)

    N = cf_pop.shape[0]

    if N <= 1 or vary_cat_idx is None or len(vary_cat_idx) == 0:
        return np.zeros((N, N), dtype=np.float64)

    x = cf_pop[:, :, vary_cat_idx]
    mask = change_mask[:, :, vary_cat_idx]
    
    mismatch = (x[:, None, :, :] != x[None, :, :, :]).astype(np.float32)
    valid = mask[:, None, :, :] * mask[None, :, :, :]
    mismatch = mismatch * valid

    mismatch_sum = mismatch.sum(axis=(2, 3))
    valid_count = valid.sum(axis=(2, 3)) + 1e-8

    return (mismatch_sum / valid_count).astype(np.float64)


def calculate_pairwise_distance_matrix(
    cf_pop:                  np.ndarray, 
    vary_cont_idx:           np.ndarray,
    vary_cat_idx:            np.ndarray,
    mad_arr:                 np.ndarray,
    norm_arr:                np.ndarray,
    
    # --- Option A: direct mask ---
    change_mask:             Optional[np.ndarray] = None,

    # --- Option B: construct mask ---
    positions:               Optional[List[int]] = None,
    vary_static_idx:         Optional[np.ndarray] = None,
    vary_dynamic_idx:        Optional[np.ndarray] = None,
) -> np.ndarray:             # (n, n)
    
    """
    Returns pairwise distance matrix between the cfs in the population.
    """

    if change_mask is None:

        if positions is None or vary_static_idx is None or vary_dynamic_idx is None:
            raise ValueError(
                "Provide either change_mask OR (positions, vary_static_idx, vary_dynamic_idx)"
            )
        
        change_mask = _precompute_change_mask(
            n=cf_pop.shape[0],
            seq_len=cf_pop.shape[1],
            n_features=cf_pop.shape[2],
            positions=positions,
            vary_static_idx=vary_static_idx,
            vary_dynamic_idx=vary_dynamic_idx
        )

    D_cont = _calculate_pairwise_distance_matrix_for_continuous_features(
        cf_pop=cf_pop,
        vary_cont_idx=vary_cont_idx,
        mad_arr=mad_arr,
        norm_arr=norm_arr,
        change_mask=change_mask
    ) 

    D_cat = _calculate_pairwise_distance_matrix_for_categorical_features(
        cf_pop=cf_pop,
        vary_cat_idx=vary_cat_idx,
        change_mask=change_mask
    )

    D = (D_cont + D_cat) / 2

    return D.astype(np.float64)



# --- Evaluate all counterfactuals - components ---
def _calculate_diversity(
    cf_pop:                  np.ndarray,    # (n, seq_len, n_features)
    vary_cont_idx:           np.ndarray,
    vary_cat_idx:            np.ndarray,
    mad_arr:                 np.ndarray,
    norm_arr:                np.ndarray,
    change_mask:             np.ndarray,
) -> np.ndarray:             # (n*n-1)       

    """
    Computes diversity = average pairwise distance between the cfs in the population.
    """
    
    N = cf_pop.shape[0]

    if N <= 1:
        return 0.0
    
    D = calculate_pairwise_distance_matrix(
        cf_pop=cf_pop,
        vary_cont_idx=vary_cont_idx,
        vary_cat_idx=vary_cat_idx,
        mad_arr=mad_arr,
        norm_arr=norm_arr,
        change_mask=change_mask
    )

    iu = np.triu_indices(N, k=1)

    return D[iu].astype(np.float64)


def _calculate_continuous_implausibility(
    cf_pop:                  np.ndarray, 
    train_arr:               np.ndarray,  
    vary_cont_idx:           np.ndarray,
    mad_arr:                 np.ndarray,
    norm_arr:                np.ndarray,
    change_mask:             np.ndarray,  
) -> np.ndarray:             # (n)

    """
    Imlausibility — distance to nearest training instance
    at changed positions.
    (n,) float64 — implausibility score, lower = better
    """
    
    N = cf_pop.shape[0]
    M = train_arr.shape[0]    

    if vary_cont_idx is None or len(vary_cont_idx) == 0:
        return np.zeros((N, M), dtype=np.float64)

    cf_cont = cf_pop[:, :, vary_cont_idx].astype(np.float32)
    tr_cont = train_arr[:, :, vary_cont_idx].astype(np.float32)
    mask = change_mask[:, :, vary_cont_idx].astype(np.float32)

    mad = mad_arr[vary_cont_idx]
    norm = norm_arr[vary_cont_idx]

    raw_diff = np.abs(cf_cont[:, None, :, :] - tr_cont[None, :, :, :])
    raw_diff = raw_diff / mad[None, None, None, :]
    raw_max = norm / mad

    diff = np.log1p(raw_diff) / np.log1p(raw_max[None, None, :])
    diff = np.clip(diff, 0.0, 1.0)
    
    diff = diff * mask[:, None, :, :]

    diff_sum = diff.sum(axis=(2, 3))
    valid_count = np.repeat(mask.sum(axis=(1, 2))[:, None], M, axis=1) + 1e-8

    return (diff_sum / valid_count).astype(np.float64)
    

def _calculate_categorical_implausibility(
    cf_pop:          np.ndarray, 
    train_arr:       np.ndarray,  
    vary_cat_idx:    np.ndarray,
    change_mask:     np.ndarray,  
) -> np.ndarray:     # (n)

    N = cf_pop.shape[0]
    M = train_arr.shape[0]    

    if vary_cat_idx is None or len(vary_cat_idx) == 0:
        return np.zeros((N, M), dtype=np.float64)

    cf_cat = cf_pop[:, :, vary_cat_idx] 
    tr_cat = train_arr[:, :, vary_cat_idx]
    mask = change_mask[:, :, vary_cat_idx].astype(np.float32)

    mismatch = (cf_cat[:, None, :, :] != tr_cat[None, :, :, :]).astype(np.float32)
    mismatch = mismatch * mask[:, None, :, :]
    
    mismatch_sum = mismatch.sum(axis=(2, 3))
    valid_count = np.repeat(mask.sum(axis=(1, 2))[:, None], M, axis=1) + 1e-8

    return (mismatch_sum / valid_count).astype(np.float64)
    

def _calculate_approx_implausibility(
    cf_pop:                  np.ndarray, 
    train_arr:               np.ndarray,  
    vary_cont_idx:           np.ndarray,
    vary_cat_idx:            np.ndarray,
    mad_arr:                 np.ndarray,
    norm_arr:                np.ndarray,
    change_mask:             np.ndarray, 
    sample_ratio:            float = 1.0,
) -> np.ndarray:             # (n)

    """
    Approximate implausibility using random subset of training traces.
    """

    M = train_arr.shape[0]
    
    # --- 1: random sampling of training data ---
    sample_size = max(1, int(M * sample_ratio))

    idx = np.random.choice(M, size=sample_size, replace=False)
    train_sub = train_arr[idx]

    # --- 2: call full implausibility on subset ---
    cont_dist  = _calculate_continuous_implausibility(
        cf_pop=cf_pop,
        train_arr=train_sub,
        change_mask=change_mask, 
        vary_cont_idx=vary_cont_idx,
        mad_arr=mad_arr,
        norm_arr=norm_arr,
    )

    cat_dist = _calculate_categorical_implausibility(
        cf_pop=cf_pop,
        train_arr=train_sub,
        change_mask=change_mask, 
        vary_cat_idx=vary_cat_idx,
    )

    distance = (cont_dist + cat_dist) / 2
    implausibility = distance.min(axis=1)

    return implausibility.astype(np.float64)



# --- Full fitness calculation per each counterfactual ---
def evaluate_cf_pop(
    trace:                        np.ndarray,
    cf_pop:                       np.ndarray,
    simulated_cfs:                np.ndarray,
    train_arr:                    np.ndarray,
    positions:                    List[int],
    
    logits_store:                 List[Dict[int, Dict[str, float]]],
    logits_store_na:              List[Dict[int, Dict[str, float]]],
    
    desired_constraints:          ProcessConstraint,
    flexible_constraints:         ProcessConstraint,
    
    activity_idx:                 int,
    vary_static_idx:              np.ndarray,
    vary_dynamic_idx:             np.ndarray,
    vary_cont_idx:                np.ndarray,
    vary_cat_idx:                 np.ndarray,
    mad_arr:                      np.ndarray,
    norm_arr:                     np.ndarray,

    confidence_ratio:             float = 2.0,

    penalty_desired_nonflip:      float = 3.0,
    penalty_desired_wrongflip:    float = 5.0,
    penalty_undesired_flip:       float = 2.0,
    penalty_flexible_wrongflip:   float = 2.0,

    w_distance:                   float = 1.0,  
    w_sparsity:                   float = 1.0,  
    w_margin:                     float = 1.0,
    w_process_violation:          float = 1.0,
    
    sample_ratio:                 float = 1.0,
) -> Dict[str, float]:
    
    """
    Full CF population evaluation (lower score = better CF).
    """
    
    change_mask = _precompute_change_mask(
        n=cf_pop.shape[0],
        seq_len=cf_pop.shape[1],
        n_features=cf_pop.shape[2],
        positions=positions,
        vary_static_idx=vary_static_idx,
        vary_dynamic_idx=vary_dynamic_idx
    )
    
    # --- Components ---
    cont_distance = _calculate_distance_for_continuous_features(
        trace=trace,
        cf_pop=cf_pop,
        vary_cont_idx=vary_cont_idx,
        mad_arr=mad_arr,
        norm_arr=norm_arr,
        change_mask=change_mask
    )
    cat_distance = _calculate_distance_for_categorical_features(
        trace=trace,
        cf_pop=cf_pop,
        vary_cat_idx=vary_cat_idx,
        change_mask=change_mask
    )
    distance = (cont_distance + cat_distance) / 2

    sparsity = _calculate_sparsity(
        trace=trace,
        cf_pop=cf_pop,
        change_mask=change_mask
    )

    margin_loss = _calculate_margin_loss(
        logits_store=logits_store, 
        desired_constraints=desired_constraints,
        confidence_ratio=confidence_ratio,
    )

    margin_loss_na = _calculate_margin_loss(
        logits_store=logits_store_na, 
        desired_constraints=desired_constraints,
        confidence_ratio=confidence_ratio,
    )

    margin_loss_flipped = _calculate_margin_loss(
        logits_store=logits_store, 
        desired_constraints=desired_constraints,
        confidence_ratio=1.000001
    )

    margin_loss_na_flipped = _calculate_margin_loss(
        logits_store=logits_store_na, 
        desired_constraints=desired_constraints,
        confidence_ratio=1.000001
    )
    
    process_violation = _calculate_process_violation(
        trace=trace,
        simulated_cfs=simulated_cfs,
        desired_constraints=desired_constraints,
        flexible_constraints=flexible_constraints,
        activity_idx=activity_idx,
        penalty_desired_nonflip=penalty_desired_nonflip,
        penalty_desired_wrongflip=penalty_desired_wrongflip,
        penalty_undesired_flip=penalty_undesired_flip,
        penalty_flexible_wrongflip=penalty_flexible_wrongflip
    )

    # --- Weighted fitness ---
    fitness = (
        w_distance * distance +
        w_sparsity * sparsity +
        w_margin * margin_loss +
        w_process_violation * process_violation
    ).astype(np.float64)

    implausibility = _calculate_approx_implausibility(
        cf_pop=cf_pop, 
        train_arr=train_arr,
        vary_cont_idx=vary_cont_idx,
        vary_cat_idx=vary_cat_idx,
        mad_arr=mad_arr,
        norm_arr=norm_arr,
        change_mask=change_mask,
        sample_ratio=sample_ratio
    )

    diversity = _calculate_diversity(
        cf_pop=cf_pop,
        vary_cont_idx=vary_cont_idx,
        vary_cat_idx=vary_cat_idx,
        mad_arr=mad_arr,
        norm_arr=norm_arr,
        change_mask=change_mask
    )

    population_summary = {
        "DISTANCE": float(np.mean(distance)),
        "CONT_DISTANCE": float(np.mean(cont_distance)),
        "CAT_DISTANCE": float(np.mean(cat_distance)),
        "SPARSITY": float(np.mean(sparsity)),
        "PROCESS_VIOLATION": float(np.mean(process_violation)),
        "MARGIN_LOSS": float(np.mean(margin_loss)),
        "MARGIN_LOSS_NA": float(np.mean(margin_loss_na)),
        "MARGIN_LOSS_FLIPPED": float(np.mean(margin_loss_flipped)),
        "MARGIN_LOSS_NA_FLIPPED": float(np.mean(margin_loss_na_flipped)),
        "FITNESS":  float(np.mean(fitness)),
        "IMPLAUSIBILITY": float(np.mean(implausibility)),
        "DIVERSITY": float(np.mean(diversity)),
    }

    return population_summary
