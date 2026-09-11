import numpy as np

from typing import Dict, List, Tuple



def _build_population_sparsity_mask(
    N:                    int,
    T:                    int,
    F:                    int,
    sparse_n:             int = 0,
    sparsity_feature_p:   float = 0.2,
) -> np.ndarray:

    sparse_idx = np.random.permutation(N)[:sparse_n]
    
    feature_mask = np.ones((N, F), dtype=bool)
    time_mask = np.ones((N, T, F), dtype=bool)

    feature_mask[sparse_idx] = (
        np.random.rand(len(sparse_idx), F) < sparsity_feature_p
    )

    for i in sparse_idx:
        time_mask[i] = (np.random.rand(T, F) < sparsity_feature_p)

    return feature_mask, time_mask



# --- Initialization strategies ---
def _rbi_initialization_strategy(
    trace:                      np.ndarray,
    positions:                  List[int],
    vary_static_cont_idx:       np.ndarray,
    vary_static_cat_idx:        np.ndarray,
    vary_dynamic_cont_idx:      np.ndarray,
    vary_dynamic_cat_idx:       np.ndarray,
    strategies_per_idx:         Dict[int, str],
    valid_ranges_per_idx:       Dict[int, Tuple[float, float]],
    valid_categories_per_idx:   Dict[int, List[str]],
    n:                          int,
    sparse_n:                   int = 0,
    sparsity_feature_p:         float = 0.2,
) -> np.ndarray:
    
    """
    Random-Based Initialization.
    Samples uniformly from permitted_range for continuous, randomly from valid_categories (encoded) for categorical.
    Maximum diversity, no class guidance.
    """

    population = np.tile(trace.copy(), (n, 1, 1)).astype(object)
    T, F = trace.shape
    pos_arr = np.asarray(positions, dtype=int)

    feature_mask, time_mask = _build_population_sparsity_mask(n, T, F, sparse_n, sparsity_feature_p)

    # --- Static continuous  ---
    for f in vary_static_cont_idx:
        lo, hi = valid_ranges_per_idx[f]
        strategy = strategies_per_idx.get(f, "linear")

        if strategy == "log":
            # Log-Uniform Initialization
            log_lo, log_hi = np.log(max(lo, 1e-5)), np.log(max(hi, 1e-5))
            noise = np.exp(np.random.uniform(log_lo, log_hi, size=(n, 1)))
        else:
            # Standard Linear Uniform Initialization
            noise = np.random.uniform(lo, hi, size=(n, 1))
            
        population[:, :, f] = np.where(feature_mask[:, f][:, None], noise, population[:, :, f])

    # --- Static categorical ---
    for f in vary_static_cat_idx:
        cats = np.asarray(valid_categories_per_idx[f])
        noise = np.random.choice(cats, size=(n, 1))
        population[:, :, f] = np.where(feature_mask[:, f][:, None], noise, population[:, :, f])

    # --- Dynamic continuous ---
    for f in vary_dynamic_cont_idx:
        lo, hi = valid_ranges_per_idx[f]
        strategy = strategies_per_idx.get(f, "linear")

        if strategy == "log":
            # Log-Uniform Initialization
            log_lo, log_hi = np.log(max(lo, 1e-5)), np.log(max(hi, 1e-5))
            noise = np.exp(np.random.uniform(log_lo, log_hi, size=(n, len(pos_arr))))
        else:
            # Standard Linear Uniform Initialization
            noise = np.random.uniform(lo, hi, size=(n, len(pos_arr)))
            
        current_slice = population[:, pos_arr, f]
        mask_slice = time_mask[:, pos_arr, f]
        population[:, pos_arr, f] = np.where(mask_slice, noise, current_slice)

    # --- Dynamic categorical ---
    for f in vary_dynamic_cat_idx:
        cats = np.asarray(valid_categories_per_idx[f])
        noise = np.random.choice(cats, size=(n, len(pos_arr)))

        current_slice = population[:, pos_arr, f]
        mask_slice = time_mask[:, pos_arr, f]
        population[:, pos_arr, f] = np.where(mask_slice, noise, current_slice)

    return population


def _sbi_initialization_strategy(
    trace:                np.ndarray,
    positions:            List[int],
    train_arr:            np.ndarray,      # (n_cases, seq_len, n_features)
    vary_static_idx:      np.ndarray,
    vary_dynamic_idx:     np.ndarray,
    n:                    int,
    sparse_n:             int = 0,
    sparsity_feature_p:   float = 0.2,
) -> np.ndarray:
    
    """
    Sampling-Based Initialization.

    Samples real attribute combinations from training data.
    Realistic values, no class guidance.
    Independent sample per position — maximizes diversity.
    """

    population = np.tile(trace.copy(), (n, 1, 1)).astype(object)
    T, F = trace.shape

    feature_mask, time_mask = _build_population_sparsity_mask(n, T, F, sparse_n, sparsity_feature_p)

    idx = np.random.randint(0, len(train_arr), size=n)
    sampled = train_arr[idx]

    # --- Static ---
    for f in vary_static_idx:
        noise = sampled[:, 0, f][:, None]  # correct scalar per case
        noise = np.repeat(noise, T, axis=1)

        population[:, :, f] = np.where(feature_mask[:, f][:, None], noise, population[:, :, f])

    # --- Dynamic ---
    for f in vary_dynamic_idx:
        for p in positions:
            noise = sampled[:, p, f]
            population[:, p, f] = np.where(time_mask[:, p, f], noise, population[:, p, f])

    return population


def _cbi_initialization_strategy(
    trace:                np.ndarray,
    positions:            List[int],
    conformant_arr:       np.ndarray,     # (n_cases, seq_len, n_features)
    vary_static_idx:      np.ndarray,
    vary_dynamic_idx:     np.ndarray,
    n:                    int,
    sparse_n:             int = 0,
    sparsity_feature_p:   float = 0.2,
) -> np.ndarray:
    
    """
    Case-Based Initialization.

    Samples from conformant training cases (next activity = conformant class).
    Same conformant case used across all positions — preserves realistic attribute correlations between positions.
    Fastest convergence toward conformant class.
    """

    population = np.tile(trace.copy(), (n, 1, 1)).astype(object)
    T, F = trace.shape

    feature_mask, time_mask = _build_population_sparsity_mask(n, T, F, sparse_n, sparsity_feature_p)

    idx = np.random.randint(0, len(conformant_arr), size=n)
    sampled = conformant_arr[idx]

    # --- Static ---
    for f in vary_static_idx:
        noise = sampled[:, 0, f][:, None]
        noise = np.repeat(noise, T, axis=1)

        population[:, :, f] = np.where(feature_mask[:, f][:, None], noise, population[:, :, f])

    # --- Dynamic ---
    for f in vary_dynamic_idx:
        for p in positions:
            population[:, p, f] = np.where(time_mask[:, p, f], sampled[:, p, f], population[:, p, f])

    return population



# --- Population initialization ---
def initialize_population(
    trace:                      np.ndarray,
    positions:                  List[int],
    train_arr:                  np.ndarray,
    conformant_arr:             np.ndarray,
    vary_static_idx:            np.ndarray,
    vary_dynamic_idx:           np.ndarray,
    vary_static_cont_idx:       np.ndarray,
    vary_static_cat_idx:        np.ndarray,
    vary_dynamic_cont_idx:      np.ndarray,
    vary_dynamic_cat_idx:       np.ndarray,
    strategies_per_idx:         Dict[int, str],
    valid_ranges_per_idx:       Dict[int, Tuple[float, float]],
    valid_categories_per_idx:   Dict[int, List[str]],
    population_size:            int = 200,
    cbi_ratio:                  float = 0.4,
    sbi_ratio:                  float = 0.3,
    sparsity_pop_ratio:         float = 0.02,
    sparsity_feature_p:         float = 0.2,
) -> np.ndarray:
    
    """
    Hybrid population initialization — CBI + SBI + RBI.

    Returns:
        population: (population_size, seq_len, n_features)
    """

    # --- Validate positions ---
    seq_len = trace.shape[0]
    for pos in positions:
        if pos < 0 or pos >= seq_len:
            raise ValueError(
                f"Position {pos} out of range for trace length {seq_len}."
            )

    # --- Initialize sub-populations ---
    n_cbi = int(population_size * cbi_ratio)
    n_sbi = int(population_size * sbi_ratio)
    n_rbi = population_size - n_cbi - n_sbi

    sparse_n = int(population_size * sparsity_pop_ratio)
    sparse_n_cbi = int(sparse_n * cbi_ratio)
    sparse_n_sbi = int(sparse_n * sbi_ratio)
    sparse_n_rbi = sparse_n - sparse_n_cbi - sparse_n_sbi
    

    pop_cbi = _cbi_initialization_strategy(
        trace=trace,
        positions=positions,
        conformant_arr=conformant_arr,
        vary_static_idx=vary_static_idx,
        vary_dynamic_idx=vary_dynamic_idx,
        n=n_cbi,
        sparse_n=sparse_n_cbi,
        sparsity_feature_p=sparsity_feature_p
    )

    pop_sbi = _sbi_initialization_strategy(
        trace=trace,
        positions=positions,
        train_arr=train_arr,
        vary_static_idx=vary_static_idx,
        vary_dynamic_idx=vary_dynamic_idx,
        n=n_sbi,
        sparse_n=sparse_n_sbi,
        sparsity_feature_p=sparsity_feature_p
    )

    pop_rbi = _rbi_initialization_strategy(
        trace=trace,
        positions=positions,
        vary_static_cont_idx=vary_static_cont_idx,
        vary_static_cat_idx=vary_static_cat_idx,
        vary_dynamic_cont_idx=vary_dynamic_cont_idx,
        vary_dynamic_cat_idx=vary_dynamic_cat_idx,
        strategies_per_idx=strategies_per_idx,
        valid_ranges_per_idx=valid_ranges_per_idx,
        valid_categories_per_idx=valid_categories_per_idx,
        n=n_rbi,
        sparse_n=sparse_n_rbi,
        sparsity_feature_p=sparsity_feature_p
    )

    # --- Merge populations ---
    population = np.concatenate(
        [pop_cbi, pop_sbi, pop_rbi],
        axis=0
    )

    return population
