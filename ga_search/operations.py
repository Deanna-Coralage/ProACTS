import numpy as np

from typing import Tuple, List, Dict, Optional



def crossover(
    parents:            np.ndarray,                     # (num_parent_pairs, 2, sequence_len, num_features)
    positions:          List[int],
    vary_static_idx:    np.ndarray,
    vary_dynamic_idx:   np.ndarray,
    crossover_rate:     Optional[np.ndarray] = None,    # (num_parent,)
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Batched crossover on paired parents.
    Enforces time-series uniformity for static features and runs 100% vectorised.
    """
    
    P, _, T, F = parents.shape

    if crossover_rate is None:
        crossover_rate = np.full(P, 0.9, dtype=np.float64)

    parent1 = parents[:, 0].copy().astype(object)
    parent2 = parents[:, 1].copy().astype(object)

    child1 = parent1.copy()
    child2 = parent2.copy()

    # --- Global crossover gate ---
    do_cross = (np.random.rand(P) < crossover_rate)   # (P,)
   
    if not np.any(do_cross):
        return child1, child2

    # --- Static features ---
    if len(vary_static_idx) > 0:
        p1_static_t0 = parent1[:, 0, vary_static_idx]  # (P, len(vary_static_idx))
        p2_static_t0 = parent2[:, 0, vary_static_idx]  # (P, len(vary_static_idx))

        mask = (np.random.rand(P, len(vary_static_idx)) < 0.5)
        mask &= do_cross[:, None]   

        c1_static_t0 = np.where(mask, p2_static_t0, p1_static_t0)
        c2_static_t0 = np.where(mask, p1_static_t0, p2_static_t0)

        child1[:, :, vary_static_idx] = c1_static_t0[:, None, :]
        child2[:, :, vary_static_idx] = c2_static_t0[:, None, :]

    # --- Dynamic features ---
    if len(vary_dynamic_idx) > 0 and len(positions) > 0:
        pos_arr = np.asarray(positions, dtype=int)
        
        dyn_mask = np.random.rand(P, len(pos_arr), len(vary_dynamic_idx)) < 0.5
        dyn_mask &= do_cross[:, None, None]

        p1_slice = parent1[:, pos_arr[:, None], vary_dynamic_idx]
        p2_slice = parent2[:, pos_arr[:, None], vary_dynamic_idx]
        c1_slice = child1[:, pos_arr[:, None], vary_dynamic_idx]
        c2_slice = child2[:, pos_arr[:, None], vary_dynamic_idx]

        child1[:, pos_arr[:, None], vary_dynamic_idx] = np.where(dyn_mask, p2_slice, c1_slice)
        child2[:, pos_arr[:, None], vary_dynamic_idx] = np.where(dyn_mask, p1_slice, c2_slice)

    return child1, child2



def mutation(
    children:                   np.ndarray,              # (num_children, sequence_len, num_features)
    positions:                  List[int],
    vary_static_cont_idx:       np.ndarray,
    vary_static_cat_idx:        np.ndarray,
    vary_dynamic_cont_idx:      np.ndarray,
    vary_dynamic_cat_idx:       np.ndarray,
    sigma_per_idx:              Dict[int, float],
    strategies_per_idx:         Dict[int, str],
    valid_ranges_per_idx:       Dict[int, Tuple[float, float]],
    valid_categories_per_idx:   Dict[int, List[str]],
    mutation_rate:              float = 0.01,
) -> np.ndarray:
    
    """
    Batched mutation on each offspring, safe for mixed string/float object arrays.
    """
    
    mutated = children.copy().astype(object)
    N, T, _ = mutated.shape
    positions_arr = np.asarray(positions, dtype=int)

    # --- Static continuous ---
    for f in vary_static_cont_idx:
        mask = np.random.rand(N) < mutation_rate
        if np.any(mask):
            num_mutated = mask.sum()
            noise = np.random.normal(loc=0.0, scale=sigma_per_idx[f], size=num_mutated)
           
            current_vals = mutated[mask, 0, f].astype(float)
            strategy = strategies_per_idx.get(f, "linear")

            if strategy == "log":
                current_vals = np.maximum(current_vals, 1e-5)
                # Mutate in log-space, then transform back using exponential
                current_vals = np.exp(np.log(current_vals) + noise)
            else:
                # Standard linear addition
                current_vals = current_vals + noise
            
            lo, hi = valid_ranges_per_idx[f]
            mutated_vals = np.clip(current_vals, lo, hi)
            
            mutated[mask, :, f] = mutated_vals[:, None]

    # --- Static categorical ---
    for f in vary_static_cat_idx:
        if f not in valid_categories_per_idx:
            raise ValueError(f"No valid categories provided for feature index {f}")

        mask = np.random.rand(N) < mutation_rate
        if np.any(mask):
            cats = np.asarray(valid_categories_per_idx[f], dtype=str)
            current_vals = mutated[mask, 0, f].astype(str)
            mutated_vals = np.empty(len(current_vals), dtype=object)

            for i, curr in enumerate(current_vals):
                allowed_cats = cats[cats != curr]
                mutated_vals[i] = str(np.random.choice(allowed_cats)) if len(allowed_cats) > 0 else str(curr)

            mutated[mask, :, f] = mutated_vals[:, None]

    # --- Dynamic continuous ---
    for f in vary_dynamic_cont_idx:
        mask = np.random.rand(N, len(positions_arr)) < mutation_rate
        if np.any(mask):
            
            
            row_indices, pos_indices = np.where(mask)
            for r, p_idx in zip(row_indices, pos_indices):
                target_pos = positions_arr[p_idx]
                noise = np.random.normal(loc=0.0, scale=sigma_per_idx[f])
                
                curr_vals = float(mutated[r, target_pos, f])
                strategy = strategies_per_idx.get(f, "linear")

                if strategy == "log":
                    curr_vals = max(curr_vals, 1e-5)
                    curr_vals = np.exp(np.log(curr_vals) + noise)
                else:
                    curr_vals = curr_vals + noise
                    
                lo, hi = valid_ranges_per_idx[f]
                mutated[r, target_pos, f] = float(np.clip(curr_vals, lo, hi))

    # --- Dynamic categorical ---
    for f in vary_dynamic_cat_idx:
        if f not in valid_categories_per_idx:
            raise ValueError(f"No valid categories provided for feature index {f}")

        mask = np.random.rand(N, len(positions_arr)) < mutation_rate
        if np.any(mask):
            cats = np.asarray(valid_categories_per_idx[f], dtype=str)
            row_indices, pos_indices = np.where(mask)
            
            for r, p_idx in zip(row_indices, pos_indices):
                target_pos = positions_arr[p_idx]
                curr_val = str(mutated[r, target_pos, f])
                
                if len(cats) > 1:
                    allowed_cats = cats[cats != curr_val]
                    mutated[r, target_pos, f] = str(np.random.choice(allowed_cats))
                else:
                    mutated[r, target_pos, f] = str(cats[0])

    
    return mutated
