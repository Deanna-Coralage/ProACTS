import numpy as np



# --- Niche fitness sharing ---
def niche_sharing(
    fitness:                    np.ndarray,      
    pairwise_distance_matrix:   np.ndarray, 
    sigma_quantile:             float = 0.25,
    alpha:                      float = 1.0,
) -> np.ndarray:

    """
    Niche sharing with adaptive sigma based on global distance distributions.
    Protects niche penalties from scaling distortion in minimization tasks.
    """
    
    n = len(fitness)

    if n <= 1:
        return fitness.copy()

    iu = np.triu_indices(n, k=1)
    pairwise_distances = pairwise_distance_matrix[iu]

    if len(pairwise_distances) == 0:
        return fitness.copy()

    sigma_share = np.percentile(pairwise_distances, sigma_quantile * 100)
    sigma_share = max(sigma_share, 1e-4)

    # --- Sharing function (Goldberg) ---
    sharing = np.where(
        pairwise_distance_matrix < sigma_share,
        1.0 - (pairwise_distance_matrix / sigma_share) ** alpha,
        0.0
    )
    np.fill_diagonal(sharing, 1.0)

    # --- Niche count ---
    niche_count = sharing.sum(axis=1)
    
    # --- Robust fitness scale ---
    q5, q95 = np.percentile(fitness, [5, 95])
    fitness_range = q95 - q5
    
    if fitness_range < 1e-8:
        penalty_scale = np.median(np.abs(fitness)) * 0.10
    else:
        penalty_scale = fitness_range * 0.10
    
    min_allowable_scale = max(
        float(np.median(np.abs(fitness))) * 0.01,
        1e-4,
    )
    
    penalty_scale = max(float(penalty_scale), min_allowable_scale)
    
    niche_penalty = (niche_count - 1.0) * penalty_scale
    shared_fitness = fitness + niche_penalty
    
    return shared_fitness.astype(np.float64)



# --- Elitism ---
def elitism(
    cf_pop:      np.ndarray,
    fitness:     np.ndarray,
    n_elites:    int,
) -> np.ndarray:
    
    """
    Select best individuals based on fitness (lower = better).
    """

    elite_idx = np.argsort(fitness)[:n_elites]

    return cf_pop[elite_idx].copy()



# --- Tournament selection ---
def _tournament_selection(
    fitness:          np.ndarray,
    tournament_size:  int = 3,
) -> int:
    
    """
    Returns index of selected individual.
    """

    candidates = np.random.choice(
        len(fitness),
        size=tournament_size,
        replace=False
    )

    winner = candidates[np.argmin(fitness[candidates])]

    return winner


def tournament_selection_batch(
    fitness:          np.ndarray,
    n_select:         int,
    tournament_size:  int = 3,
) -> np.ndarray:

    """
    Returns indices of selected individuals.
    """

    selected = np.empty(n_select, dtype=np.int32)

    for i in range(n_select):
        selected[i] = _tournament_selection(
            fitness,
            tournament_size
        )

    return selected
