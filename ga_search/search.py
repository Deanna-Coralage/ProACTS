import time

import numpy as np
import pandas as pd

from dataclasses import dataclass

from typing import Dict, List, Tuple, Set, Union, Optional

from collections import deque

from utils.general_utils import save_counterfactual_results
from utils.feature_utils import compute_vary_indices, compute_feature_configs
from utils.trace_utils import get_rollout_positions, add_simulation_mismatches_as_flexible

from config.feature_config import FeatureConfig
from config.ga_config import GAConfig

from model.model_wrapper import ModelWrapper

from ga_search.initialization import initialize_population
from ga_search.fitness import calculate_fitness, calculate_pairwise_distance_matrix, evaluate_cf_pop
from ga_search.selection import niche_sharing, elitism, tournament_selection_batch
from ga_search.operations import crossover, mutation

from process.constraint import ProcessConstraint



# --- Adaptive parameters ---    
def _process_constraint_based_adaptive_crossover_rate(
    parent1_process_violation:       np.ndarray,    # (P,)
    parent2_process_violation:       np.ndarray,    # (P,)
    population_process_violation:    np.ndarray,    # (N,)
    parent1_margin_loss:             np.ndarray,
    parent2_margin_loss:             np.ndarray,
    population_margin_loss:          np.ndarray,
    crossover_rate_min:              float = 0.5,
    crossover_rate_max:              float = 0.9,
) -> np.ndarray:                     # (P,)
    
    """
    Population-relative adaptive crossover rate.
    MINIMIZATION: Low violation = Elite fitness -> HIGH crossover rate.
    """
    
    # --- Pair quality ---
    pair_violation = np.minimum(parent1_process_violation, parent2_process_violation)

    pair_margin = np.minimum(parent1_margin_loss, parent2_margin_loss)
    
    # --- Population stats ---
    def _normalize_minimization(x, pop):
        pop_min = float(np.min(pop))
        pop_max = float(np.max(pop))

        if abs(pop_max - pop_min) < 1e-8:
            return np.zeros_like(x, dtype=np.float64)

        return np.clip(
            (x - pop_min) / (pop_max - pop_min),
            0.0,
            1.0,
        )

    violation_score = _normalize_minimization(
        pair_violation,
        population_process_violation,
    )

    margin_score = _normalize_minimization(
        pair_margin,
        population_margin_loss,
    )

    # --- Combine scores into a performance penalty (0.0 = Best, 1.0 = Worst) ---
    w_process = 1.0
    w_margin = 1.0
    penalty_score = (
        w_process * violation_score
        + w_margin * margin_score
    )
    penalty_score = penalty_score / (w_process + w_margin + 1e-8)

    # --- Map directly: Low penalty (Elite) -> min rate | High penalty (Poor) -> max rate ---
    crossover_rate = (
        crossover_rate_min
        + (crossover_rate_max - crossover_rate_min) * penalty_score
    )

    return np.clip(crossover_rate, crossover_rate_min, crossover_rate_max).astype(np.float64)


def _precompute_base_mutation_sigma(
    valid_ranges_per_idx:    Dict[int, Tuple[float, float]],
    strategies_per_idx:      Dict[int, str],
    ratio:                   float = 0.10
) -> Dict[int, float]:
    
    """
    Precompute feature-wise base sigma using a 10% rule.
    Dynamically identifies if a feature requires linear or logarithmic 
    mutation based on its scale span.
    """
    
    base_mutation_sigma = {}

    for idx, (lo, hi) in valid_ranges_per_idx.items():
        strategy = strategies_per_idx.get(idx, "linear")
        
        if strategy == "log":
            log_span = np.log(hi) - np.log(lo)
            base_mutation_sigma[idx] = float(ratio * log_span)
        else:
            base_mutation_sigma[idx] = float(ratio * (hi - lo))

    return base_mutation_sigma

    
def _reciprocal_diversity_based_adaptive_mutation_rate(
    diversity:                 float,
    initial_diversity:         float = 1.0, 
    windowed_max_diversity:    float = 0.0,
    mutation_rate_min:         float = 0.01,
    mutation_rate_max:         float = 0.3,
) -> float:

    """
    Hybrid reciprocal diversity adaptation for mutation rates.
    
    1. Uses windowed max diversity for local relative tracking.
    2. Implements a scale-agnostic safety floor anchored to Gen 0 diversity.
    3. Applies non-linear exponential scaling to force aggressive spikes 
       if the population approaches clone flatlining.
    """

    if windowed_max_diversity < 1e-4:
        relative_factor = 1.0
    else:
        # Invert: low relative diversity = high mutation factor
        normalized_diversity = np.clip(diversity / windowed_max_diversity, 0.0, 1.0)
        relative_factor = float(1.0 - (normalized_diversity ** 2))

    # --- Scale-Agnostic Absolute Safety Floor ---
    # Trigger an aggressive safety override if diversity drops below 25% of its starting value
    abs_diversity_floor = initial_diversity * 0.25

    if diversity < abs_diversity_floor:
        # Measure structural collapse severity relative to your dynamic floor scale
        severity = 1.0 - (diversity / (abs_diversity_floor + 1e-8))
        # Exponential kick makes mutation spike fast if diversity approaches zero
        absolute_factor = float(severity ** 2)
    else:
        absolute_factor = 0.0

    # --- Aggregate ---
    # Take whichever factor demands a higher mutation rate to rescue the population
    final_factor = max(relative_factor, absolute_factor)
    
    # --- Adaptive mutation rate ---
    mutation_rate = mutation_rate_min + (mutation_rate_max - mutation_rate_min) * final_factor
    
    return float(np.clip(mutation_rate, mutation_rate_min, mutation_rate_max))


def _simulated_annealing_based_diversity_aware_mutation_sigma(
    base_mutation_sigma:    Dict[int, float],
    current_gen:            int,
    max_gen:                int,
    diversity:              float,
    initial_diversity:      float = 1.0, 
    collapse_ratio:         float = 0.10,
    boost_factor:           float = 2.0,
) -> Dict[int, float]:
    
    """
    Applies non-linear cosine cooling to all feature sigmas.
    Preserves large exploratory jumps longer into the search phase.
    """
    
    if diversity < (initial_diversity * collapse_ratio):
        decay = boost_factor
    else:
        min_decay_floor = 0.05
        
        # --- Cosine annealing maintains solid exploratory mutation width during the crucial first 50% of runs ---
        decay = 0.5 * (1.0 + np.cos(np.pi * (current_gen / max_gen)))
        decay = float(np.clip(decay, min_decay_floor, 1.0))

    return {
        idx: float(sigma * decay)
        for idx, sigma in base_mutation_sigma.items()
    }



@dataclass
class WindowedDiversity:
    
    """
    Tracks windowed maximum diversity over last k generations.
    """

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, diversity: float) -> float:
        self.buffer.append(diversity)

        return float(np.max(self.buffer))

    def get_max(self) -> float:
        if len(self.buffer) == 0:
            return 0.0
        return float(np.max(self.buffer))

    def get_values(self):
        return list(self.buffer)



@dataclass(frozen=True)
class CounterfactualGA:

    """
    Counterfactual Genetic Algorithm (CF-GA)

    This class implements an evolutionary search framework for generating
    counterfactual explanations using a structured genetic algorithm.

    The algorithm operates over a population of candidate counterfactuals
    and iteratively improves them using:

    - Tournament selection 
    - Crossover
    - Mutation (perturbations)
    - Niche sharing (forces diversity)
    - Elitism (preserving best solutions)

    The GA is designed for structured data (sequence + tabular features)
    with explicit handling of:
    - static vs dynamic features
    - categorical vs continuous variables
    - valid intervention positions in time series
    """
    
    ga_config:            GAConfig
    feature_config:       FeatureConfig
    model_wrapper:        ModelWrapper

    def __post_init__(self):
        object.__setattr__(self, "diversity_window", WindowedDiversity(15))


    def search(
        self,
        trace:                         np.ndarray,
        train_arr:                     np.ndarray,
        conformant_arr:                np.ndarray,
        change_allowed_positions:      List[int],
        desired_constraints:           ProcessConstraint,
        flexible_constraints:          ProcessConstraint,
        output_path:                   Optional[str] = None,
    ) -> Dict[str, Union[Dict[str, float], np.ndarray]]:

        """
        Main evolutionary search loop for CounterfactualGA.
    
        Steps:
        1. Initialize population
        2. Simulate forward autoregresively through model
        3. Evaluate fitness
        4. Check early stopping criteria
        5. Apply optional niche sharing
        6. Selection (tournament)
        7. Crossover + mutation
        8. Elitism preservation
        9. Recombination
        """

        desired_positions = desired_constraints.get_constrained_positions()
        # --- Safeguard against undesired changes ---
        if not desired_positions:
            raise ValueError("desired_constraints has no constrained positions.")
        
        if change_allowed_positions:
            max_allowed = max(change_allowed_positions)
            max_desired = max(desired_positions)
        
            if max_allowed > max_desired:
                raise ValueError(
                    f"Invalid config: change_allowed_positions includes {max_allowed}, "
                    f"which is after the last desired position {max_desired}."
                )
        

        start = time.perf_counter()

        # --- Precompute indices for reuse ---
        vary_indices = compute_vary_indices(self.feature_config)
        precomputed_feature_configs = compute_feature_configs(self.feature_config)
       
        activity_idx = vary_indices["activity_idx"]
        vary_static_idx = vary_indices["vary_static_idx"]
        vary_dynamic_idx = vary_indices["vary_dynamic_idx"]
        vary_cont_idx = vary_indices["vary_cont_idx"]
        vary_cat_idx = vary_indices["vary_cat_idx"]
        vary_static_cont_idx = vary_indices["vary_static_cont_idx"]
        vary_static_cat_idx = vary_indices["vary_static_cat_idx"]
        vary_dynamic_cont_idx = vary_indices["vary_dynamic_cont_idx"]
        vary_dynamic_cat_idx = vary_indices["vary_dynamic_cat_idx"]
        
        valid_ranges_per_idx = precomputed_feature_configs["valid_ranges_per_idx"]
        valid_categories_per_idx = precomputed_feature_configs["valid_categories_per_idx"]
        strategies_per_idx = precomputed_feature_configs["feature_strategies_per_idx"]
        mad_arr = precomputed_feature_configs["mad_arr"]
        norm_arr = precomputed_feature_configs["norm_arr"]

        # --- Simulated trace for original trace from model ---
        simulated_trace = self.model_wrapper.simulate_forward_nonautoregressive(
            traces=trace,
            desired_positions=list(range(1, len(trace))),
        )["simulated_traces"][0]

        # --- Sigma for mutation ---
        base_mutation_sigma =  _precompute_base_mutation_sigma(
            valid_ranges_per_idx=valid_ranges_per_idx,
            strategies_per_idx=strategies_per_idx
        )

        # --- Add mismatches as flexible positions ---
        flexible_constraints_with_simulation_mismatches = add_simulation_mismatches_as_flexible(
            original_trace=trace,
            simulated_trace=simulated_trace,
            activity_idx=self.feature_config.feature_to_idx[self.feature_config.activity_feature],
            flexible_constraints=flexible_constraints,
        )
    
        # --- Fitness kwargs for every call ---
        fitness_kwargs = dict(
            trace = simulated_trace,
            positions = change_allowed_positions,
            desired_constraints = desired_constraints,
            flexible_constraints = flexible_constraints_with_simulation_mismatches,
            activity_idx = activity_idx,
            vary_static_idx = vary_static_idx,
            vary_dynamic_idx = vary_dynamic_idx,
            vary_cont_idx = vary_cont_idx,
            vary_cat_idx = vary_cat_idx,
            mad_arr = mad_arr,
            norm_arr=norm_arr,
            confidence_ratio=self.ga_config.confidence_ratio,
            penalty_desired_nonflip = self.ga_config.penalty_desired_nonflip,
            penalty_desired_wrongflip = self.ga_config.penalty_desired_wrongflip,
            penalty_undesired_flip = self.ga_config.penalty_undesired_flip,
            penalty_flexible_wrongflip = self.ga_config.penalty_flexible_wrongflip,
            w_distance = self.ga_config.w_distance,
            w_sparsity = self.ga_config.w_sparsity,
            w_margin = self.ga_config.w_margin,
            w_process_violation = self.ga_config.w_process_violation,
        )
    
        # --- Helper — evaluate one population ---
        def _evaluate(cf_pop: np.ndarray):
            
            """Simulate + fitness + sharing for one population."""
            
            rollout_positions = get_rollout_positions(trace=simulated_trace, cf_pop=cf_pop)
            forward_simul = self.model_wrapper.simulate_forward_autoregressive(
                traces=cf_pop,
                rollout_positions=rollout_positions,
                desired_positions=desired_positions,
                flexible_constraints=flexible_constraints,
            )

            forward_simul_na = self.model_wrapper.simulate_forward_nonautoregressive(
                traces=cf_pop,
                desired_positions=desired_positions,
            )

            fitness_components=calculate_fitness(
                cf_pop=cf_pop,
                simulated_cfs=forward_simul["simulated_traces"],
                logits_store=forward_simul["logits_store"],
                logits_store_na=forward_simul_na["logits_store"],
                **fitness_kwargs,
            )

            return fitness_components, forward_simul, forward_simul_na

        def _niche(cf_pop: np.ndarray, fitness: np.ndarray):

            pdm = calculate_pairwise_distance_matrix(
                    cf_pop=cf_pop,
                    vary_cont_idx=vary_cont_idx,
                    vary_cat_idx=vary_cat_idx,
                    mad_arr=mad_arr,
                    norm_arr=norm_arr,
                    positions=change_allowed_positions,
                    vary_static_idx=vary_static_idx,
                    vary_dynamic_idx=vary_dynamic_idx,
                )
            
            if self.ga_config.use_fitness_sharing:
                shared_fitness = niche_sharing(
                    fitness=fitness,
                    pairwise_distance_matrix=pdm,
                    sigma_quantile=self.ga_config.niche_sigma_quantile,
                    alpha=self.ga_config.niche_alpha,
                )
            else:
                shared_fitness = fitness
    
            return shared_fitness, pdm

        
        # --- Initialize population ---
        cf_pop = initialize_population(
            trace=simulated_trace,
            positions=change_allowed_positions,
            train_arr=train_arr,
            conformant_arr=conformant_arr,
            vary_static_idx=vary_static_idx,
            vary_dynamic_idx=vary_dynamic_idx,
            vary_static_cont_idx=vary_static_cont_idx,
            vary_static_cat_idx=vary_static_cat_idx,
            vary_dynamic_cont_idx=vary_dynamic_cont_idx,
            vary_dynamic_cat_idx=vary_dynamic_cat_idx,
            strategies_per_idx=strategies_per_idx,
            valid_ranges_per_idx=valid_ranges_per_idx,
            valid_categories_per_idx=valid_categories_per_idx,
            population_size=self.ga_config.population_size,
            cbi_ratio=self.ga_config.cbi_ratio,
            sbi_ratio=self.ga_config.sbi_ratio,
            sparsity_pop_ratio=self.ga_config.sparsity_pop_ratio,
            sparsity_feature_p=self.ga_config.sparsity_feature_p
        )

        # --- Evaluate fitness ---
        fitness_components, _, _ = _evaluate(cf_pop)
        fitness = fitness_components["fitness"]
        process_violation = fitness_components["process_violation"]
        margin_loss = fitness_components["margin_loss"]
        
        shared_fitness, pdm = _niche(cf_pop, fitness)
        iu = np.triu_indices(pdm.shape[0], k=1)
        initial_diversity = np.mean(pdm[iu]).astype(np.float64)
        
        
        # --- Generation loop ---
        N, T, F = cf_pop.shape
        global_best = np.inf
        no_improvement  = 0
    
        for current_gen in range(self.ga_config.num_generations):
    
            # --- Early stopping ---
            current_best = fitness.min()
            if current_best < global_best:
                global_best = current_best
                no_improvement = 0
            else:
                no_improvement += 1
    
            if no_improvement >= self.ga_config.early_stopping:
                print(
                    f"Early stopping at generation {current_gen}: "
                    f"no improvement for {no_improvement} generations. "
                    f"Best fitness = {global_best:.6f}"
                )
                break
            
            # --- Selection ---
            parent_indices = tournament_selection_batch(
                fitness=shared_fitness,
                n_select=self.ga_config.offspring_size,
                tournament_size=self.ga_config.tournament_size
            )
            parents = cf_pop[parent_indices]               # (P, T, F)
            parents = parents.reshape(-1, 2, T, F)         # pair parents: (P/2, 2, T, F)

            parent1_process_violation = process_violation[parent_indices[0::2]]
            parent2_process_violation = process_violation[parent_indices[1::2]]

            parent1_margin_loss = margin_loss[parent_indices[0::2]]
            parent2_margin_loss = margin_loss[parent_indices[1::2]]
            
            # --- Genetic operations ---

            # --- Calculate adaptive parameters ---
            crossover_rate = _process_constraint_based_adaptive_crossover_rate(
                parent1_process_violation=parent1_process_violation,
                parent2_process_violation=parent2_process_violation,
                population_process_violation=process_violation,
                parent1_margin_loss=parent1_margin_loss,
                parent2_margin_loss=parent2_margin_loss,
                population_margin_loss=margin_loss,
                crossover_rate_min=self.ga_config.crossover_rate_min,
                crossover_rate_max=self.ga_config.crossover_rate_max
            )

            iu = np.triu_indices(pdm.shape[0], k=1)
            diversity = np.mean(pdm[iu]).astype(np.float64)
            windowed_max_diversity = self.diversity_window.update(diversity)
            
            mutation_rate = _reciprocal_diversity_based_adaptive_mutation_rate(
                diversity=diversity,
                initial_diversity=initial_diversity,
                windowed_max_diversity=windowed_max_diversity,
                mutation_rate_min=self.ga_config.mutation_rate_min,
                mutation_rate_max=self.ga_config.mutation_rate_max
            )
            mutation_sigma =  _simulated_annealing_based_diversity_aware_mutation_sigma(
                base_mutation_sigma=base_mutation_sigma,
                current_gen=current_gen,
                max_gen=self.ga_config.num_generations,
                diversity=diversity,
                initial_diversity=initial_diversity,
            )
   
            # --- Crossover + mutation ---
            children1, children2 = crossover(
                parents=parents,
                positions=change_allowed_positions,
                vary_static_idx=vary_static_idx,
                vary_dynamic_idx=vary_dynamic_idx,
                crossover_rate=crossover_rate
            )
            children = np.concatenate([children1, children2], axis=0)     # merge children: (offspring_size: P, T, F)

            children = mutation(
                children=children,
                positions=change_allowed_positions,
                vary_static_cont_idx=vary_static_cont_idx,
                vary_static_cat_idx=vary_static_cat_idx,
                vary_dynamic_cont_idx=vary_dynamic_cont_idx,
                vary_dynamic_cat_idx=vary_dynamic_cat_idx,
                sigma_per_idx=mutation_sigma,
                strategies_per_idx=strategies_per_idx,
                valid_ranges_per_idx=valid_ranges_per_idx,
                valid_categories_per_idx=valid_categories_per_idx,
                mutation_rate=mutation_rate
            )

            # --- Evaluate children ---
            children_fitness_components, _, _ = _evaluate(children)
            children_fitness = children_fitness_components["fitness"]
            children_process_violation = children_fitness_components["process_violation"]
            children_margin_loss = children_fitness_components["margin_loss"]

            # --- Unified pool ---
            pool = np.concatenate([cf_pop, children], axis=0)
            pool_fitness = np.concatenate([fitness, children_fitness], axis=0)
            pool_process_violation = np.concatenate([process_violation, children_process_violation], axis=0)
            pool_margin_loss = np.concatenate([margin_loss, children_margin_loss], axis=0)

            pool_shared_fitness, pool_pdm = _niche(pool, pool_fitness)

            # --- Elitism ---
            elite_idx = np.argsort(pool_fitness)[:self.ga_config.elite_k]
            original_elite_shared = pool_shared_fitness[elite_idx].copy()
            pool_shared_fitness[elite_idx] = -np.inf

            # --- Select survivors ---
            selected_idx = np.argsort(pool_shared_fitness)[:self.ga_config.population_size]

            # --- New population ---
            cf_pop = pool[selected_idx]
            fitness = pool_fitness[selected_idx]
            process_violation = pool_process_violation[selected_idx]
            margin_loss = pool_margin_loss[selected_idx]

            shared_fitness = pool_shared_fitness[selected_idx]
            pdm = pool_pdm[np.ix_(selected_idx, selected_idx)]
            
            for idx_in_pool, original_val in zip(elite_idx, original_elite_shared):
                mask = (selected_idx == idx_in_pool)
                shared_fitness[mask] = original_val
            

        end = time.perf_counter()
        print(f"Time taken GA loop: {end - start:.6f} seconds")
        
        # --- Evaluate and return final cf population ---
        start = time.perf_counter()

        output_idx = np.argsort(fitness)[:self.ga_config.output_population_size]
        output_cf_pop = cf_pop[output_idx]

        output_cf_pop_fitness_components, output_forward_simul, output_forward_simul_na = _evaluate(output_cf_pop)

        output_cf_pop_eval = evaluate_cf_pop(
            cf_pop=output_cf_pop,
            simulated_cfs=output_forward_simul["simulated_traces"],
            train_arr=train_arr,
            logits_store=output_forward_simul["logits_store"],
            logits_store_na=output_forward_simul_na["logits_store"],
            sample_ratio=self.ga_config.implaus_sample_ratio,
            **fitness_kwargs,
        )

        end = time.perf_counter()
        print(f"Time taken for fitness calculation: {end - start:.6f} seconds")
        
        print("=== Evaluation metrics of counterfactuals ===")
        print(output_cf_pop_eval)

        # --- Save results to excel ---
        if output_path is not None:
            save_counterfactual_results(
                output_path=output_path,
                fitness_components=output_cf_pop_fitness_components,
                evaluation=output_cf_pop_eval,
                trace=trace,
                cf_pop=output_cf_pop,
                simulated_trace=simulated_trace,
                simulated_cfs=output_forward_simul["simulated_traces"],
                simulated_cfs_na=output_forward_simul_na["simulated_traces"],
                feature_order=self.feature_config.feature_order
            )

        return {
            "eval": output_cf_pop_eval,
            "cf_pop": output_cf_pop,
            "fitness_components": output_cf_pop_fitness_components
        }
