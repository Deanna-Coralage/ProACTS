from dataclasses import dataclass



@dataclass(frozen=True)
class GAConfig:
    
    """
    Holds all hyperparameters for genetic search.
    """
    
    # --- Population ---
    population_size:  int   = 50
    offspring_size:   int   = 200
    cbi_ratio:        float = 0.08
    sbi_ratio:        float = 0.08

    # --- Generations ---
    num_generations:  int   = 200
    early_stopping:   int   = 50 

    # --- Sparsity control during initialization ---
    sparsity_pop_ratio:      float = 0.04
    sparsity_feature_p:      float = 0.1

    # --- Fitness weights ---
    w_distance:              float = 1.0
    w_sparsity:              float = 1.0
    w_margin:                float = 1.0
    w_process_violation:     float = 1.0
    
    # --- Margin loss ---
    confidence_ratio:        float = 2.0

    # --- Process violations ---
    penalty_desired_nonflip:      float = 3.0
    penalty_desired_wrongflip:    float = 5.0
    penalty_undesired_flip:       float = 2.0
    penalty_flexible_wrongflip:   float = 2.0
    
    # --- Fitness sharing ---
    use_fitness_sharing:    bool  = True
    niche_sigma_quantile:   float = 0.15    # small neighbor distance; small niches
    niche_alpha:            float = 1.0     # controls the shape of the sharing function.
                                            # alpha = 1 -> linear decay; alpha > 1 -> sharing drops faster near the niche boundary; 
    
    # --- Elitism ---
    elite_k:          int   = 1

    # --- Selection ---
    tournament_size:  int   = 3

    # --- crossover --- 
    crossover_rate_min:  float = 0.5
    crossover_rate_max:  float = 0.9  
                                
    # --- Mutation ---
    mutation_rate_min:   float = 0.05
    mutation_rate_max:   float = 0.3

    # --- Implausibility ---
    implaus_sample_ratio:     float = 1.0

    # --- Output ---
    output_population_size:   int   = 20

    
    # --- Helpers ---
    def validate(self):
        if self.cbi_ratio + self.sbi_ratio >= 1.0:
            raise ValueError(
                f"cbi_ratio + sbi_ratio must be < 1.0. "
                f"Got {self.cbi_ratio + self.sbi_ratio:.2f}."
            )
        if self.elite_k >= self.population_size:
            raise ValueError(
                f"elite_k must be < population_size. "
                f"Got elite_k={self.elite_k}, population_size={self.population_size}."
            )

        if self.population_size % 2 != 0:
            raise ValueError(
                f"population_size must be even. "
                f"Got population_size={self.population_size}."
            )
        if self.output_population_size > self.population_size:
            raise ValueError(
                f"output_population_size must be < population_size. "
                f"Got output_population_size={self.output_population_size}, population_size={self.population_size}."
            )
        if self.confidence_ratio < 1.0:
            raise ValueError(
                f"Confidence_ratio must be > 1.0. "
                f"Got {self.confidence_ratio}."
            )
