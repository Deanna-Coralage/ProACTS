from dataclasses import dataclass



@dataclass(frozen=True)
class EventLogDiCEConfig:
    
    """
    Holds all hyperparameters for genetic search.
    """

    # --- Output ---
    output_population_size:   int   = 1

    # --- Multi positions ---
    mode:            str = "multi"   # ["last", "multi"]

    # --- Population ---
    cbi_ratio:       float = 0.08
    sbi_ratio:       float = 0.08

    # --- Generations ---
    learning_rate:             float = 0.01
    optimization_steps:        int   = 500

    # --- Early stopping ---
    early_stopping_min_steps:  int = 100
    early_stopping_patience:   int = 50

    # --- Sparsity control during initialization ---
    sparsity_pop_ratio:      float = 1.0
    sparsity_feature_p:      float = 0.0

    # --- Loss weights ---
    w_margin_loss:        float = 1.0
    w_scenario_loss:      float = 1.0
    w_distance_loss:      float = 1.0
    w_cat_loss:           float = 1.0
    # w_diversity_loss:     float = 0.0
    # w_sparsity_loss:      float = 0.0

    # --- Margin loss ---
    confidence_ratio:        float = 2.0

    # --- Catgorical feature handling ---
    use_valid_cf_only:    bool = False
    use_sampling:         bool = False

    # --- Fitness weights ---
    w_distance:              float = 1.0
    w_sparsity:              float = 1.0
    w_margin:                float = 1.0
    w_process_violation:     float = 1.0

    # --- Process violations ---
    penalty_desired_nonflip:      float = 3.0
    penalty_desired_wrongflip:    float = 5.0
    penalty_undesired_flip:       float = 2.0
    penalty_flexible_wrongflip:   float = 2.0

    # --- Implausibility ---
    implaus_sample_ratio:     float = 1.0

    # --- Gumbel softmax sampling (optimized) ---
    initial_logit_scale:          float = 0.5
    gumbel_initial_temperature:   float = 2.0
    gumbel_min_temperature:       float = 0.5
    gumbel_anneal_rate:           float = 0.998
    gumbel_hard:                  bool  = False

    
    # --- Helpers ---
    def validate(self):
        if self.output_population_size != 1:
            raise ValueError(
                f"DiCE4EL requires output_population_size == 1. "
                f"Got {self.output_population_size}."
            )
        if self.mode not in {"last", "multi"}:
            raise ValueError(
                f"Mode must be 'last' or 'multi'. Got '{self.mode}'."
            )
        if self.confidence_ratio < 1.0:
            raise ValueError(
                f"Confidence_ratio must be > 1.0. "
                f"Got {self.confidence_ratio}."
            )
