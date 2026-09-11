# ProACTS (<u>Pro</u>pagation-<u>A</u>ware <u>C</u>oun<u>T</u>erfactual <u>S</u>equences)

This repository contains the source code and experimental evaluation of **ProACTS**, a propagation-aware counterfactual generation method for next-activity prediction.

ProACTS uses a genetic algorithm to search for counterfactual interventions that achieve desired activity predictions at target positions through margin loss while optimizing three complementary quality criteria: propagation loss, intervention distance, and sparsity. Propagation loss accounts for intervention effects across subsequent predictions using autoregressive rollout and distinguishes desired, permissible, and unwarranted activity changes.

## Repository Contents

The repository contains:

- **ProACTS** – the proposed propagation-aware counterfactual generation method.
- **DiCE4EL** – a baseline counterfactual generation method for next-activity prediction that evaluates target predictions in isolation without accounting for propagation.
- **DiCE4EL+** – our strengthened implementation of DiCE4EL using Gumbel--Softmax for categorical optimization.
- Event-log preprocessing scripts.
- Counterfactual query generation scripts.
- Experimental evaluation and statistical analysis.
- Scripts for generating the reported tables and figures.

### Propagation Diagnostics

To assess the practical relevance of propagation, we autoregressively evaluate counterfactuals that DiCE4EL+ considers target-successful under non-autoregressive evaluation.

| Dataset | Target-Successful CFs | CFs with Propagation | Propagation Prevalence (%) | AR Target Failures | AR Target Failure (%) |
|---|---:|---:|---:|---:|---:|
| BPIC2012 | 234 | 162 | 69.2 | 58 | 24.8 |
| BPIC2017 | 363 | 357 | 98.3 | 2 | 0.6 |
| BPIC2019 | 308 | 283 | 91.9 | 214 | 69.5 |
| BPIC2020-Int | 300 | 297 | 99.0 | 153 | 51.0 |
| BPIC2020-Dom | 56 | 52 | 92.9 | 44 | 78.6 |
| BPIC2020-Ptc | 156 | 151 | 96.8 | 65 | 41.7 |
| BPIC2020-Rfp | 59 | 51 | 86.4 | 32 | 54.2 |
| Road Fines | 224 | 171 | 76.3 | 3 | 1.3 |
| Sepsis Cases | 134 | 129 | 96.3 | 9 | 6.7 |
| Hospital Billing | 368 | 337 | 91.6 | 177 | 48.1 |
| **Overall** | **2,202** | **1,990** | **90.4** | **757** | **34.4** |

**Propagation Prevalence** is the percentage of non-autoregressively target-successful DiCE4EL+ counterfactuals exhibiting at least one unwarranted propagated activity change. **AR Target Failure** is the percentage of these counterfactuals that fail to retain the desired target under autoregressive rollout.

***Across 10 datasets, propagation prevalence ranges from 69.2\% to 99.0\% across datasets, while autoregressive target failure varies more substantially, from 0.6\% to 78.6\%.***

## Requirements

The experiments were implemented using Python 3.9.0 and PyTorch 2.2.2 with CUDA 12.1.

Create a Python 3.9 virtual environment:

```bash
python -m venv proacts
```

Activate the environment on macOS/Linux:

```bash
source proacts/bin/activate
```

or on Windows:

```bash
proacts\Scripts\activate
```

Install Jupyter, the IPython kernel, and the required dependencies:

```bash
python -m pip install --upgrade pip
pip install jupyter ipykernel
pip install -r requirements.txt
```

Install the ProACTS package:

```bash
pip install -e .
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name proacts --display-name "Python (ProACTS)"
```

Launch Jupyter:

```bash
jupyter notebook
```

Select **Python (ProACTS)** as the kernel when running the experiment notebooks.

## Datasets

The experiments use ten publicly available real-life event logs from the [4TU Centre for Research Data](https://data.4tu.nl/):

- BPIC2012
- BPIC2017
- BPIC2019
- BPIC2020 – International Declarations
- BPIC2020 – Domestic Declarations
- BPIC2020 – Prepaid Travel Cost
- BPIC2020 – Request for Payment
- Road Traffic Fine Management Process
- Sepsis Cases
- Hospital Billing

The datasets are not included in this repository. Please obtain the original event logs from the 4TU Centre for Research Data and place them in the `data/` directory.

## Configuration

The main experimental parameters are defined in `GAConfig` for ProACTS and `EventLogDiCEConfig` for DiCE4EL and DiCE4EL+.
The default values correspond to the configurations used in the experiments.

### ProACTS Configuration

| Parameter | Default | Description |
|---|---:|---|
| `population_size` | `50` | Number of candidates initialized and retained in each generation. |
| `offspring_size` | `200` | Number of offspring generated per generation. |
| `cbi_ratio` | `0.08` | Proportion of the initial population generated using case-based initialization (CBI). |
| `sbi_ratio` | `0.08` | Proportion of the initial population generated using sampling-based initialization (SBI). The remaining population is randomly initialized. |
| `num_generations` | `200` | Maximum number of genetic-algorithm generations. |
| `early_stopping` | `50` | Number of generations without improvement before early stopping. |
| `sparsity_pop_ratio` | `0.04` | Proportion of the initial population seeded with sparse interventions. |
| `sparsity_feature_p` | `0.1` | Probability of modifying each intervenable attribute-position pair when generating a sparse candidate. |
| `w_distance` | `1.0` | Weight of intervention-distance loss. |
| `w_sparsity` | `1.0` | Weight of sparsity loss. |
| `w_margin` | `1.0` | Weight of margin loss. |
| `w_process_violation` | `1.0` | Weight of propagation loss. |
| `confidence_ratio` | `2.0` | Required target-to-strongest-competitor probability ratio. A value of `2.0` corresponds to a logit margin of `log(2)`. |
| `penalty_desired_nonflip` | `3.0` | Propagation penalty when a desired position does not change from the factual prediction to the desired activity. |
| `penalty_desired_wrongflip` | `5.0` | Propagation penalty when a desired position changes to an activity other than the desired activity. |
| `penalty_undesired_flip` | `2.0` | Propagation penalty for a prediction change at an unwarranted position. |
| `penalty_flexible_wrongflip` | `2.0` | Propagation penalty when a permissible position changes to an activity outside its permissible set. |
| `use_fitness_sharing` | `True` | Enables fitness sharing to maintain population diversity. |
| `niche_sigma_quantile` | `0.15` | Quantile of pairwise candidate distances used to determine the fitness-sharing niche radius. |
| `niche_alpha` | `1.0` | Shape parameter of the sharing function. A value of `1.0` gives linear decay. |
| `elite_k` | `1` | Number of highest-quality candidates preserved unchanged between generations. |
| `tournament_size` | `3` | Number of candidates participating in tournament selection. |
| `crossover_rate_min` | `0.5` | Minimum adaptive crossover probability. |
| `crossover_rate_max` | `0.9` | Maximum adaptive crossover probability. |
| `mutation_rate_min` | `0.05` | Minimum adaptive mutation probability. |
| `mutation_rate_max` | `0.3` | Maximum adaptive mutation probability. |
| `implaus_sample_ratio` | `1.0` | Proportion of training instances sampled as candidates for the nearest training instance used to compute implausibility. |
| `output_population_size` | `20` | Maximum number of counterfactual candidates returned for each query. |

The random-initialization proportion is determined by

```text
1 - cbi_ratio - sbi_ratio
```

and is therefore `0.84` under the default configuration.

### DiCE4EL and DiCE4EL+ Configuration

| Parameter | Default | Description |
|---|---:|---|
| `output_population_size` | `1` | Number of counterfactuals returned for each query. The implementation requires this value to be `1`. |
| `mode` | `"multi"` | Target-position mode (extended for multiple desired positions). Supported values are `"multi"` and `"last"`. |
| `cbi_ratio` | `0.08` | Proportion used for case-based initialization, where applicable. |
| `sbi_ratio` | `0.08` | Proportion used for sampling-based initialization, where applicable. |
| `learning_rate` | `0.01` | Learning rate used by the optimizer. |
| `optimization_steps` | `500` | Maximum number of optimization steps. |
| `early_stopping_min_steps` | `100` | Minimum number of optimization steps before early stopping can occur. |
| `early_stopping_patience` | `50` | Number of optimization steps without improvement before early stopping. |
| `sparsity_pop_ratio` | `1.0` | Proportion used for sparse initialization, where applicable. |
| `sparsity_feature_p` | `0.0` | Probability of modifying each intervenable attribute during sparse initialization. A value of `0.0` initializes the counterfactual with the factual trace values. |
| `w_margin_loss` | `1.0` | Weight of the margin loss. |
| `w_scenario_loss` | `1.0` | Weight of the scenario-model plausibility loss. |
| `w_distance_loss` | `1.0` | Weight of the distance loss. |
| `w_cat_loss` | `1.0` | Weight of the categorical constraint loss. |
| `confidence_ratio` | `2.0` | Required target-to-strongest-competitor probability ratio. A value of `2.0` corresponds to a logit margin of `log(2)`. |
| `use_sampling` | `False` | Enables stochastic categorical sampling in DiCE4EL; otherwise, deterministic argmax projection is used. |
| `initial_logit_scale` | `0.5` | Controls how strongly categorical optimization is initialized toward the factual category. |
| `gumbel_initial_temperature` | `2.0` | Starting temperature controlling how soft the categorical choices are. |
| `gumbel_min_temperature` | `0.5` | Lowest temperature allowed as categorical choices become more discrete. |
| `gumbel_anneal_rate` | `0.998` | Rate at which the temperature decreases after each optimization step. |
| `gumbel_hard` | `False` | If `True`, uses one-hot categorical choices; if `False`, uses soft categorical choices during optimization. |
| `implaus_sample_ratio` | `1.0` | Proportion of training instances sampled as candidates for the nearest training instance used to compute implausibility. |

The following parameters in `EventLogDiCEConfig` are used to evaluate generated counterfactuals using the common evaluation criteria:

| Parameter | Default | Description |
|---|---:|---|
| `w_distance` | `1.0` | Weight of intervention distance in the common evaluation. |
| `w_sparsity` | `1.0` | Weight of sparsity in the common evaluation. |
| `w_margin` | `1.0` | Weight of margin loss in the common evaluation. |
| `w_process_violation` | `1.0` | Weight of propagation loss in the common evaluation. |
| `penalty_desired_nonflip` | `3.0` | Penalty for a desired non-flip. |
| `penalty_desired_wrongflip` | `5.0` | Penalty for an incorrect change at a desired position. |
| `penalty_undesired_flip` | `2.0` | Penalty for an unwarranted prediction change. |
| `penalty_flexible_wrongflip` | `2.0` | Penalty for a change outside the permissible activity set at a permissible position. |

### Changing Configuration Parameters

Configuration parameters can be overridden when creating the configuration object. Only the parameters that differ from the defaults need to be specified.

For example, to run ProACTS with a larger population and more generations:

```python
config = GAConfig(
    population_size=100,
    offspring_size=400,
    num_generations=300,
)
```

To change the propagation penalties:

```python
config = GAConfig(
    penalty_desired_nonflip=3.0,
    penalty_desired_wrongflip=5.0,
    penalty_undesired_flip=2.0,
    penalty_flexible_wrongflip=2.0,
)
```

To run DiCE4EL with stochastic categorical sampling:

```python
config = EventLogDiCEConfig(
    use_sampling=True,
)
```

When `use_sampling=True`, DiCE4EL is stochastic and should be evaluated across the specified random seeds.

To configure the Gumbel--Softmax optimization used by DiCE4EL+:

```python
config = EventLogDiCEConfig(
    initial_logit_scale=0.5,
    gumbel_initial_temperature=2.0,
    gumbel_min_temperature=0.5,
    gumbel_anneal_rate=0.998,
    gumbel_hard=False,
)
```

## Experiments

The experimental pipeline consists of:

1. Preprocess the event logs.

2. Train the next-activity prediction model.  
   Run: `experiments/<dataset>/<dataset>-model.ipynb`

3. Train the scenario model used by DiCE4EL and DiCE4EL+.  
   Run: `experiments/<dataset>/dice4el/<dataset>-scenario_model.ipynb`

4. Generate counterfactual queries.

5. Generate counterfactual sequences using ProACTS.  
   Run: `experiments/<dataset>/<dataset>-cf_generated_experiments_ga.ipynb`

6. Generate counterfactual sequences using DiCE4EL and DiCE4EL+.  
   Run: `experiments/<dataset>/dice4el/<dataset>-cf_generated_experiments_dice4el.ipynb`

7. Repeat the ProACTS counterfactual experiments for seeds `13`, `42`, and `777`.  
   Run: `experiments/<dataset>/seed_experiments/<dataset>-cf_seed_experiments_ga.ipynb`

   Alternatively, run `experiments/<dataset>/<dataset>-cf_generated_experiments_ga.ipynb` for each seed by:
   - setting `set_seed(seed=seed)` in cell `[2]`;
   - updating the log filename in cell `[16]`; and
   - setting `technique="GA_multiple_desired_seed{}"` with the corresponding seed in the result save paths in cells `[29]` and `[31]`.

8. Repeat the DiCE4EL+ counterfactual experiments for seeds `13`, `42`, and `777`.  
   Run: `experiments/<dataset>/dice4el/seed_experiments/<dataset>-cf_seed_experiments_dice4el.ipynb`

   DiCE4EL also requires seed repetitions when `use_sampling=True`; when `use_sampling=False`, categorical values are selected using deterministic argmax projection.

9. Generate counterfactual sequences using the ProACTS ablated variant (without the propagation-loss objective).  
   Run: `experiments/<dataset>/seed_experiments/Ablated/<dataset>-cf_ablated_experiments_ga.ipynb`

10. Evaluate the generated counterfactuals, perform statistical comparisons, and generate the reported tables and figures.  
   Run: `experiments/summary_evaluation.ipynb`

Replace `<dataset>` with the corresponding dataset directory.

### Evaluation Metrics

The evaluation considers:

- **Target success** – whether the desired activity has a target-to-strongest-competitor probability ratio of at least `1.0`;
- **Margin success** – whether the desired activity has a target-to-strongest-competitor probability ratio of at least `2.0`;
- **Propagation loss** – penalizes undesired effects of the intervention across the predicted trace suffix while accounting for desired and permissible activity changes;
- **Intervention distance** – measures the magnitude of the changes between the factual and counterfactual instances;
- **Sparsity** – measures the number of intervened attribute-position pairs; and
- **Runtime** – measures the optimization time required to generate a counterfactual.

The optimization margin uses `confidence_ratio=2.0`, corresponding to a logit margin of `log(2)`. Target success is evaluated at the ordinary decision boundary (target-to-strongest-competitor probability ratio approximately `1.0`).

Experiments involving stochastic counterfactual generation are repeated using three random seeds: `13`, `42`, and `777`.

## Reproducing the Results

All artifacts required to reproduce the reported evaluation are included in the repository. This includes the trained predictive and scenario models, counterfactual queries, and generated counterfactual results for all methods and seeds.

To reproduce the reported results, run:

`experiments/summary_evaluation.ipynb`

This notebook performs the final aggregation and evaluation, statistical analysis, and qualitative analysis. All resulting outputs, including the final results, statistical comparisons, tables, figures, and qualitative-analysis outputs, are saved to:

`experiments/summary_evaluation/`

The provided experimental artifacts are organized as follows:

| Artifact | Location |
|---|---|
| Trained predictive model and configs | `experiments/<dataset>/pretrained_models/` |
| Trained scenario model and configs | `experiments/<dataset>/dice4el/pretrained_models/` |
| Counterfactual queries | `experiments/<dataset>/experiments/` |
| ProACTS results | `experiments/<dataset>/generated_experiment_results/` |
| ProACTS logs | `experiments/<dataset>/logs/` |
| ProACTS seed-experiment results | `experiments/<dataset>/seed_experiments/generated_experiment_results/` |
| ProACTS seed-experiment logs | `experiments/<dataset>/seed_experiments/logs/` |
| ProACTS ablated-experiment results | `experiments/<dataset>/seed_experiments/Ablated/generated_experiment_results/` |
| ProACTS ablated-experiment logs | `experiments/<dataset>/seed_experiments/Ablated/logs/` |
| DiCE4EL and DiCE4EL+ results | `experiments/<dataset>/dice4el/generated_experiment_results/` |
| DiCE4EL and DiCE4EL+ logs | `experiments/<dataset>/dice4el/logs/` |
| DiCE4EL and DiCE4EL+ seed-experiment results | `experiments/<dataset>/dice4el/seed_experiments/generated_experiment_results/` |
| DiCE4EL and DiCE4EL+ seed-experiment logs | `experiments/<dataset>/dice4el/seed_experiments/logs/` |

## Citation

If you use this repository, please cite the ProACTS paper.

Citation information will be added following publication.