import os
import sys
from pathlib import Path

import torch

import random

import numpy as np
import pandas as pd

from typing import Dict, Tuple, Optional



# --- Forward stdout ---
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def set_stdout_to_file(
    filepath: str = "logs/global_output.txt",
    console:  bool = True,
):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    original_stdout = sys.stdout
    log_file = open(filepath, "w")

    if console:
        sys.stdout = Tee(original_stdout, log_file)
    else:
        sys.stdout = log_file

    return log_file, original_stdout



# --- Global seed for models ---
def set_seed(
    seed: int = 42
):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def save_counterfactual_results(
    output_path:           str,
    fitness_components:    Dict[str, np.ndarray],
    evaluation:            Dict[str, float],
    trace:                 np.ndarray,         # (T, F)
    cf_pop:                np.ndarray,         # (N,T,F)
    simulated_trace:       np.ndarray,         # (T,F)
    simulated_cfs:         np.ndarray,         # (N,T,F)
    simulated_cfs_na:      np.ndarray,         # (N,T,F)
    feature_order:         Tuple[str, ...]
) -> None:
    
    """
    Save counterfactual search results to Excel.

    Sheets:
        evaluation            - General stats + top performer's component breakdown
        cf_fitness_analysis   - Row number and full fitness breakdown per counterfactual individual
        trace
        counterfactuals
        simulated_cfs
    """

    feature_names = list(feature_order)

    def flatten_trace_population(
        traces:               np.ndarray,
        fitness_components:   Optional[Dict[str, np.ndarray]] = None
    ) -> pd.DataFrame:

        rows = []

        if traces.ndim == 2:
            traces = traces[None, :, :]

        n, t, f = traces.shape

        for cf_idx in range(n):
            for pos in range(t):

                row = {
                    "cf_id": cf_idx,
                    "position": pos,
                }

                for feat_idx, feat_name in enumerate(feature_names):
                    row[feat_name] = traces[cf_idx, pos, feat_idx]

                if fitness_components is not None and pos == 0:
                    for component, values in fitness_components.items():
                        row[component] = float(values[cf_idx])

                rows.append(row)

        return pd.DataFrame(rows)

    # --- Append Best Fitness Components to Evaluation ---
    evaluation_extended = evaluation.copy()
    for component, values in fitness_components.items():
        # Because the population is sorted, index 0 is always your top performer
        evaluation_extended[f"best_cf_{component}"] = float(values[0])
    
    evaluation_df = pd.DataFrame([evaluation_extended])

    # --- Build the New Standalone CF Fitness Components Analysis Sheet ---
    # This transforms your dictionary of arrays into a tabular row-per-individual format
    cf_analysis_rows = []
    num_individuals = len(next(iter(fitness_components.values())))

    for cf_idx in range(num_individuals):
        analysis_row = {
            "row_idx": cf_idx,  # Automatically generated number from 0..N
            "cf_id": cf_idx,
        }
        for component, values in fitness_components.items():
            analysis_row[component] = float(values[cf_idx])
            
        cf_analysis_rows.append(analysis_row)
        
    cf_fitness_analysis_df = pd.DataFrame(cf_analysis_rows)

    # --- Process Structural Sequence Frames ---
    trace_df = flatten_trace_population(trace)
    simulated_trace_df = flatten_trace_population(simulated_trace)
    cf_df = flatten_trace_population(cf_pop, fitness_components)
    simulated_df = flatten_trace_population(simulated_cfs, fitness_components)
    simulated_df_na = flatten_trace_population(simulated_cfs_na, fitness_components)

    # --- Write Excel Workbook ---
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl"
    ) as writer:

        evaluation_df.to_excel(
            writer,
            sheet_name="evaluation",
            index=False
        )

        cf_fitness_analysis_df.to_excel(
            writer,
            sheet_name="cf_fitness_analysis",
            index=False
        )

        trace_df.to_excel(
            writer,
            sheet_name="trace",
            index=False
        )

        simulated_trace_df.to_excel(
            writer,
            sheet_name="simulated_trace",
            index=False
        )

        cf_df.to_excel(
            writer,
            sheet_name="counterfactuals",
            index=False
        )

        simulated_df.to_excel(
            writer,
            sheet_name="simulated_counterfactuals",
            index=False
        )

        simulated_df_na.to_excel(
            writer,
            sheet_name="simulated_counterfactuals_na",
            index=False
        )


    print(f"Saved results to {output_path}")
