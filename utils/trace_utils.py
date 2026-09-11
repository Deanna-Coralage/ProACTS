import numpy as np

import copy

from typing import List, Optional

from process.constraint import ProcessConstraint



# --- Rollout positions for autoregressive simulation ---
def get_rollout_positions(
    trace:        np.ndarray,
    cf_pop:       np.ndarray,
) -> np.ndarray:  # (n, )

    """
    Returns earliest timestep where cf differs from trace.
    -1 indicates no change.
    """

    changed = np.not_equal(cf_pop, trace[None, :, :]).any(axis=2)

    time_idx = np.arange(changed.shape[1])

    masked = np.where(changed, time_idx, np.inf)

    rollout_positions = np.min(masked, axis=1)

    rollout_positions = np.where(np.isinf(rollout_positions), -1, rollout_positions).astype(np.int32)

    return rollout_positions



# --- Handle model predictions vs original trace ---
def add_simulation_mismatches_as_flexible(
    original_trace:        np.ndarray,
    simulated_trace:       np.ndarray,
    activity_idx:          int,
    flexible_constraints:  ProcessConstraint,
) -> ProcessConstraint:
    
    """
    Return a new ProcessConstraint in which positions where the baseline
    autoregressive simulation differs from the original trace are added as
    single-position flexible constraints.
    """

    updated_constraints = copy.deepcopy(flexible_constraints)

    T = min(len(original_trace), len(simulated_trace))

    for pos in range(1, T):

        original_act = str(original_trace[pos, activity_idx])
        simulated_act = str(simulated_trace[pos, activity_idx])

        if original_act != simulated_act:
            updated_constraints.insert(
                start=pos,
                end=pos,
                allowed=[original_act, simulated_act],
            )

    return updated_constraints
