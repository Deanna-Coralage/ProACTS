import numpy as np

import torch
from torch.utils.data import Dataset

from typing import Dict, List, Union

from dataclasses import dataclass



@dataclass
class ProcessDataset(Dataset):

    """
    Dataset for training process-aware sequence model.

    Each sample corresponds to a single process case (trace), preserving
    the full temporal structure required for next-event prediction with LSTMs.

    Data Organization:
    - Dynamic categorical features:
        Represent event-level categorical attributes (e.g., activity, resource).
        Each feature is stored as a separate sequence and later embedded in the model.

    - Dynamic continuous features:
        Event-level numerical attributes (e.g., time_delta, cost, amount),
        represented as (seq_len, num_features) tensors.

    - Static categorical features:
        Case-level categorical attributes (e.g., loan_goal),
        shared across all events in a sequence.

    - Static continuous features:
        Case-level numerical attributes (if any), broadcasted per case.

    - Labels:
        Next-event targets for each timestep in the sequence.
        Used for sequence-to-sequence supervision.

    - Mask:
        Binary indicator identifying valid timesteps (1 = valid event,
        0 = padding or end-of-case token). Ensures loss is computed only
        over meaningful process steps.

    Returns:
        A dictionary containing:
            - dynamic_cat: list of (seq_len,) tensors
            - dynamic_cont: (seq_len, num_cont_features)
            - static_cat: list of (1,) tensors
            - static_cont: (num_static_cont_features,)
            - labels: (seq_len,)
            - mask: (seq_len,)
    """

    def __init__(
        self,
        dynamic_cat:   List[np.ndarray],     
        dynamic_cont:  np.ndarray,          
        static_cat:    List[np.ndarray],    
        static_cont:   np.ndarray,          
        labels:        np.ndarray,       
        mask:          np.ndarray       
    ):

        self.dynamic_cat = [torch.tensor(x, dtype=torch.long) for x in dynamic_cat]
        self.dynamic_cont = torch.tensor(dynamic_cont, dtype=torch.float32)

        self.static_cat = [torch.tensor(x, dtype=torch.long) for x in static_cat]
        self.static_cont = torch.tensor(static_cont, dtype=torch.float32)

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.bool)

    
    def __len__(self):
        return self.labels.shape[0]

        
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:

        return {
            "dynamic_cat": [x[idx] for x in self.dynamic_cat],  
            "dynamic_cont": self.dynamic_cont[idx],             
            "static_cat": [x[idx] for x in self.static_cat],   
            "static_cont": self.static_cont[idx],            
            "labels": self.labels[idx],                   
            "mask": self.mask[idx]                          
        }
