import numpy as np

import torch
from torch.utils.data import Dataset

from typing import Dict, List, Union



class ScenarioDataset(Dataset):

    """
    Dataset for ScenarioLSTM training.
    Designed for:
        ScenarioLSTM(cat_x: List[Tensor], cont_x: Tensor, mask)
    """

    def __init__(
        self,
        cat:      List[np.ndarray],     # list of (B, T)
        cont:     np.ndarray,           # (B, T, C)
        labels:   np.ndarray,           # (B, T)
        mask:     np.ndarray            # (B, T)
    ):

        self.cat = [torch.tensor(x, dtype=torch.long) for x in cat]
        self.cont = torch.tensor(cont, dtype=torch.float32)

        self.labels = torch.tensor(labels, dtype=torch.float32)

        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:

        return {
            "cat": [x[idx] for x in self.cat], 
            "cont": self.cont[idx],    
            "labels": self.labels[idx], 
            "mask": self.mask[idx]     
        }
