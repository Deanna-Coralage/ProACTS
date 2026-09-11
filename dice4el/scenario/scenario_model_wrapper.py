import numpy as np

import torch

from dataclasses import dataclass

from typing import Dict, List, Tuple, Union, Optional

from dice4el.scenario.scenario_model import ScenarioLSTM
from dice4el.scenario.scenario_handler import ScenarioHandler



@dataclass(frozen=True)
class ScenarioModelWrapper:
    
    """
    Wraps scenario model and handles:
      - inference (on model-ready inputs - population (B, T, F), single prefix (T, F)
    """

    scenario_model:      ScenarioLSTM          
    scenario_handler:    ScenarioHandler
    device:              str = "cpu"

    
    def __post_init__(self):
        model = self.scenario_model.to(self.device)
        model.eval()
    
        for param in model.parameters():
            param.requires_grad_(False)

        object.__setattr__(self, "scenario_model", model)

    @property
    def differentiable(self):
        return ScenarioModelWrapper._Differentiable(wrapper=self)


    def _to_tensor(self, arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        
        """Convert numpy array to float32 tensor on a wrapper device."""
        
        return torch.as_tensor(arr, dtype=dtype, device=self.device)


    def _prepare_model_inputs(
        self, 
        cat:    List[np.ndarray], 
        cont:   np.ndarray, 
        mask:   Optional[np.ndarray] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        
        """Converts incoming numpy components directly into model-ready device tensors."""
        
        cat_tensor = [self._to_tensor(x, dtype=torch.long) for x in cat]
        cont_tensor = self._to_tensor(cont, dtype=torch.float32)
            
        if mask is not None:
            mask_tensor = self._to_tensor(mask, dtype=torch.float32)
        else:
            mask_tensor = None
    
        return cat_tensor, cont_tensor, mask_tensor


    def _forward_logits(
        self, 
        cat:    List[torch.Tensor], 
        cont:   torch.Tensor,
        mask:   Optional[torch.Tensor] = None
    ) -> np.ndarray:
        
        """Raw logits — shape (B, C)"""
        
        with torch.no_grad():
            logits = self.scenario_model(
                cat=cat,
                cont=cont,
                mask=mask
            ) # (B, 1)

        B, T, _ = logits.shape
        current_device = logits.device
    
        if mask is not None:
            lengths = mask.to(current_device).long().sum(dim=1)
        else:
            lengths = torch.full((B,), T, dtype=torch.long, device=current_device)
    
        lengths = lengths.clamp(min=1) - 1  # avoid -1 indexing
    
        batch_idx = torch.arange(B, device=current_device)
    
        last_logits = logits[batch_idx, lengths, 0]  # (B,)
    
        return last_logits.cpu().numpy()
        

    # --- Simulation ---
    def simulate_forward(
        self, 
        prefix:          np.ndarray,
    ) -> np.ndarray:
        
        """
        Runs a single forward step during simulation.
        """

        if prefix is None or prefix.shape[0] < 1:
            raise ValueError(
                f"Invalid prefix: expected (T, F) with T>=1, got {None if prefix is None else list(prefix.shape)}"
            )
        
        model_inputs = self.scenario_handler.build_scenario_model_input(prefix)
        cat_tensor, cont_tensor, mask_tensor = self._prepare_model_inputs(
            cat=model_inputs["cat"], 
            cont=model_inputs["cont"], 
            mask=model_inputs["mask"]
        )
        
        logits = self._forward_logits(
            cat=cat_tensor,
            cont=cont_tensor,
            mask=mask_tensor
        )

        return logits


    # --- Differentiable wrapper ---
    @dataclass(frozen=True)
    class _Differentiable:
    
        wrapper: "ScenarioModelWrapper"
    
        def _forward_logits(
            self,
            cat:   List[torch.Tensor],   # each: (B, T, K_i)
            cont:  torch.Tensor,         # (B, T, C)
            mask:  Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            
            """
            Return full-sequence differentiable scenario logits.

            Returns
            -------
            torch.Tensor
                Shape (B, T, 1).
            """

            model = self.wrapper.scenario_model
            model.eval()

            with torch.set_grad_enabled(True):
                with torch.backends.cudnn.flags(enabled=False):
                    logits = model.forward_differentiable(
                        cat=[
                            x.to(
                                device=self.wrapper.device,
                                dtype=torch.float32,
                            )
                            for x in cat
                        ],
                        cont=cont.to(
                            device=self.wrapper.device,
                            dtype=torch.float32,
                        ),
                        mask=(
                            mask.to(self.wrapper.device)
                            if mask is not None
                            else None
                        ),
                    )

            return logits
            

        def forward_last_state_logits(
            self,
            scenario_model_input:    Dict[str, Union[List[torch.Tensor], torch.Tensor, None]],
        ) -> torch.Tensor:
            
            """
            Last valid scenario logit.
    
            Returns:
                last_logits: (B,)
            """

            mask = scenario_model_input["mask"]
    
            logits = self._forward_logits(
                cat=scenario_model_input["cat"],
                cont=scenario_model_input["cont"],
                mask=mask,
            )  # (B, T, 1)
    
            device = logits.device
            B, T, _ = logits.shape
    
            if mask is not None:
                lengths = mask.to(device).bool().sum(dim=1)
            else:
                lengths = torch.full(
                    (B,),
                    T,
                    dtype=torch.long,
                    device=device,
                )
    
            lengths = lengths.clamp(min=1) - 1
            batch_idx = torch.arange(B, device=device)
    
            return logits[batch_idx, lengths, 0]  # (B,)


        def forward_sequence_logits(
            self,
            scenario_model_input:    Dict[str, Union[List[torch.Tensor], torch.Tensor, None]],
        ) -> torch.Tensor:
            
            """
            Scenario logits at selected positions.
    
            Returns:
                position_logits: (B, P)
            """
    
            logits = self._forward_logits(
                cat=scenario_model_input["cat"],
                cont=scenario_model_input["cont"],
                mask=scenario_model_input["mask"],
            )  # (B, T, 1)
    
            return logits.squeeze(-1)    # (B, T)
