import numpy as np

import torch
import torch.nn as nn

from dataclasses import dataclass

from collections import defaultdict

import copy

from typing import Dict, List, Tuple, Union, Optional

from model.next_event_model import ProcessLSTM
from model.preprocessor import PreprocessorArtifacts

from process.constraint import ProcessConstraint



@dataclass(frozen=True)
class ModelWrapper:
    
    """
    Wraps sequence model and handles:
      - inference (on model-ready inputs - population (B, T, F), single prefix (T, F)
      - process-aware constrained autoregressive simulation
    """

    model:                   ProcessLSTM          
    preprocessor_artifacts:  PreprocessorArtifacts
    device:                  str = "cpu"

    
    def __post_init__(self):
        model = self.model.to(self.device)
        model.eval()
    
        for param in model.parameters():
            param.requires_grad_(False)

        object.__setattr__(self, "model", model)

    
    @property
    def differentiable(self):
        return ModelWrapper._Differentiable(wrapper=self)
        
    
    # --- Tensor conversion ---
    def _to_tensor(self, arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        
        """Convert numpy array to float32 tensor on a wrapper device."""
        
        return torch.as_tensor(arr, dtype=dtype, device=self.device)


    def _prepare_model_inputs(
        self, 
        dynamic_cat:   List[np.ndarray], 
        dynamic_cont:  np.ndarray, 
        static_cat:    List[np.ndarray], 
        static_cont:   np.ndarray, 
        mask:          Optional[np.ndarray] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        
        """Converts incoming numpy components directly into model-ready device tensors."""
        
        dynamic_cat_tensor = [self._to_tensor(x, dtype=torch.long) for x in dynamic_cat]
        dynamic_cont_tensor = self._to_tensor(dynamic_cont, dtype=torch.float32)
        static_cat_tensor = [self._to_tensor(x, dtype=torch.long) for x in static_cat]
        static_cont_tensor = self._to_tensor(static_cont, dtype=torch.float32)
            
        if mask is not None:
            mask_tensor = self._to_tensor(mask, dtype=torch.float32)
        else:
            mask_tensor = None
    
        return dynamic_cat_tensor, dynamic_cont_tensor, static_cat_tensor, static_cont_tensor, mask_tensor


    # --- Forward passes ---
    def _forward_proba(
        self, 
        dynamic_cat:   List[torch.Tensor], 
        dynamic_cont:  torch.Tensor, 
        static_cat:    List[torch.Tensor], 
        static_cont:   torch.Tensor, 
        mask:          Optional[torch.Tensor] = None
    ) -> np.ndarray:

        """Softmax probabilities — shape (B, C)"""
    
        with torch.no_grad():
            logits = self.model(
                dynamic_cat=dynamic_cat,
                dynamic_cont=dynamic_cont,
                static_cat=static_cat,
                static_cont=static_cont,
                mask=mask
            ) # (B, T, C)

        current_device = logits.device

        # Only last state of prefix needed for inference
        if mask is not None:
            lengths = mask.to(current_device).bool().sum(dim=1)
        else:
            # If no mask, every batch element uses the final time step step
            lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=current_device)
            
        last_step_logits = logits[torch.arange(logits.size(0), device=current_device), lengths - 1]  # (B, C)
        
        return torch.softmax(last_step_logits, dim=-1).cpu().numpy()
                
            
    def _forward_logits(
        self, 
        dynamic_cat:   List[torch.Tensor], 
        dynamic_cont:  torch.Tensor, 
        static_cat:    List[torch.Tensor], 
        static_cont:   torch.Tensor, 
        mask:          Optional[torch.Tensor] = None
    ) -> np.ndarray:
        
        """Raw logits — shape (B, C)"""
        
        with torch.no_grad():
            logits = self.model(
                dynamic_cat=dynamic_cat,
                dynamic_cont=dynamic_cont,
                static_cat=static_cat,
                static_cont=static_cont,
                mask=mask
            )

        current_device = logits.device

        if mask is not None:
            lengths = mask.to(current_device).bool().sum(dim=1)
        else:
            lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=current_device)
            
        last_step_logits = logits[torch.arange(logits.size(0), device=current_device), lengths - 1]

        return last_step_logits.cpu().numpy()


    # --- Simulation ---
    def simulate_forward(
        self, 
        prefix:          np.ndarray,
    ) -> Dict[str, Union[np.ndarray, List[Dict[str, float]]]]:
        
        """
        Runs a single forward step during simulation.
        """

        if prefix is None or prefix.shape[0] < 1:
            raise ValueError(
                f"Invalid prefix: expected (T, F) with T>=1, got {None if prefix is None else list(prefix.shape)}"
            )
        
        model_inputs = self.preprocessor_artifacts.build_model_input(prefix)
        dynamic_cat_tensor, dynamic_cont_tensor, static_cat_tensor, static_cont_tensor, mask_tensor = self._prepare_model_inputs(
            dynamic_cat=model_inputs["dynamic_cat"], 
            dynamic_cont=model_inputs["dynamic_cont"], 
            static_cat=model_inputs["static_cat"], 
            static_cont=model_inputs["static_cont"], 
            mask=model_inputs["mask"]
        )
        
        logits = self._forward_logits(
            dynamic_cat=dynamic_cat_tensor,
            dynamic_cont=dynamic_cont_tensor,
            static_cat=static_cat_tensor,
            static_cont=static_cont_tensor,
            mask=mask_tensor
        ) # (B, C)

        predictions = np.argmax(logits, axis=-1).astype(int) # Shape: (B,)

        decoded_predictions = self.preprocessor_artifacts.decode_feature(
            self.preprocessor_artifacts.feature_config.activity_feature,
            predictions
        )

        # --- Decoded class labels ---
        class_ids = np.arange(logits.shape[1])
        class_labels = self.preprocessor_artifacts.decode_feature(
            self.preprocessor_artifacts.feature_config.activity_feature,
            class_ids
        )

        logits_labeled = [
            {
                class_labels[c]: float(logits[b, c])
                for c in range(logits.shape[1])
            }
            for b in range(logits.shape[0])
        ]

        # --- Return raw logits and next activities ---
        return {
            "logits": logits_labeled,                     # (B, C)
            "next_activity": decoded_predictions          # (B,)
        }


    # --- Process-aware autoregressive simulation ---
    def simulate_forward_autoregressive(
        self,
        traces:                np.ndarray,  # (B, T, F) or (T, F)
        rollout_positions:     np.ndarray,  # (B,) or scalar
        desired_positions:     List[int],
        flexible_constraints:  Optional[ProcessConstraint] = None,
    ) -> Dict[str, Union[np.ndarray, List[Dict[int, Dict[str, float]]]]]:
    
        """
        Autoregressively simulate each trace after its rollout position.
    
        flexible_constraints:
            Genuine process-model flexibility, such as parallel activities.
            If the predicted activity is valid and already appears later in the
            flexible segment, its full event row is reused. Otherwise, an
            activity-specific prototype is used.
    
        desired_positions:
            Event positions whose prediction logits are stored for margin loss.
    
        The simulated trace always follows the model's predicted activity.
        Simulation-mismatch constraints should be handled separately during
        process-violation evaluation.
        """
    
        # --- Normalize trace batch ---
        if traces.ndim == 2:
            traces = traces[None, ...]
    
        if traces.ndim != 3:
            raise ValueError(
                f"Expected traces with shape (T, F) or (B, T, F), "
                f"got {traces.shape}."
            )
    
        B, T, _ = traces.shape
    
        # --- Normalize rollout positions ---
        rollout_positions = np.asarray(
            rollout_positions,
            dtype=np.int64,
        ).reshape(-1)
    
        if rollout_positions.size == 1 and B > 1:
            rollout_positions = np.repeat(
                rollout_positions,
                B,
            )
    
        if rollout_positions.size != B:
            raise ValueError(
                f"Expected {B} rollout positions, "
                f"but received {rollout_positions.size}."
            )
    
        pre_art = self.preprocessor_artifacts
        feature_config = pre_art.feature_config
        activity_idx = feature_config.feature_to_idx[
            feature_config.activity_feature
        ]
    
        desired_pos_set = set(desired_positions)
    
        # Flexible state.
        flexible_states: List[Optional[ProcessConstraint]] = [
            copy.deepcopy(flexible_constraints)
            if flexible_constraints is not None
            else None
            for _ in range(B)
        ]
    
        # Mutable rollout state.
        current = traces.copy()
    
        # Logits predicting each desired event position.
        logits_store: List[Dict[int, Dict[str, float]]] = [
            {} for _ in range(B)
        ]
    
        # Group candidates according to the timesteps they need simulated.
        batches: Dict[int, List[int]] = defaultdict(list)
    
        for b in range(B):
            rollout_pos = int(rollout_positions[b])
            start_t = 1 if rollout_pos < 0 else rollout_pos + 1
    
            for t in range(start_t, T):
                batches[t].append(b)
    
        # --- Autoregressive rollout ---
        for t in sorted(batches):
    
            batch_ids = batches[t]
    
            if not batch_ids:
                continue
    
            # Prefix [:t] predicts the event at position t.
            prefixes = np.stack(
                [current[b, :t] for b in batch_ids],
                axis=0,
            )
    
            forward_sim = self.simulate_forward(prefixes)
    
            logits = forward_sim["logits"]
            next_activities = forward_sim["next_activity"]
    
            for i, b in enumerate(batch_ids):
    
                pred_act = next_activities[i]
                current_act = current[b, t, activity_idx]
    
                # Store logits under the event position they predict.
                if t in desired_pos_set:
                    logits_store[b][t] = logits[i]
    
                # Check and consume genuine flexible constraints.
                flex_state = flexible_states[b]
    
                if flex_state is not None:
                    is_flex, flex_ok, flex_segment = (
                        flex_state.check_constrained_position(
                            current_pos=t, activity=pred_act
                        )
                    )
                else:
                    is_flex, flex_ok, flex_segment = (False, False, None)
    
                # 1. Predicted activity already matches the existing event.
                if pred_act == current_act:
                    next_event = current[b, t].copy()
    
                # 2. Predicted activity is valid within a flexible segment.
                elif is_flex and flex_ok and flex_segment is not None:
                    _, end = flex_segment
                    next_event = None
    
                    # Reuse a matching event row from the remaining segment.
                    for j in range(t, end + 1):
                        if current[b, j, activity_idx] == pred_act:
                            next_event = current[b, j].copy()
                            break
    
                    # The predicted activity is allowed, but no matching event
                    # row remains in the current segment.
                    if next_event is None:
                        next_event = pre_art.get_event_prototype(
                            pred_act,
                            current[b, t],
                        )
    
                # 3. Ordinary prediction or invalid flexible prediction.
                else:
                    next_event = pre_art.get_event_prototype(
                        pred_act,
                        current[b, t],
                    )
    
                current[b, t] = next_event
    
        return {
            "simulated_traces": current,
            "logits_store": logits_store,
        }


    # --- Process-aware non-autoregressive simulation ---
    def simulate_forward_nonautoregressive(
        self,
        traces:             np.ndarray,
        desired_positions:  List[int],
    ) -> Dict[str, Union[np.ndarray, List[Dict[int, Dict[str, float]]]]]:
    
        """
        Predict next activity at desired positions only.
        No autoregressive rollout.
        """
    
        if traces.ndim == 2:
            traces = traces[None, ...]

        if traces.ndim != 3:
            raise ValueError(
                f"Expected traces with shape (T, F) or (B, T, F), "
                f"got {traces.shape}."
            )
    
        B, T, _ = traces.shape
    
        predictable_positions = [
            position
            for position in desired_positions
            if 1 <= position < T
        ]
    
        pre_art = self.preprocessor_artifacts
        f_cfg = pre_art.feature_config
        activity_idx = f_cfg.feature_to_idx[f_cfg.activity_feature]
    
        outputs = traces.copy()
    
        logits_store: List[Dict[int, Dict[str, float]]] = [
            {} for _ in range(B)
        ]
    
        for t in sorted(predictable_positions):
    
            prefixes = traces[:, :t, :]
    
            forward_sim = self.simulate_forward(prefixes)
            logits = forward_sim["logits"]
            next_activity = forward_sim["next_activity"]
    
            for b in range(B):
                logits_store[b][t] = logits[b]
    
                next_event = traces[b, t].copy()
                next_event[activity_idx] = next_activity[b]
    
                outputs[b, t] = next_event
    
        return {
            "simulated_traces": outputs,
            "logits_store": logits_store,
        }


    # --- Differentiable wrapper ---
    @dataclass(frozen=True)
    class _Differentiable:
    
        wrapper: "ModelWrapper"
    
        def _forward_logits(
            self,
            dynamic_cat:   List[torch.Tensor],
            dynamic_cont:  torch.Tensor,
            static_cat:    List[torch.Tensor],
            static_cont:   torch.Tensor,
            mask:          Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        
            model = self.wrapper.model
            model.eval()
        
            with torch.set_grad_enabled(True):
                with torch.backends.cudnn.flags(enabled=False):
                    logits = model.forward_differentiable(
                        dynamic_cat=[
                            x.to(device=self.wrapper.device, dtype=torch.float32)
                            for x in dynamic_cat
                        ],
                        dynamic_cont=dynamic_cont.to(
                            device=self.wrapper.device,
                            dtype=torch.float32,
                        ),
                        static_cat=[
                            x.to(device=self.wrapper.device, dtype=torch.float32)
                            for x in static_cat
                        ],
                        static_cont=static_cont.to(
                            device=self.wrapper.device,
                            dtype=torch.float32,
                        ),
                        mask=(
                            mask.to(self.wrapper.device) if mask is not None else None
                        ),
                    )

            return logits
        
                
        def forward_last_state_logits(
            self,
            model_input:    Dict[str, Union[List[torch.Tensor], torch.Tensor, None]],
        ) -> torch.Tensor:

            mask = model_input["mask"]
            
            logits = self._forward_logits(
                dynamic_cat=model_input["dynamic_cat"],
                dynamic_cont=model_input["dynamic_cont"],
                static_cat=model_input["static_cat"],
                static_cont=model_input["static_cont"],
                mask=mask,
            )
    
            device = logits.device
    
            if mask is not None:
                lengths = mask.to(device).bool().sum(dim=1)
            else:
                lengths = torch.full(
                    (logits.size(0),),
                    logits.size(1),
                    dtype=torch.long,
                    device=device,
                )
    
            return logits[
                torch.arange(logits.size(0), device=device),
                lengths - 1,
            ]   # (B, C)

    
        def forward_sequence_logits(
            self,
            model_input:    Dict[str, Union[List[torch.Tensor], torch.Tensor, None]],
        ) -> torch.Tensor:
                
            """
            Return logits at selected prediction positions.
            """
        
            logits = self._forward_logits(
                dynamic_cat=model_input["dynamic_cat"],
                dynamic_cont=model_input["dynamic_cont"],
                static_cat=model_input["static_cat"],
                static_cont=model_input["static_cont"],
                mask=model_input["mask"],
            )  # (B, T, C)
        
            return logits
