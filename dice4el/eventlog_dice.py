import time

import math

import numpy as np

import torch
import torch.nn.functional as F

from typing import Dict, List, Set, Tuple, Union, Optional

from dataclasses import dataclass

from utils.general_utils import save_counterfactual_results
from utils.feature_utils import compute_vary_indices, compute_feature_configs
from utils.trace_utils import get_rollout_positions, add_simulation_mismatches_as_flexible

from model.model_wrapper import ModelWrapper

from dice4el.scenario.scenario_model_wrapper import ScenarioModelWrapper
from dice4el.dice4el_config import EventLogDiCEConfig

from dice4el.dice4el_utils import GradientMask
from dice4el.dice4el_utils import _compute_scaled_continuous_feature_configs, _get_split_scaled_continuous_configs, _compute_local_split_vary_indices
from dice4el.dice4el_utils import _scaled_categorical_distance, _scaled_continuous_distance

from ga_search.initialization import initialize_population
from ga_search.fitness import calculate_fitness, evaluate_cf_pop, _calculate_margin_loss

from process.constraint import ProcessConstraint



@dataclass(frozen=True)
class EventLogDiCE:

    dice4el_config:             EventLogDiCEConfig
    next_event_model_wrapper:   ModelWrapper
    scenario_model_wrapper:     ScenarioModelWrapper

    def __post_init__(self):
        object.__setattr__(
            self, "next_event_model_wrapper_differentiable", self.next_event_model_wrapper.differentiable
        )

        object.__setattr__(
            self, "scenario_model_wrapper_differentiable", self.scenario_model_wrapper.differentiable
        )

    
    @property
    def device(self):
        return next(
            self.next_event_model_wrapper.model.parameters()
        ).device


    # --- Helpers ---
    def _encode_desired_constraints(
        self,
        desired_constraints:  ProcessConstraint,
    ) -> Dict[Tuple[int, int], List[int]]:
    
        pre_art = self.next_event_model_wrapper.preprocessor_artifacts
        activity_feature = pre_art.feature_config.activity_feature
    
        return {
            (start, end): [
                pre_art.encode_feature(activity_feature, activity)
                for activity in allowed
            ]
            for (start, end), allowed in desired_constraints.segments.items()
        }

    def _to_numpy(self, x:  torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    def _get_categorical_feature_cardinality(self) -> Dict[str, Dict[str, int]]:
        return self.next_event_model_wrapper.preprocessor_artifacts.get_categorical_feature_cardinality()


    # --- Classfication loss / margin loss for multiple positions and classes ---
    def _multi_target_class_loss(
        self,
        sequence_logits:   torch.Tensor,           # (B, T, C)
        desired_targets:   Dict[int, Set[int]],
    ) -> torch.Tensor:
    
        """
        Multi-position / multi-class target loss.
    
        Parameters
        ----------
        sequence_logits : torch.Tensor
            Shape (B, T, C)
        desired_targets : Dict[int, Set[int]]
            Mapping: prediction_position -> set of acceptable classes
    
            Examples
            --------
            Single position, single class:
                {8: {3}}
    
            Single position, multiple classes (OR):
                {8: {3, 5}}
    
            Multiple positions:
                {
                    8: {3},
                    12: {7}
                }
    
            Multiple positions with multiple acceptable classes:
                {
                    8: {3,5},
                    12: {7,9}
                }
    
        Loss
        ----
        For each target position p:
            loss_p = -log( sum_{c ∈ allowed_classes} p(class=c | position=p) )
            
            - Classes inside a set are treated with OR logic.
            - Different positions are combined with AND logic.
    
        Special case
        ------------
        If only one class is allowed, e.g. {8:{3}}
        then loss = -log p(class=3) -> ordinary CrossEntropyLoss.
    
        Returns
        -------
        torch.Tensor
            Mean loss across all target positions.
        """
    
        losses = []
    
        for target_pos, allowed_classes in desired_targets.items():
    
            # logits at this prediction position
            logits_pos = sequence_logits[:, target_pos, :]      # (B,C)
    
            # convert logits to log-probabilities
            log_probs = F.log_softmax(logits_pos, dim=-1)
    
            # allowed classes at this position
            class_ids = torch.tensor(
                sorted(list(allowed_classes)),
                dtype=torch.long,
                device=sequence_logits.device
            )
    
            # OR condition over acceptable classes:
            loss_pos = -torch.logsumexp(
                log_probs[:, class_ids],
                dim=-1
            ).mean()
    
            losses.append(loss_pos)
    
        return torch.stack(losses).mean()

    
    def _multi_target_margin_loss(
        self,
        sequence_logits:   torch.Tensor,  # (B, T, C)
        desired_segments:  Dict[Tuple[int, int], List[int]],
        confidence_ratio:  float = 2.0,
        prediction_offset: int = 1,
    ) -> torch.Tensor:
        
        """
        Segment-aware margin loss.
    
        At each desired event position:
        1. Finds the active desired segment.
        2. Selects the remaining desired class with the highest logit.
        3. Computes the standard margin loss against the best non-desired class.
        4. Consumes one occurrence of the selected desired class.
        5. Continues to the next position with the remaining desired classes.
    
        The class selection is discrete, but the selected desired logit and
        competing logit remain differentiable.
        """
    
        B, logit_T, C = sequence_logits.shape
        margin = math.log(confidence_ratio)
    
        # Independent remaining constraint state for every CF candidate.
        batch_states = [
            {
                key: list(class_ids)
                for key, class_ids in desired_segments.items()
            }
            for _ in range(B)
        ]
    
        desired_positions = sorted({
            pos
            for start, end in desired_segments
            for pos in range(start, end + 1)
        })
    
        candidate_losses = sequence_logits.new_zeros(B)
        candidate_counts = sequence_logits.new_zeros(B)
    
        for target_pos in desired_positions:
    
            # For next-event models, output at t-1 predicts event at t.
            logit_pos = target_pos - prediction_offset
    
            if logit_pos < 0 or logit_pos >= logit_T:
                continue
    
            for b in range(B):
                state = batch_states[b]
    
                # Remove segments that have already expired.
                for key in list(state.keys()):
                    _, end = key
                    if target_pos > end:
                        del state[key]
    
                # Match the first active segment, consistent with
                # ProcessConstraint.check_constrained_position().
                matched_key = None
    
                for key in state.keys():
                    start, end = key
    
                    if start <= target_pos <= end:
                        matched_key = key
                        break
    
                if matched_key is None:
                    continue
    
                remaining_classes = state[matched_key]
    
                if not remaining_classes:
                    del state[matched_key]
                    continue
    
                # Duplicates are retained in state but only unique IDs are needed
                # when comparing logits at this position.
                desired_ids = sorted(set(remaining_classes))
    
                valid_desired_ids = [
                    class_id
                    for class_id in desired_ids
                    if 0 <= class_id < C
                ]
    
                candidate_counts[b] += 1.0
    
                if not valid_desired_ids:
                    candidate_losses[b] += 1.0
                    continue
    
                class_ids = torch.tensor(
                    valid_desired_ids,
                    dtype=torch.long,
                    device=sequence_logits.device,
                )
    
                logits_pos = sequence_logits[b, logit_pos, :]  # (C,)
    
                desired_values = logits_pos[class_ids]
                best_local_idx = torch.argmax(desired_values)
    
                best_desired_id = int(
                    class_ids[best_local_idx].detach().item()
                )
                desired_logit = desired_values[best_local_idx]
    
                other_mask = torch.ones(
                    C,
                    dtype=torch.bool,
                    device=sequence_logits.device,
                )
                other_mask[class_ids] = False
    
                if other_mask.any():
                    best_other_logit = logits_pos[other_mask].max()
    
                    violation = F.relu(
                        best_other_logit - desired_logit + margin
                    )
    
                    # position_loss = violation / (
                    #     violation + margin + 1e-8
                    # )
    
                    candidate_losses[b] += violation
    
                # Consume one occurrence of the desired class selected for
                # optimization at this position.
                remaining_classes.remove(best_desired_id)
    
                if not remaining_classes:
                    del state[matched_key]
    
        valid_candidates = candidate_counts > 0
    
        if not valid_candidates.any():
            return sequence_logits.new_tensor(0.0)
    
        per_candidate_loss = candidate_losses[valid_candidates] / (
            candidate_counts[valid_candidates] + 1e-8
        )
    
        return per_candidate_loss.mean()


    # --- Build and transform model inputs and views ---
    def _build_model_input(self, prefix: np.ndarray) -> Dict[str, Union[List[np.ndarray], np.ndarray, None]]:
        return self.next_event_model_wrapper.preprocessor_artifacts.build_model_input(prefix)

    def _convert_to_model_input_view(
        self,  
        dyn_cat_ohe:      List[torch.Tensor],   
        dyn_cont:         torch.Tensor,        
        static_cat_ohe:   List[torch.Tensor],       
        static_cont:      torch.Tensor,          
        mask:             Optional[torch.Tensor] = None,
    )  -> Dict[str, Union[List[torch.Tensor], torch.Tensor, None]]:

        model_input = {
            "dynamic_cat": dyn_cat_ohe,
            "dynamic_cont": dyn_cont,
            "static_cat": static_cat_ohe,
            "static_cont": static_cont,
            "mask": mask
        }
        return model_input
        
    def _convert_to_scenario_model_input_view(
        self,  
        dyn_cat_ohe:      List[torch.Tensor],   
        dyn_cont:         torch.Tensor,        
        static_cat_ohe:   List[torch.Tensor],       
        static_cont:      torch.Tensor,          
        mask:             Optional[torch.Tensor] = None,
    )  -> Dict[str, Union[List[torch.Tensor], torch.Tensor, None]]:

        model_input = {
            "dynamic_cat": dyn_cat_ohe,
            "dynamic_cont": dyn_cont,
            "static_cat": static_cat_ohe,
            "static_cont": static_cont,
            "mask": mask
        }
        return self.scenario_model_wrapper.scenario_handler.convert_to_scenario_model_input_view(
            model_input=model_input,
            next_event_preprocessor_artifacts=self.next_event_model_wrapper.preprocessor_artifacts
        )


    # --- Cat feature conversion to ohe and vice-versa ---
    def _init_cat_ohe(
        self,
        cat_arrs:     List[np.ndarray],
        cat_dims:     List[int],
    ) -> List[torch.Tensor]:

        out = []

        for arr, dim in zip(cat_arrs, cat_dims):
            x = torch.from_numpy(arr).long().to(self.device)
            p = F.one_hot(x, num_classes=dim).float()
            p = p.clone().detach()
            out.append(p)

        return out

    def _valid_cat_from_ohe(
        self,
        cat_ohe:    List[torch.Tensor],
    ) -> List[np.ndarray]:
    
        return [self._to_numpy(torch.argmax(p, dim=-1)) for p in cat_ohe]

    def _project_mutable_cat_ohe(
        self,
        cat_ohe:       List[torch.Tensor],
        vary_indices:  np.ndarray,
        use_sampling:  bool = False,
    ) -> List[torch.Tensor]:
    
        vary_set = set(vary_indices.tolist())
        projected = []
    
        for feature_idx, tensor in enumerate(cat_ohe):
    
            if feature_idx not in vary_set:
                # Preserve immutable feature exactly.
                projected.append(tensor.detach().clone())
                continue
    
            projected.append(
                self._project_cat_ohe(
                    cat_ohe=[tensor],
                    use_sampling=use_sampling,
                )[0]
            )
    
        return projected

    def _project_cat_ohe(
        self,
        cat_ohe:       List[torch.Tensor],
        use_sampling:  bool = False,
    ) -> List[torch.Tensor]:
        
        """
        Project relaxed categorical tensors into valid one-hot tensors.
        use_sampling:
            False: Select the category with the highest relaxed value.
    
            True: Normalize the relaxed values into probabilities and sample
                  one category from each categorical distribution.
        """
    
        projected_ohe = []
    
        for relaxed in cat_ohe:
            num_categories = relaxed.shape[-1]
    
            if use_sampling:
                nonnegative = relaxed.clamp_min(0.0)
    
                probability_sum = nonnegative.sum(
                    dim=-1,
                    keepdim=True,
                )
    
                uniform_probs = torch.full_like(
                    nonnegative,
                    fill_value=1.0 / num_categories,
                )
    
                probabilities = torch.where(
                    probability_sum > 1e-8,
                    nonnegative / probability_sum.clamp_min(1e-8),
                    uniform_probs,
                )
    
                flat_probabilities = probabilities.reshape(
                    -1,
                    num_categories,
                )
    
                selected_ids = torch.multinomial(
                    flat_probabilities,
                    num_samples=1,
                    replacement=True,
                ).squeeze(-1)
    
                selected_ids = selected_ids.reshape(
                    probabilities.shape[:-1]
                )
    
            else:
                selected_ids = torch.argmax(
                    relaxed,
                    dim=-1,
                )
    
            valid_one_hot = F.one_hot(
                selected_ids,
                num_classes=num_categories,
            ).to(
                device=relaxed.device,
                dtype=relaxed.dtype,
            )
    
            projected_ohe.append(valid_one_hot)
    
        return projected_ohe


    # --- Cont feature ---
    def _init_cont(
        self,
        cont:    np.ndarray,
    ) -> torch.Tensor:
    
        return torch.from_numpy(cont).float().to(self.device).clone().detach()


    # --- Immutable feature removal from optimization ---
    def _collect_cat_optimizer_vars(
        self,
        cat_ohe: List[torch.Tensor],
        split_vary_idx: np.ndarray,
        restricted_positions: Optional[Set[int]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[GradientMask], Dict[int, torch.Tensor]]:
        
        output: List[torch.Tensor] = []
        optimizer_vars: List[torch.Tensor] = []
        gradient_masks: List[GradientMask] = []
        output_masks: Dict[int, torch.Tensor] = {}
    
        vary_features = set(split_vary_idx.tolist())
    
        for feature_idx, tensor in enumerate(cat_ohe):
            original = tensor.detach().clone()
    
            if feature_idx not in vary_features:
                output.append(original)
                continue
    
            variable = original.clone().requires_grad_(True)
    
            if variable.ndim == 2:
                update_mask = torch.ones_like(variable)
    
            elif variable.ndim == 3:
                if restricted_positions is None:
                    update_mask = torch.ones_like(variable)
                else:
                    update_mask = torch.zeros_like(variable)
    
                    for position in restricted_positions:
                        if 0 <= position < variable.shape[1]:
                            update_mask[:, position, :] = 1.0
            else:
                raise ValueError(
                    "Unsupported categorical tensor shape: "
                    f"{tuple(variable.shape)}"
                )
    
            if not bool(update_mask.any().item()):
                output.append(original)
                continue
    
            output.append(variable)
            optimizer_vars.append(variable)
            gradient_masks.append(
                (variable, update_mask, original)
            )
            output_masks[feature_idx] = update_mask
    
        return (
            output,
            optimizer_vars,
            gradient_masks,
            output_masks,
        )

        
    def _collect_cont_optimizer_vars(
        self,
        cont:                  torch.Tensor,
        split_vary_idx:        np.ndarray,
        restricted_positions:  Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[GradientMask], Optional[torch.Tensor]]:
        
        optimizer_vars: List[torch.Tensor] = []
        gradient_masks: List[GradientMask] = []
    
        original = cont.detach().clone()
    
        if cont.shape[-1] == 0:
            return (
                original,
                optimizer_vars,
                gradient_masks,
                None,
            )
    
        vary_features = set(split_vary_idx.tolist())
        update_mask = torch.zeros_like(original)
    
        if original.ndim == 2:
            for feature_idx in vary_features:
                if 0 <= feature_idx < original.shape[-1]:
                    update_mask[:, feature_idx] = 1.0
    
        elif original.ndim == 3:
            for feature_idx in vary_features:
                if not 0 <= feature_idx < original.shape[-1]:
                    continue
    
                if restricted_positions is None:
                    update_mask[:, :, feature_idx] = 1.0
                else:
                    for position in restricted_positions:
                        if 0 <= position < original.shape[1]:
                            update_mask[:, position, feature_idx] = 1.0
        else:
            raise ValueError(
                "Unsupported continuous tensor shape: "
                f"{tuple(original.shape)}"
            )
    
        if not bool(update_mask.any().item()):
            return (
                original,
                optimizer_vars,
                gradient_masks,
                None,
            )
    
        variable = original.clone().requires_grad_(True)
    
        optimizer_vars.append(variable)
        gradient_masks.append(
            (variable, update_mask, original)
        )
    
        return (
            variable,
            optimizer_vars,
            gradient_masks,
            update_mask,
        )


    def _calculate_optimization_losses(
        self,
        dyn_cat_ohe:              List[torch.Tensor],
        dyn_cont:                 torch.Tensor,
        static_cat_ohe:           List[torch.Tensor],
        static_cont:              torch.Tensor,
        mask:                     Optional[torch.Tensor],
        desired_segments:         Dict[Tuple[int, int], List[int]],
        dyn_cat_0:                List[torch.Tensor],
        static_cat_0:             List[torch.Tensor],
        dyn_cont_0:               torch.Tensor,
        static_cont_0:            torch.Tensor,
        dynamic_cont_scaled_norm: torch.Tensor,
        static_cont_scaled_norm:  torch.Tensor,
        dyn_cat_update_masks:     Dict[int, torch.Tensor],
        static_cat_update_masks:  Dict[int, torch.Tensor],
        dyn_cont_update_mask:     Optional[torch.Tensor],
        static_cont_update_mask:  Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
    
        # --- Next-event margin ---
        model_input_view = self._convert_to_model_input_view(
            dyn_cat_ohe=dyn_cat_ohe,
            dyn_cont=dyn_cont,
            static_cat_ohe=static_cat_ohe,
            static_cont=static_cont,
            mask=mask,
        )
    
        sequence_logits = (
            self.next_event_model_wrapper_differentiable
            .forward_sequence_logits(
                model_input=model_input_view
            )
        )
    
        margin_loss = self._multi_target_margin_loss(
            sequence_logits=sequence_logits,
            desired_segments=desired_segments,
            confidence_ratio=self.dice4el_config.confidence_ratio,
        )

        margin_loss_flipped = self._multi_target_margin_loss(
            sequence_logits=sequence_logits,
            desired_segments=desired_segments,
            confidence_ratio=1.0 + 1e-6
        )
    
        # --- Scenario loss ---
        scenario_input_view = (
            self._convert_to_scenario_model_input_view(
                dyn_cat_ohe=dyn_cat_ohe,
                dyn_cont=dyn_cont,
                static_cat_ohe=static_cat_ohe,
                static_cont=static_cont,
                mask=mask,
            )
        )
    
        scenario_logits = (
            self.scenario_model_wrapper_differentiable
            .forward_sequence_logits(
                scenario_model_input=scenario_input_view
            )
        )
    
        if mask is not None:
            valid_mask = mask.to(scenario_logits.device).bool()
            valid_scenario_logits = scenario_logits[valid_mask]
        else:
            valid_scenario_logits = scenario_logits.reshape(-1)
    
        scenario_loss = F.binary_cross_entropy_with_logits(
            valid_scenario_logits,
            torch.zeros_like(valid_scenario_logits),
        )
    
        # --- Distance loss ---
        use_euclidean_distance = False
        if use_euclidean_distance:

            cat_distance_terms = [
                (p - p0).pow(2).mean()
                for p, p0 in zip(dyn_cat_ohe, dyn_cat_0)
            ]
            
            cat_distance_terms.extend(
                (p - p0).pow(2).mean()
                for p, p0 in zip(static_cat_ohe, static_cat_0)
            )
            
            cat_distance = (
                torch.stack(cat_distance_terms).mean()
                if cat_distance_terms
                else margin_loss.new_tensor(0.0)
            
            )
            
            cont_distance_terms = []
    
            if dyn_cont.numel() > 0:
                cont_distance_terms.append(
                    (dyn_cont - dyn_cont_0).pow(2).mean()
                )
        
            if static_cont.numel() > 0:
                cont_distance_terms.append(
                    (static_cont - static_cont_0).pow(2).mean()
                )
        
            cont_distance = (
                torch.stack(cont_distance_terms).mean()
                if cont_distance_terms
                else margin_loss.new_tensor(0.0)
            )
        
        else:
            cat_distance = _scaled_categorical_distance(
                dyn_cat_ohe=dyn_cat_ohe,
                dyn_cat_0=dyn_cat_0,
                static_cat_ohe=static_cat_ohe,
                static_cat_0=static_cat_0,
                mask=mask,
                dyn_cat_update_masks=dyn_cat_update_masks,
                static_cat_update_masks=static_cat_update_masks,
            )
            
            cont_distance_terms = []
    
            if dyn_cont.numel() > 0:
                cont_distance_terms.append(
                    _scaled_continuous_distance(
                        current=dyn_cont,
                        original=dyn_cont_0,
                        scaled_norm=dynamic_cont_scaled_norm,
                        update_mask=dyn_cont_update_mask,
                        mask=mask,
                    )
                )
                
            if static_cont.numel() > 0:
                cont_distance_terms.append(
                    _scaled_continuous_distance(
                        current=static_cont,
                        original=static_cont_0,
                        scaled_norm=static_cont_scaled_norm,
                        update_mask=static_cont_update_mask,
                        mask=None,
                    )
                )
            
            cont_distance = (
                torch.stack(cont_distance_terms).mean()
                if cont_distance_terms
                else margin_loss.new_tensor(0.0)
            )
    
        distance_loss = (
            cat_distance + cont_distance
        ) / 2.0
    
        # --- Categorical simplex loss ---
        simplex_terms = [
            (p.sum(dim=-1) - 1.0).pow(2).mean()
            for p in dyn_cat_ohe
        ]
    
        simplex_terms.extend(
            (p.sum(dim=-1) - 1.0).pow(2).mean()
            for p in static_cat_ohe
        )
    
        cat_loss = (
            torch.stack(simplex_terms).mean()
            if simplex_terms
            else margin_loss.new_tensor(0.0)
        )

        # --- Diversity and sparsity ---
        # diversity_loss = self._diversity_loss(
        #     dyn_cat_ohe=dyn_cat_ohe,
        #     dyn_cont=dyn_cont,
        #     static_cat_ohe=static_cat_ohe,
        #     static_cont=static_cont,
        # )
        # sparsity_loss = self._sparsity_loss(
        #     dyn_cat_ohe, dyn_cat_0,
        #     static_cat_ohe, static_cat_0,
        #     dyn_cont, dyn_cont_0,
        #     static_cont, static_cont_0,
        # )

        # --- Total loss ---
        total_loss = (
            self.dice4el_config.w_margin_loss * margin_loss +
            self.dice4el_config.w_scenario_loss * scenario_loss +
            self.dice4el_config.w_distance_loss * distance_loss +
            self.dice4el_config.w_cat_loss * cat_loss
            # self.dice4el_config.w_diversity_loss * diversity_loss +
            # self.dice4el_config.w_sparsity_loss * sparsity_loss
        )
        
        return {
            "margin_loss": margin_loss,
            "margin_loss_flipped": margin_loss_flipped,
            "scenario_loss": scenario_loss,
            "distance_loss": distance_loss,
            "cat_loss": cat_loss,
            "total_loss": total_loss,
        }


    # --- Diversity and sparsity ---
    def _diversity_loss(
        self,
        dyn_cat_ohe:      List[torch.Tensor],
        dyn_cont:         torch.Tensor,
        static_cat_ohe:   List[torch.Tensor],
        static_cont:      torch.Tensor,
    ) -> torch.Tensor:
    
        parts = []
    
        for p in dyn_cat_ohe:
            parts.append(p.reshape(p.size(0), -1))
    
        for p in static_cat_ohe:
            parts.append(p.reshape(p.size(0), -1))
    
        if dyn_cont.shape[-1] > 0:
            parts.append(dyn_cont.reshape(dyn_cont.size(0), -1))
    
        if static_cont.shape[-1] > 0:
            parts.append(static_cont.reshape(static_cont.size(0), -1))
    
        x = torch.cat(parts, dim=1)  # (N, D)
    
        if x.size(0) <= 1:
            return torch.tensor(0.0, device=x.device)
    
        dist = torch.cdist(x, x, p=2)
    
        eye = torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        pairwise_dist = dist[~eye]
    
        # Minimizing this encourages pairwise distances to become larger.
        return torch.exp(-pairwise_dist).mean()

    def _sparsity_loss(
        self,
        dyn_cat_ohe:      List[torch.Tensor], 
        dyn_cat_0:        List[torch.Tensor],
        static_cat_ohe:   List[torch.Tensor], 
        static_cat_0:     List[torch.Tensor],
        dyn_cont:         torch.Tensor,
        dyn_cont_0:       torch.Tensor,
        static_cont:      torch.Tensor, 
        static_cont_0:    torch.Tensor,
        eps:              float = 1e-3,
    ) -> torch.Tensor:
    
        loss = torch.tensor(0.0, device=self.device)
    
        for p, p0 in zip(dyn_cat_ohe, dyn_cat_0):
            diff = (p - p0).abs().sum(dim=-1)
            loss = loss + torch.sigmoid((diff - eps) * 20).sum()
    
        for p, p0 in zip(static_cat_ohe, static_cat_0):
            diff = (p - p0).abs().sum(dim=-1)
            loss = loss + torch.sigmoid((diff - eps) * 20).sum()
    
        if dyn_cont.shape[-1] > 0:
            diff = (dyn_cont - dyn_cont_0).abs()
            loss = loss + torch.sigmoid((diff - eps) * 20).sum()
    
        if static_cont.shape[-1] > 0:
            diff = (static_cont - static_cont_0).abs()
            loss = loss + torch.sigmoid((diff - eps) * 20).sum()
    
        return loss

    
    # --- Build original trace passed optimized prefix ---
    def _replace_prefix_in_trace(
        self,
        trace:                   np.ndarray,
        optimized_prefix_pop:    np.ndarray,
    ) -> np.ndarray:
    
        if trace.ndim == 2:
            trace = trace[None, ...]
    
        if optimized_prefix_pop.ndim == 2:
            optimized_prefix_pop = optimized_prefix_pop[None, ...]
    
        N, Tp, F = optimized_prefix_pop.shape
        B, T, F0 = trace.shape
    
        if F != F0:
            raise ValueError(f"Feature mismatch: {F=} vs {F0=}")
    
        if B == 1:
            full_pop = np.repeat(trace, N, axis=0)
        elif B == N:
            full_pop = trace.copy()
        else:
            raise ValueError(
                f"Cannot align trace batch B={B} with CF population N={N}"
            )
    
        full_pop[:, :Tp, :] = optimized_prefix_pop
    
        return full_pop


    # --- Main optimization loop ---
    def _run(
        self,
        trace:                     np.ndarray,
        cf_pop:                    np.ndarray,
        positions:                 List[int],
        desired_constraints:       ProcessConstraint,
        verbose_freq:              int = 20,
    ) -> np.ndarray:

        """DiCE4EL Optimization"""

        if cf_pop.ndim == 2:
            cf_pop = cf_pop[None, ...]

        restricted_positions = (
            set(positions)
            if positions is not None
            else None
        )
    
        if self.dice4el_config.mode not in {"last", "multi"}:
            raise ValueError("self.dice4el_config.mode must be either 'last' or 'multi'.")
    
        desired_segments = self._encode_desired_constraints(desired_constraints)
        if not desired_segments:
            raise ValueError("No valid desired segments were encoded.")

        feature_config = self.next_event_model_wrapper.preprocessor_artifacts.feature_config
        
        dynamic_cat_features = [
            feature for feature in feature_config.feature_order if feature in feature_config.dynamic_categorical_features
        ]
        
        static_cat_features = [
            feature for feature in feature_config.feature_order if feature in feature_config.static_categorical_features
        ]
        
        split_vary_indices = _compute_local_split_vary_indices(
            feature_config=feature_config,
        )
        
        activity_local_idx = dynamic_cat_features.index(
            feature_config.activity_feature
        )
        
        if activity_local_idx in split_vary_indices["split_vary_dynamic_cat_idx"]:
            raise ValueError(
                "Activity feature must not be optimized."
            )
        
        cardinality_info = self._get_categorical_feature_cardinality()

        dynamic_cat_dims = [
            cardinality_info["dynamic_categorical_info"][feature] for feature in dynamic_cat_features
        ]
        
        static_cat_dims = [
            cardinality_info["static_categorical_info"][feature] for feature in static_cat_features
        ]
    
        if self.dice4el_config.mode == "last":
            last_key = max(
                desired_segments,
                key=lambda key: (key[1], key[0]),
            )
            desired_segments = {
                last_key: desired_segments[last_key]
            }

        max_target_pos = max(end for _, end in desired_segments)
    
        # Need prefix up to event max_target_pos.
        # Logit max_target_pos-1 predicts event max_target_pos.
        trace_prefix = trace[:max_target_pos + 1][None, ...]
        trace_prefix = np.repeat(trace_prefix, cf_pop.shape[0], axis=0)
        cf_prefix = cf_pop[:, :max_target_pos + 1]

        original_inputs = self._build_model_input(trace_prefix)
        model_inputs = self._build_model_input(cf_prefix)
    
    
        # --- Initialize categorical CF variables - relaxed ohe ---
        dyn_cat_ohe = self._init_cat_ohe(
            model_inputs["dynamic_cat"], dynamic_cat_dims
        )
        static_cat_ohe = self._init_cat_ohe(
            model_inputs["static_cat"], static_cat_dims
        )

        # --- Originals for distance ---
        dyn_cat_0 = self._init_cat_ohe(
            original_inputs["dynamic_cat"], dynamic_cat_dims
        )
        static_cat_0 = self._init_cat_ohe(
            original_inputs["static_cat"], static_cat_dims
        )

        (dyn_cat_ohe, dyn_cat_optimizer_vars, dyn_cat_grad_masks, dyn_cat_update_masks) = self._collect_cat_optimizer_vars(
            cat_ohe=dyn_cat_ohe,
            split_vary_idx=split_vary_indices["split_vary_dynamic_cat_idx"],
            restricted_positions=restricted_positions,
        )
        
        (static_cat_ohe, static_cat_optimizer_vars, static_cat_grad_masks, static_cat_update_masks) = self._collect_cat_optimizer_vars(
            cat_ohe=static_cat_ohe,
            split_vary_idx=split_vary_indices["split_vary_static_cat_idx"],
            restricted_positions=None,
        )
        
        # --- Initialize continuous CF variables ---
        dyn_cont = self._init_cont(model_inputs["dynamic_cont"])
        static_cont = self._init_cont(model_inputs["static_cont"])

        # Distance measures
        scaled_configs = _compute_scaled_continuous_feature_configs(
            feature_config=feature_config,
            scaler=self.next_event_model_wrapper.preprocessor_artifacts.scaler,
        )
        
        split_scaled_configs = _get_split_scaled_continuous_configs(
            feature_config=feature_config,
            scaled_lower_arr=scaled_configs["scaled_lower_arr"],
            scaled_upper_arr=scaled_configs["scaled_upper_arr"],
            scaled_norm_arr=scaled_configs["scaled_norm_arr"],
        )
        
        dynamic_cont_scaled_norm = torch.as_tensor(
            split_scaled_configs["dynamic_cont_scaled_norm"],
            dtype=dyn_cont.dtype,
            device=dyn_cont.device,
        )
        
        static_cont_scaled_norm = torch.as_tensor(
            split_scaled_configs["static_cont_scaled_norm"],
            dtype=static_cont.dtype,
            device=static_cont.device,
        )

        dynamic_cont_lower = torch.as_tensor(
            split_scaled_configs["dynamic_cont_lower"],
            dtype=dyn_cont.dtype,
            device=dyn_cont.device,
        )
            
        dynamic_cont_upper = torch.as_tensor(
            split_scaled_configs["dynamic_cont_upper"],
            dtype=dyn_cont.dtype,
            device=dyn_cont.device,
        )

        static_cont_lower = torch.as_tensor(
            split_scaled_configs["static_cont_lower"],
            dtype=static_cont.dtype,
            device=static_cont.device,
        )
            
        static_cont_upper = torch.as_tensor(
            split_scaled_configs["static_cont_upper"],
            dtype=static_cont.dtype,
            device=static_cont.device,
        )

        # --- Originals for distance ---
        dyn_cont_0 = self._init_cont(original_inputs["dynamic_cont"])
        static_cont_0 = self._init_cont(original_inputs["static_cont"])

        (dyn_cont, dyn_cont_optimizer_vars, dyn_cont_grad_masks, dyn_cont_update_mask) = self._collect_cont_optimizer_vars(
            cont=dyn_cont,
            split_vary_idx=split_vary_indices["split_vary_dynamic_cont_idx"],
            restricted_positions=restricted_positions,
        )
        
        (static_cont, static_cont_optimizer_vars, static_cont_grad_masks, static_cont_update_mask) = self._collect_cont_optimizer_vars(
            cont=static_cont,
            split_vary_idx=split_vary_indices["split_vary_static_cont_idx"],
            restricted_positions=None,
        )
        
        mask = model_inputs.get("mask", None)
        if mask is not None:
            mask = torch.from_numpy(mask).float().to(self.device)
    
        gradient_masks: List[GradientMask] = (
            dyn_cat_grad_masks + static_cat_grad_masks + dyn_cont_grad_masks + static_cont_grad_masks
        )
        
        optimizer_vars = (
            dyn_cat_optimizer_vars + static_cat_optimizer_vars + dyn_cont_optimizer_vars + static_cont_optimizer_vars
        )
        
        if not optimizer_vars:
            raise ValueError("No features are marked vary=True.")

        best_projected_flip_loss = float("inf")
        best_projected_total_loss = float("inf")
        best_projected_step = -1
        best_projected_state = None
        
        tolerance = 1e-5

        steps_without_improvement = 0
        stopped_early = False
        
        optimizer = torch.optim.Adam(optimizer_vars, lr=self.dice4el_config.learning_rate)
            
        for step in range(self.dice4el_config.optimization_steps):
    
            optimizer.zero_grad()

            soft_losses = self._calculate_optimization_losses(
                dyn_cat_ohe=dyn_cat_ohe,
                dyn_cont=dyn_cont,
                static_cat_ohe=static_cat_ohe,
                static_cont=static_cont,
                mask=mask,
                desired_segments=desired_segments,
                dyn_cat_0=dyn_cat_0,
                static_cat_0=static_cat_0,
                dyn_cont_0=dyn_cont_0,
                static_cont_0=static_cont_0,
                dynamic_cont_scaled_norm=dynamic_cont_scaled_norm,
                static_cont_scaled_norm=static_cont_scaled_norm,
                dyn_cat_update_masks=dyn_cat_update_masks,
                static_cat_update_masks=static_cat_update_masks,
                dyn_cont_update_mask=dyn_cont_update_mask,
                static_cont_update_mask=static_cont_update_mask,
            )

            # --- Backprop ---
            soft_losses["total_loss"].backward()
            # Remove forbidden gradients
            with torch.no_grad():
                for variable, update_mask, _ in gradient_masks:
                    if variable.grad is not None:
                        variable.grad.mul_(update_mask)
            
            optimizer.step()

            # Restore immutable entries
            with torch.no_grad():
                for variable, update_mask, original in gradient_masks:
                    variable.copy_(
                        torch.where(
                            update_mask.bool(),
                            variable,
                            original,
                        )
                    )


            # --- Hard clipping (projected gradient descent) ---
            with torch.no_grad():
                if dyn_cont.numel() > 0:
                    lower = dynamic_cont_lower.view(1, 1, -1)
                    upper = dynamic_cont_upper.view(1, 1, -1)
                    dyn_cont.copy_(torch.maximum(torch.minimum(dyn_cont, upper), lower))
            
                if static_cont.numel() > 0:
                    lower = static_cont_lower.view(1, -1)
                    upper = static_cont_upper.view(1, -1)
                    static_cont.copy_(
                        torch.maximum(torch.minimum(static_cont, upper), lower)
                    )
                    
                for feature_idx in split_vary_indices["split_vary_dynamic_cat_idx"]:
                    dyn_cat_ohe[feature_idx].clamp_(0.0, 1.0)
            
                for feature_idx in split_vary_indices["split_vary_static_cat_idx"]:
                    static_cat_ohe[feature_idx].clamp_(0.0, 1.0)
            
                # Preserve immutable values after clipping too.
                for variable, update_mask, original in gradient_masks:
                    variable.copy_(
                        torch.where(
                            update_mask.bool(),
                            variable,
                            original,
                        )
                    )
                    
                                                                     
            # --- Project updated relaxed categories into valid one-hot values ---
            with torch.no_grad():
                projected_dyn_cat_ohe = self._project_cat_ohe(
                    cat_ohe=dyn_cat_ohe,
                    use_sampling=False
                )
                projected_static_cat_ohe = self._project_cat_ohe(
                    cat_ohe=static_cat_ohe,
                    use_sampling=False
                )

                projected_losses = self._calculate_optimization_losses(
                    dyn_cat_ohe=projected_dyn_cat_ohe,
                    dyn_cont=dyn_cont,
                    static_cat_ohe=projected_static_cat_ohe,
                    static_cont=static_cont,
                    mask=mask,
                    desired_segments=desired_segments,
                    dyn_cat_0=dyn_cat_0,
                    static_cat_0=static_cat_0,
                    dyn_cont_0=dyn_cont_0,
                    static_cont_0=static_cont_0,
                    dynamic_cont_scaled_norm=dynamic_cont_scaled_norm,
                    static_cont_scaled_norm=static_cont_scaled_norm,
                    dyn_cat_update_masks=dyn_cat_update_masks,
                    static_cat_update_masks=static_cat_update_masks,
                    dyn_cont_update_mask=dyn_cont_update_mask,
                    static_cont_update_mask=static_cont_update_mask,
                )

            
           # --- Retain best projected candidate ---
            projected_margin_loss = projected_losses["margin_loss"].item()
            projected_flip_loss = projected_losses["margin_loss_flipped"].item()
            projected_total_loss = projected_losses["total_loss"].item()
            
            candidate_flipped = (projected_flip_loss <= tolerance)
            best_flipped = (best_projected_flip_loss <= tolerance)
            
            should_update = False
            
            if candidate_flipped:
                # A valid candidate always beats an invalid best.
                if not best_flipped:
                    should_update = True
            
                # Both are valid: retain the lower-total-loss candidate.
                elif projected_total_loss < best_projected_total_loss - tolerance:
                    should_update = True
            
            else:
                # Only compare invalid candidates when the best is also invalid.
                if not best_flipped:
                    if projected_flip_loss < best_projected_flip_loss - tolerance:
                        should_update = True
            
                    # Optional tie-breaker.
                    elif (
                        abs(
                            projected_flip_loss
                            - best_projected_flip_loss
                        ) <= tolerance
                        and projected_total_loss < best_projected_total_loss - tolerance
                    ):
                        should_update = True
            
            if should_update:
                best_projected_flip_loss = projected_flip_loss
                best_projected_total_loss = projected_total_loss
                best_projected_step = step + 1
            
                best_projected_state = {
                    "dynamic_cat": [
                        p.detach().clone()
                        for p in projected_dyn_cat_ohe
                    ],
                    "static_cat": [
                        p.detach().clone()
                        for p in projected_static_cat_ohe
                    ],
                    "dynamic_cont": dyn_cont.detach().clone(),
                    "static_cont": static_cont.detach().clone(),
                }
            
                steps_without_improvement = 0
            
            else:
                current_best_flipped = (
                    best_projected_flip_loss <= tolerance
                )
            
                if current_best_flipped:
                    steps_without_improvement += 1
            
            current_best_flipped = (
                best_projected_flip_loss <= tolerance
            )
            
            if (
                current_best_flipped
                and step + 1
                    >= self.dice4el_config.early_stopping_min_steps
                and steps_without_improvement
                    >= self.dice4el_config.early_stopping_patience
            ):
                stopped_early = True
            
                print(
                    f"Early stopping at step {step + 1}. "
                    f"Best projected candidate occurred at "
                    f"step {best_projected_step}."
                )
                break
            

            # --- Valid-CF replacement ---
            if self.dice4el_config.use_valid_cf_only:
                with torch.no_grad():
                    replacement_dyn_cat_ohe = self._project_mutable_cat_ohe(
                        cat_ohe=dyn_cat_ohe,
                        vary_indices=split_vary_indices[
                            "split_vary_dynamic_cat_idx"
                        ],
                        use_sampling=self.dice4el_config.use_sampling,
                    )
            
                    replacement_static_cat_ohe = self._project_mutable_cat_ohe(
                        cat_ohe=static_cat_ohe,
                        vary_indices=split_vary_indices[
                            "split_vary_static_cat_idx"
                        ],
                        use_sampling=self.dice4el_config.use_sampling,
                    )
            
                    for relaxed, replacement in zip(
                        dyn_cat_ohe,
                        replacement_dyn_cat_ohe,
                    ):
                        relaxed.copy_(replacement)
            
                    for relaxed, replacement in zip(
                        static_cat_ohe,
                        replacement_static_cat_ohe,
                    ):
                        relaxed.copy_(replacement)
            
                    # Restore forbidden positions exactly.
                    for variable, update_mask, original in gradient_masks:
                        variable.copy_(
                            torch.where(
                                update_mask.bool(),
                                variable,
                                original,
                            )
                        )
            
                # The variables were changed discontinuously, so clear Adam momentum.
                for variable in optimizer_vars:
                    state = optimizer.state.get(variable)
            
                    if not state:
                        continue
            
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
            
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()
            
                    if "max_exp_avg_sq" in state:
                        state["max_exp_avg_sq"].zero_()
                        
            
            # --- Printing ---
            if (step + 1) % verbose_freq == 0:
                print(
                    f"[{step + 1}] "
                    f"soft_total={soft_losses['total_loss'].item():.4f} "
                    f"soft_margin={soft_losses['margin_loss'].item():.4f} "
                    f"soft_scenario={soft_losses['scenario_loss'].item():.4f} "
                    f"soft_distance={soft_losses['distance_loss'].item():.4f} "
                    f"soft_cat={soft_losses['cat_loss'].item():.4f} | "
                    f"projected_total={projected_total_loss:.4f} "
                    f"projected_margin={projected_margin_loss:.4f} "
                    f"projected_margin_flipped={projected_flip_loss:.4f} "
                    f"projected_scenario="
                    f"{projected_losses['scenario_loss'].item():.4f} "
                    f"projected_distance="
                    f"{projected_losses['distance_loss'].item():.4f} "
                    f"projected_cat="
                    f"{projected_losses['cat_loss'].item():.4f}"
                )
                
        # --- Return best cf ---
        best_flipped = (best_projected_flip_loss <= tolerance)
    
        if not best_flipped:
            print(
                f"===== "
                f"No projected candidate satisfied all desired flips. "
                f"Returning the closest projected candidate from "
                f"step={best_projected_step}, with "
                f"flip loss={best_projected_flip_loss:.6f} "
                f"and total loss={best_projected_total_loss:.6f}."
            )

        final_dyn_cat = self._valid_cat_from_ohe(best_projected_state["dynamic_cat"])
        final_static_cat = self._valid_cat_from_ohe(best_projected_state["static_cat"])
        final_dyn_cont = best_projected_state["dynamic_cont"]
        final_static_cont = best_projected_state["static_cont"]

        optimized_cf_pop = self.next_event_model_wrapper.preprocessor_artifacts.reconstruct_trace_from_model_input(
            model_input={
                "dynamic_cat": final_dyn_cat,
                "dynamic_cont": self._to_numpy(final_dyn_cont),
                "static_cat": final_static_cat,
                "static_cont": self._to_numpy(final_static_cont),
                "mask": self._to_numpy(mask) if mask is not None else None,
            }
        )

        output_cf_pop = self._replace_prefix_in_trace(
            trace=cf_pop,
            optimized_prefix_pop=optimized_cf_pop,
        )
        
        return output_cf_pop


    # --- Counterfactual search ---
    def search(
        self,
        trace:                       np.ndarray,
        train_arr:                   np.ndarray,
        conformant_arr:              np.ndarray,
        change_allowed_positions:    List[int],
        desired_constraints:         ProcessConstraint,
        flexible_constraints:        ProcessConstraint,
        output_path:                 Optional[str] = None,
    ) -> Dict[str, Union[Dict[str, float], np.ndarray]]:

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
            

        start_time = time.perf_counter()
        
        # --- Precompute indices for reuse ---
        feature_config = self.next_event_model_wrapper.preprocessor_artifacts.feature_config
        vary_indices = compute_vary_indices(
            feature_config=feature_config
        )
        precomputed_feature_configs = compute_feature_configs(
            feature_config=feature_config
        )
       
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
        simulated_trace = self.next_event_model_wrapper.simulate_forward_nonautoregressive(
            traces=trace,
            desired_positions=list(range(1, len(trace))),
        )["simulated_traces"][0]

        # --- Add mismatches as flexible positions ---
        flexible_constraints_with_simulation_mismatches = add_simulation_mismatches_as_flexible(
            original_trace=trace,
            simulated_trace=simulated_trace,
            activity_idx=feature_config.feature_to_idx[feature_config.activity_feature],
            flexible_constraints=flexible_constraints,
        )

        # --- Initialize ---
        initial_cf_pop = initialize_population(
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
            population_size=self.dice4el_config.output_population_size,
            cbi_ratio=self.dice4el_config.cbi_ratio,
            sbi_ratio=self.dice4el_config.sbi_ratio,
            sparsity_pop_ratio=self.dice4el_config.sparsity_pop_ratio,
            sparsity_feature_p=self.dice4el_config.sparsity_feature_p
        )

        # --- Run DiCE ---
        output_cf_pop = self._run(
            trace=simulated_trace,
            cf_pop=initial_cf_pop,
            positions=change_allowed_positions,
            desired_constraints=desired_constraints,
        )

        end_time = time.perf_counter()
        print(f"Time taken for DiCE4EL optimization loop: {end_time - start_time:.6f} seconds")

        # --- Evaluate and return final cf population ---
        start = time.perf_counter()

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
            confidence_ratio=self.dice4el_config.confidence_ratio,
            penalty_desired_nonflip = self.dice4el_config.penalty_desired_nonflip,
            penalty_desired_wrongflip = self.dice4el_config.penalty_desired_wrongflip,
            penalty_undesired_flip = self.dice4el_config.penalty_undesired_flip,
            penalty_flexible_wrongflip = self.dice4el_config.penalty_flexible_wrongflip,
            w_distance = self.dice4el_config.w_distance,
            w_sparsity = self.dice4el_config.w_sparsity,
            w_margin = self.dice4el_config.w_margin,
            w_process_violation = self.dice4el_config.w_process_violation,
        )
        
        output_rollout_positions = get_rollout_positions(trace=simulated_trace, cf_pop=output_cf_pop)
        output_forward_simul = self.next_event_model_wrapper.simulate_forward_autoregressive(
            traces=output_cf_pop,
            rollout_positions=output_rollout_positions,
            desired_positions=desired_positions,
            flexible_constraints = flexible_constraints,
        )

        output_forward_simul_na = self.next_event_model_wrapper.simulate_forward_nonautoregressive(
            traces=output_cf_pop,
            desired_positions=desired_positions,
        )

        output_cf_pop_fitness_components = calculate_fitness(
            cf_pop=output_cf_pop,
            simulated_cfs=output_forward_simul["simulated_traces"],
            logits_store=output_forward_simul["logits_store"],
            logits_store_na=output_forward_simul_na["logits_store"],
            **fitness_kwargs,
        )
        
        output_cf_pop_eval = evaluate_cf_pop(
            cf_pop=output_cf_pop,
            simulated_cfs=output_forward_simul["simulated_traces"],
            train_arr=train_arr,
            logits_store=output_forward_simul["logits_store"],
            logits_store_na=output_forward_simul_na["logits_store"],
            sample_ratio=self.dice4el_config.implaus_sample_ratio,
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
                feature_order=feature_config.feature_order
            )

        return {
            "eval": output_cf_pop_eval,
            "cf_pop": output_cf_pop,
            "fitness_components": output_cf_pop_fitness_components,
        }
