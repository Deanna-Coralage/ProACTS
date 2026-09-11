import numpy as np
import pandas as pd

import torch

from dataclasses import dataclass

from typing import List, Dict, Tuple, Optional, Union, Any

from utils.feature_utils import compute_all_indices, compute_feature_configs

from config.feature_config import FeatureConfig

from model.preprocessor import PreprocessorArtifacts

from dice4el.scenario.scenario_dataset import ScenarioDataset



@dataclass(frozen=True)
class ScenarioHandler:

    """
    Generates synthetic + real-aligned scenario datasets for ScenarioLSTM.

    Responsibilities:
        - extract sequences from df
        - compute valid feature metadata
        - generate synthetic scenarios
        - return unified dataframe (real + fake)
    """

    feature_config:            FeatureConfig
    preprocessor_artifacts:    PreprocessorArtifacts


    def _generate_random_scenarios(
        self,
        lengths:                    List[int],
        n_features:                 int,
        activity_idx:               int,
        static_cont_idx:            np.ndarray,
        static_cat_idx:             np.ndarray,
        dynamic_cont_idx:           np.ndarray,
        dynamic_cat_idx:            np.ndarray,
        strategies_per_idx:         Dict[int, str],
        valid_ranges_per_idx:       Dict[int, Tuple[float, float]],
        valid_categories_per_idx:   Dict[int, List[str]],
        start_activities:           List[str],
        end_activities:             List[str],
        middle_activities:          List[str],
        n:                          int = 3,
    ) -> Dict[str, np.ndarray]:
        
        """
        Fully synthetic scenario generator with:
        - start/end constrained activity
        """
    
        T_max = max(lengths)
        num_scenarios = len(lengths) * n
    
        population = np.full(
            (num_scenarios, T_max, n_features),
            None,
            dtype=object
        )
    
        row = 0
    
        for base_L in lengths:
            for _ in range(n):
    
                L = base_L
                pos_arr = np.arange(L)
    
                # --- Activity sequence ---
                seq = np.empty(L, dtype=object)
                seq[0] = np.random.choice(start_activities)
    
                if L > 1:
                    seq[-1] = np.random.choice(end_activities)
                if L > 2:
                    seq[1:-1] = np.random.choice(middle_activities, size=L - 2)
    
                population[row, pos_arr, activity_idx] = seq
    
                # --- Static continuous ---
                for f in static_cont_idx:
                    lo, hi = valid_ranges_per_idx[f]
                    strategy = strategies_per_idx.get(f, "linear")
    
                    if strategy == "log":
                        val = np.exp(np.random.uniform(
                            np.log(max(lo, 1e-5)),
                            np.log(max(hi, 1e-5))
                        ))
                    else:
                        val = np.random.uniform(lo, hi)
    
                    population[row, :, f] = val
    
                # --- Static categorical ---
                for f in static_cat_idx:
                    cats = valid_categories_per_idx[f]
                    population[row, :, f] = np.random.choice(cats)
    
                # --- Dynamic continuous ---
                for f in dynamic_cont_idx:
                    lo, hi = valid_ranges_per_idx[f]
                    strategy = strategies_per_idx.get(f, "linear")
    
                    if strategy == "log":
                        noise = np.exp(np.random.uniform(
                            np.log(max(lo, 1e-5)),
                            np.log(max(hi, 1e-5)),
                            size=L
                        ))
                    else:
                        noise = np.random.uniform(lo, hi, size=L)
    
                    population[row, pos_arr, f] = noise
    
                # --- Dynamic categorical ---
                for f in dynamic_cat_idx:
                    if f == activity_idx:
                        continue
    
                    cats = valid_categories_per_idx[f]
                    population[row, pos_arr, f] = np.random.choice(cats, size=L)
    
                row += 1
    
        return {
            "population": population,
            "lengths": np.repeat(lengths, n)
        }
        

    def generate_scenario_df(
        self,
        df:                       pd.DataFrame,
        case_id_field:            str,
        sort_field:               str,
        n_scenarios_per_length:   int = 3,
    ) -> pd.DataFrame:
    
        feature_order = list(self.feature_config.feature_order)
    
        # --- Real data ---    
        if case_id_field not in df.columns:
            raise ValueError(
                f"Required case tracking column '{case_id_field}' "
                f"not found in input DataFrame columns: {list(df.columns)}"
            )

        real_rows = []
        for case_id, group in df.groupby(case_id_field):
        
            group = group.sort_values(sort_field)
            row = {case_id_field: case_id, "fake": False}
        
            for f in feature_order:
                row[f] = group[f].tolist()
        
            real_rows.append(row)
        
        real_df = pd.DataFrame(real_rows)
    
        # --- Extract sequences ---
        activity_sequences = (
            df.groupby(case_id_field)[self.feature_config.activity_feature]
              .apply(list)
        )
        lengths = [len(x) for x in activity_sequences]
        n_features = len(feature_order)
  
        # --- Fake data setup ---
        start_activities = set(seq[0] for seq in activity_sequences if len(seq) > 0)
        end_activities = set(seq[-1] for seq in activity_sequences if len(seq) > 0)
        middle_activities = {
            activity
            for seq in activity_sequences
            if len(seq) > 2
            for activity in seq[1:-1]
        }

        indices = compute_all_indices(self.feature_config)
        precomputed_feature_configs = compute_feature_configs(self.feature_config)
       
        activity_idx = indices["activity_idx"]
        static_cont_idx = indices["static_cont_idx"]
        static_cat_idx = indices["static_cat_idx"]
        dynamic_cont_idx = indices["dynamic_cont_idx"]
        dynamic_cat_idx = indices["dynamic_cat_idx"]
        
        valid_ranges_per_idx = precomputed_feature_configs["valid_ranges_per_idx"]
        valid_categories_per_idx = precomputed_feature_configs["valid_categories_per_idx"]
        strategies_per_idx = precomputed_feature_configs["feature_strategies_per_idx"]
    
        # --- Generate fake scenarios ---
        fake_data = self._generate_random_scenarios(
            lengths=lengths,
            n_features=n_features,
            activity_idx=activity_idx,
            static_cont_idx=static_cont_idx,
            static_cat_idx=static_cat_idx,
            dynamic_cont_idx=dynamic_cont_idx,
            dynamic_cat_idx=dynamic_cat_idx,
            strategies_per_idx=strategies_per_idx,
            valid_ranges_per_idx=valid_ranges_per_idx,
            valid_categories_per_idx=valid_categories_per_idx,
            start_activities=list(start_activities),
            end_activities=list(end_activities),
            middle_activities=list(middle_activities),
            n=n_scenarios_per_length
        )
    
        fake_population = fake_data["population"]
        fake_lengths = fake_data["lengths"]
    
        # --- Build fake df ---
        fake_rows = []
        case_id = 0
        
        for i in range(fake_population.shape[0]):
    
            L = int(fake_lengths[i])
    
            row = {case_id_field: f"f_{case_id}"}
            case_id += 1
    
            for f_idx, feat_name in enumerate(feature_order):
                row[feat_name] = fake_population[i, :L, f_idx].tolist()
    
            row["fake"] = True
            fake_rows.append(row)
    
        fake_df = pd.DataFrame(fake_rows)
    
        # --- Combine ---
        combined_df = pd.concat(
            [real_df, fake_df],
            ignore_index=True
        )
        combined_df = combined_df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        # --- Expand ---
        expanded_rows = []
        for _, row in combined_df.iterrows():
    
            seq_len = len(row[feature_order[0]])
    
            for t in range(seq_len):
    
                new_row = {
                    case_id_field: row[case_id_field],
                    "time_index": t,
                    "fake": row["fake"]
                }
    
                for f in feature_order:
                    new_row[f] = row[f][t]
    
                expanded_rows.append(new_row)
    
        return pd.DataFrame(expanded_rows)


    def transform_dataframe_to_scenariodataset(
        self,
        scenario_df:      pd.DataFrame,
        case_id_field:    str,
        sort_field:       Optional[str] = None,
    ):
    
        feature_order = list(self.feature_config.feature_order)
        feature_to_idx = self.feature_config.feature_to_idx
    
        cat_features = [f for f in feature_order if f in self.feature_config.categorical_features]
        cont_features = [f for f in feature_order if f in self.feature_config.continuous_features]
    
        cat_idx = [feature_to_idx[f] for f in cat_features]
        cont_idx = [feature_to_idx[f] for f in cont_features]
    
        # --- Encode + scale ---
        scenario_df = self.preprocessor_artifacts.transform_dataframe(scenario_df)
        
        cat_sequences = []
        cont_sequences = []
        labels_list = []
        masks = []
    
        for _, group in scenario_df.groupby(case_id_field):
    
            group = (
                group.sort_values(sort_field)
                if sort_field is not None else group.sort_index()
            )
    
            x = group[feature_order].to_numpy(dtype=np.float32)
    
            if len(x) <= 0:
                continue
    
            T = len(x)
    
            # --- Features ---
            cat_seq = [x[:, i].astype(np.int64) for i in cat_idx]
            cont_seq = x[:, cont_idx]
    
            label = float(group["fake"].iloc[0])
    
            cat_sequences.append(cat_seq)
            cont_sequences.append(cont_seq)
            labels_list.append(label)
            masks.append(np.ones(T, dtype=np.int32))
    
        if not cat_sequences:
            raise ValueError("No valid cases found")
    
        # --- Padding ---
        max_len = max(len(m) for m in masks)
        B = len(masks)
    
        # --- Arrs ---
        n_cat = len(cat_features)
        n_cont = len(cont_features)
    
        cont = np.zeros((B, max_len, n_cont), dtype=np.float32)
        cat = [np.zeros((B, max_len), dtype=np.int32) for _ in range(n_cat)]
        labels = np.zeros((B, max_len), dtype=np.float32)
        mask = np.zeros((B, max_len), dtype=np.int32)
        
        for i in range(B):
            L = len(masks[i])
    
            mask[i, :L] = 1
            cont[i, :L] = cont_sequences[i]
            labels[i, :L] = labels_list[i]
    
            for j in range(n_cat):
                cat[j][i, :L] = cat_sequences[i][j]

        
        return ScenarioDataset(
            cat=cat,
            cont=cont,
            labels=labels,
            mask=mask
        )


    def get_scenario_embedding_metadata(
        self,
        max_dim:   int = 50
    ) -> Dict[str, Any]:
    
        """
        Returns embedding metadata for ScenarioLSTM:
        Output:
            {
                "categorical_info": [(num_categories, emb_dim), ...],
                "n_continuous": int,
                "n_classes": 1
            }
        """
    
        feature_order = self.feature_config.feature_order
    
        cat_features = [f for f in feature_order if f in self.feature_config.categorical_features]
    
        categorical_info = []
    
        for f in cat_features:
            n_categories = len(self.preprocessor_artifacts.categorical_encoders[f].classes_)
            embedding_dim = min(max_dim, (n_categories + 1) // 2)
            categorical_info.append((n_categories, embedding_dim))
    
        n_continuous = len(self.feature_config.continuous_features)
    
        return {
            "categorical_info": categorical_info,
            "n_continuous": n_continuous,
            "n_classes": 1
        }


    def build_scenario_model_input(self, arr: np.ndarray) -> Dict[str, Union[List[np.ndarray], np.ndarray, None]]:

        """
        Builds scenario model input from prefix or prefix batch (no padding, already preprocessed).
        Handles (seq_len, n_features) and (n, seq_len, n_features).
        """

        arr = self.preprocessor_artifacts.encode(arr)
        arr = self.preprocessor_artifacts.scale(arr)
    
        feature_to_idx = self.feature_config.feature_to_idx

        cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.categorical_features]
        cont_features = [f for f in self.feature_config.feature_order if f in self.feature_config.continuous_features]
       
        # --- Single prefix (2D) ---
        if arr.ndim == 2:
            # Standardise dynamic features to have batch dimension: (1, seq_len) and (1, seq_len, n_feats)
            cat = [arr[:, feature_to_idx[f]][None, :] for f in cat_features]
            cont = arr[:, [feature_to_idx[f] for f in cont_features]][None, :, :]
    
        # --- Batch processing (3D) ---
        elif arr.ndim == 3:
            cat = [arr[:, :, feature_to_idx[f]] for f in cat_features]
            cont = arr[:, :, [feature_to_idx[f] for f in cont_features]]
            
        else:
            raise ValueError(f"Unsupported array dimension: {arr.ndim}")
    
        return {
            "cat": cat,
            "cont": cont,
            "mask": None
        }


    def convert_to_scenario_model_input_view(
        self,
        model_input:                           Dict[str, Union[List[torch.Tensor], torch.Tensor, None]],
        next_event_preprocessor_artifacts:     PreprocessorArtifacts,
    ) -> Dict[str, Union[List[torch.Tensor], torch.Tensor, None]]:
    
        dyn_cat = model_input["dynamic_cat"]
        dyn_cont = model_input["dynamic_cont"]
        static_cat = model_input["static_cat"]
        static_cont = model_input["static_cont"]
        mask = model_input.get("mask", None)
    
        process_cfg = next_event_preprocessor_artifacts.feature_config
        scenario_cfg = self.preprocessor_artifacts.feature_config
    
        B, T = dyn_cont.shape[:2]
        device = dyn_cont.device
        dtype = dyn_cont.dtype
    
        process_dyn_cat_features = [f for f in process_cfg.feature_order if f in process_cfg.dynamic_categorical_features]
        process_static_cat_features = [f for f in process_cfg.feature_order if f in process_cfg.static_categorical_features]
        process_dyn_cont_features = [f for f in process_cfg.feature_order if f in process_cfg.dynamic_continuous_features]
        process_static_cont_features = [f for f in process_cfg.feature_order if f in process_cfg.static_continuous_features]
    
        dyn_cat_map = dict(zip(process_dyn_cat_features, dyn_cat))
        static_cat_map = dict(zip(process_static_cat_features, static_cat))
    
        dyn_cont_map = {
            f: dyn_cont[:, :, j:j + 1] for j, f in enumerate(process_dyn_cont_features)
        }
    
        static_cont_map = {
            f: static_cont[:, j:j + 1] for j, f in enumerate(process_static_cont_features)
        }
    
        scenario_cat = []
        scenario_cont_parts = []
    
        for f in scenario_cfg.feature_order:
    
            if f in scenario_cfg.categorical_features:
    
                if f in dyn_cat_map:
                    scenario_cat.append(dyn_cat_map[f])
    
                elif f in static_cat_map:
                    x = static_cat_map[f].unsqueeze(1).expand(-1, T, -1)
                    scenario_cat.append(x)
    
                else:
                    raise ValueError(
                        f"Scenario categorical feature '{f}' not found."
                    )
    
            elif f in scenario_cfg.continuous_features:
    
                if f in dyn_cont_map:
                    scenario_cont_parts.append(dyn_cont_map[f])
    
                elif f in static_cont_map:
                    x = static_cont_map[f].unsqueeze(1).expand(-1, T, -1)
                    scenario_cont_parts.append(x)
    
                else:
                    raise ValueError(
                        f"Scenario continuous feature '{f}' not found."
                    )
    
        scenario_cont = (
            torch.cat(scenario_cont_parts, dim=-1)
            if scenario_cont_parts
            else torch.empty(B, T, 0, device=device, dtype=dtype)
        )
    
        return {
            "cat": scenario_cat,
            "cont": scenario_cont,
            "mask": mask,
        }
