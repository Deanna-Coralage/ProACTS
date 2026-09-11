import os
import joblib

import numpy as np
import pandas as pd

import torch

from dataclasses import dataclass

from typing import Dict, List, Literal, Any, Union, Optional

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

from config.feature_config import FeatureConfig

from dataset.process_dataset import ProcessDataset



@dataclass(frozen=True)
class PreprocessorArtifacts:

    """ Holds all preprocessing artifacts. 
    Frozen — immutable after creation, safe to share across pipeline:
        - feature encoding/decoding 
        - scaling/unscaling 
        - activity prototypes in raw space for simulation """

    feature_config:         FeatureConfig
    categorical_encoders:   Dict[str, LabelEncoder]
    scaler:                 Union[StandardScaler, RobustScaler]
    activity_prototypes:    Dict[str, np.ndarray]

   
    # --- Save ---
    def save(self, path: str = "./pretrained_models/") -> None:

        """ Save preprocessing artifacts to disk. """

        os.makedirs(path, exist_ok=True)

        joblib.dump(self.feature_config, f"{path}/feature_config.pkl")
        joblib.dump(self.categorical_encoders, f"{path}/encoders.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.activity_prototypes, f"{path}/activity_prototypes.pkl")


    # --- Build ---
    @classmethod
    def build(
        cls,
        df:                    pd.DataFrame,
        feature_config:        FeatureConfig,      
        scaler_type:           Literal["standard", "robust"] = "robust"
    ) -> "PreprocessorArtifacts":

        if scaler_type not in ("standard", "robust"):
            raise ValueError(
                f"Invalid scaler_type: {scaler_type}"
            )
            
        activity_feature = feature_config.activity_feature
        categorical_features = feature_config.categorical_features
        scaler_features = feature_config.continuous_features
        feature_order = feature_config.feature_order
        
        # --- Categorical encoders ---
        encoders = {}
        for feature in categorical_features:
            if feature not in df.columns:
                raise ValueError(f"Missing categorical feature: {feature}")

            le = LabelEncoder()
            le.fit(df[feature].astype(str))
            encoders[feature] = le

        # --- Scaler ---
        scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        scaler.fit(df[scaler_features].values.astype(float))

        # --- Activity prototypes ---       
        activity_prototypes = {}
        
        for act, group in df.groupby(activity_feature):
            row = []
            for feature in feature_order:
                if feature in scaler_features:
                    vals = pd.to_numeric(group[feature], errors="coerce").dropna()
                    if vals.empty:
                        val = np.nan
                    else:
                        val = float(vals.median())
                else:
                    val = group[feature].mode().iloc[0]
                row.append(val)
            
            activity_prototypes[str(act)] = np.array(row, dtype=object)

        return cls(
            feature_config=feature_config,
            categorical_encoders=encoders,
            scaler=scaler,
            activity_prototypes=activity_prototypes
        )

    
    # --- Load ---
    @classmethod
    def load(cls, path: str = "./pretrained_models/") -> "PreprocessorArtifacts":

        """ Load preprocessing artifacts from disk. """

        feature_config = joblib.load(f"{path}/feature_config.pkl")
        encoders = joblib.load(f"{path}/encoders.pkl")
        scaler = joblib.load(f"{path}/scaler.pkl")
        activity_prototypes = joblib.load(f"{path}/activity_prototypes.pkl")

        return cls(
            feature_config=feature_config,
            categorical_encoders=encoders,
            scaler=scaler,
            activity_prototypes=activity_prototypes,
        )


    # --- Helpers ---
    def encode_feature(self, feature: str, original_value: Union[str, List[str], np.ndarray]) -> Union[int, np.ndarray]:
        
        """
        Encode original label(s) for a single feature.
        """
        
        le = self.categorical_encoders.get(feature)
        is_scalar = isinstance(original_value, str)
    
        if le is None:
            return original_value
    
        arr = [original_value] if is_scalar else list(original_value)
        encoded = le.transform(arr).astype(np.int32)
    
        return int(encoded[0]) if is_scalar else encoded

        
    def decode_feature(self, feature: str, encoded_value: Union[int, np.integer, List[int], np.ndarray]) -> Union[str, np.ndarray]:
        
        """
        Decode encoded integer(s) for a single feature.
        """
        
        le = self.categorical_encoders.get(feature)
        is_scalar = isinstance(encoded_value, (int, float, np.integer))
    
        if le is None:
            return encoded_value
    
        arr = np.array(
            [encoded_value] if is_scalar else encoded_value,
            dtype=np.int32
        )
        decoded = le.inverse_transform(arr)
    
        return decoded[0] if is_scalar else decoded

    
    def encode(self, arr: np.ndarray) -> np.ndarray:
        
        """
        Vectorized categorical encoding using feature indices.
        Handles (seq_len, n_features) and (n, seq_len, n_features).
        """

        arr = arr.copy()
        categorical_features = self.feature_config.categorical_features
        feature_to_idx = self.feature_config.feature_to_idx
    
        if len(categorical_features) == 0:
            return arr
    
        for feature in categorical_features:
    
            idx = feature_to_idx[feature]
            if arr.ndim == 2:
                flat = arr[:, idx].astype(str)
                arr[:, idx] = self.encode_feature(feature, flat)
    
            elif arr.ndim == 3:
                flat = arr[:, :, idx].reshape(-1).astype(str)
                arr[:, :, idx] = self.encode_feature(feature, flat).reshape(
                    arr.shape[0], arr.shape[1]
                )
                
            else:
                raise ValueError(f"Unsupported array dimension: {arr.ndim}")
    
        return arr


    def decode(self, arr: np.ndarray) -> np.ndarray:
        
        """
        Vectorized categorical decoding using feature indices.
        Handles (seq_len, n_features) and (n, seq_len, n_features).
        """

        arr = arr.astype(object, copy=True)
        categorical_features = self.feature_config.categorical_features
        feature_to_idx = self.feature_config.feature_to_idx
    
        if len(categorical_features) == 0:
            return arr
    
        for feature in categorical_features:
    
            idx = feature_to_idx[feature]
            if arr.ndim == 2:
                flat = arr[:, idx].reshape(-1)
                arr[:, idx] = self.decode_feature(feature, flat)
    
            elif arr.ndim == 3:
                flat = arr[:, :, idx].reshape(-1)
                arr[:, :, idx] = self.decode_feature(feature, flat).reshape(
                    arr.shape[0], arr.shape[1]
                )
                
            else:
                raise ValueError(f"Unsupported array dimension: {arr.ndim}")
    
        return arr


    def scale(self, arr: np.ndarray) -> np.ndarray:
        
        """
        Vectorized scaling continuous features using feature indices.
        Handles (seq_len, n_features) and (n, seq_len, n_features).
        """

        arr = arr.copy().astype(np.float32)
        scaler_features = self.feature_config.continuous_features
        feature_to_idx = self.feature_config.feature_to_idx
        
        scale_idx = np.array(
            [feature_to_idx[f] for f in scaler_features if f in feature_to_idx],
            dtype=np.int32
        )
        
        if len(scale_idx) == 0:
            return arr

        if arr.ndim == 2:
            arr[:, scale_idx] = self.scaler.transform(arr[:, scale_idx])
            
        elif arr.ndim == 3:
            n, seq_len, _ = arr.shape
            n_cont = len(scale_idx)
            flat = arr[:, :, scale_idx].reshape(-1, n_cont)
            arr[:, :, scale_idx] = self.scaler.transform(flat).reshape(
                n, seq_len, n_cont
            )
            
        else:
            raise ValueError(f"Unsupported array dimension: {arr.ndim}")
            
        return arr

    
    def unscale(self, arr: np.ndarray) -> np.ndarray:
        
        """
        Vectorized inverse scaling continuous features — back to original space using feature indices.
        Handles (seq_len, n_features) and (n, seq_len, n_features).
        """

        arr = arr.copy().astype(np.float32)
        scaler_features = self.feature_config.continuous_features
        feature_to_idx = self.feature_config.feature_to_idx
        
        scale_idx = np.array(
            [feature_to_idx[f] for f in scaler_features if f in feature_to_idx],
            dtype=np.int32
        )
        
        if len(scale_idx) == 0:
            return arr

        if arr.ndim == 2:
            arr[:, scale_idx] = self.scaler.inverse_transform(arr[:, scale_idx])
            
        elif arr.ndim == 3:
            n, seq_len, _ = arr.shape
            n_cont = len(scale_idx)
            flat = arr[:, :, scale_idx].reshape(-1, n_cont)
            arr[:, :, scale_idx] = self.scaler.inverse_transform(flat).reshape(
                n, seq_len, n_cont
            )
            
        else:
            raise ValueError(f"Unsupported array dimension: {arr.ndim}")
            
        return arr


    def get_event_prototype(self, activity: Union[str, np.ndarray], event: np.ndarray) -> np.ndarray:
    
        """Get prototype by substituting dynamic features."""
    
        feature_order = self.feature_config.feature_order
        dynamic_features = self.feature_config.dynamic_features
    
        dyn_idx = np.array([
            i for i, f in enumerate(feature_order)
            if f in dynamic_features
        ])
    
        if event.ndim == 1:
            event_prototype = event.copy()
            act = activity
            proto = self.activity_prototypes.get(act)
            if proto is None:
                raise ValueError(f"No prototype found for activity {act}")
            event_prototype[dyn_idx] = proto[dyn_idx]
    
            return event_prototype
    
        elif event.ndim == 2:
            B, F = event.shape
            activity = np.asarray(activity).reshape(-1)
            if activity.shape[0] != B:
                activity = np.full(B, activity.item() if activity.ndim == 0 else activity[0])
            output = event.copy()
            for b in range(B):
                act = activity[b]
                proto = self.activity_prototypes.get(act)
                if proto is None:
                    raise ValueError(f"No prototype found for activity {act}")
                output[b, dyn_idx] = proto[dyn_idx]
            return output

        else:
            raise ValueError(f"Unsupported array dimension: {event.ndim}")

    
    def summary(self) -> None:
         
        """Print readable summary of preprocessing artifacts."""
         
        print(f"{'PreprocessorArtifacts':=^60}")
        print(f"  scaler:              {type(self.scaler).__name__}")
        print(f"  encoders:            {list(self.categorical_encoders.keys())}")
        print(f"  activity_prototypes: {len(self.activity_prototypes)} activities")
        print("=" * 60)


    # --- Preprocess helpers ---
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Transform a given dataframe - encode + scale.
        """

        df = df.copy()
        feature_order = list(self.feature_config.feature_order)
        
        missing_features = set(feature_order) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
    
        arr = df[feature_order].to_numpy()
        
        arr = self.encode(arr)
        arr = self.scale(arr)
        
        df[feature_order] = arr
    
        return df


    def transform_dataframe_to_processdataset(
        self, df: pd.DataFrame, case_id_field: str, sort_field: Optional[str] = None
    ) -> ProcessDataset:
        
        """
        Convert dataframe into ProcessDataset for next-event prediction.
        """
    
        feature_to_idx = self.feature_config.feature_to_idx
        feature_order = list(self.feature_config.feature_order)
        activity_feature = self.feature_config.activity_feature
        
        dynamic_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.dynamic_categorical_features]
        dynamic_cont_features = [f for f in self.feature_config.feature_order if f in self.feature_config.dynamic_continuous_features]
        static_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.static_categorical_features]
        static_cont_features = [f for f in self.feature_config.feature_order if f in self.feature_config.static_continuous_features]
    
        # --- Encode + scale ---
        df = self.transform_dataframe(df)
    
        cases = []
        labels = []
    
        for _, group in df.groupby(case_id_field):
            group = (
                group.sort_values(sort_field)
                if sort_field is not None else group.sort_index()
            )
                
            x_full = group[feature_order].to_numpy(dtype=np.float32)
            y_full = group[activity_feature].to_numpy()
    
            if len(x_full) <= 1:  # Guard against single-event cases
                continue
    
            x = x_full[:-1]  # Remove last event
            y = y_full[1:]
    
            cases.append(x)
            labels.append(y)

        if not cases:
            raise ValueError("No valid process cases found with length > 1.")
    
        # --- Padding ---
        max_len = max(len(c) for c in cases)
        B = len(cases)
        F = len(feature_order)
    
        batch = np.zeros((B, max_len, F), dtype=np.float32)
        label_batch = np.full((B, max_len), -1, dtype=np.int32)
        mask = np.zeros((B, max_len), dtype=np.int32)
    
        for i, (seq, lab) in enumerate(zip(cases, labels)):
            L = len(seq)
    
            batch[i, :L] = seq
            label_batch[i, :L] = lab
            mask[i, :L] = 1
    
        # --- Unpack using Batch processing (3D) logic ---
        dynamic_cat = [batch[:, :, feature_to_idx[f]] for f in dynamic_cat_features]
        dynamic_cont = batch[:, :, [feature_to_idx[f] for f in dynamic_cont_features]]
        
        static_cat = [batch[:, 0, feature_to_idx[f]] for f in static_cat_features]
        static_cont = batch[:, 0, [feature_to_idx[f] for f in static_cont_features]]

        # --- Create ProcessDataset ---
        return ProcessDataset(
            dynamic_cat=dynamic_cat, 
            dynamic_cont=dynamic_cont, 
            static_cat=static_cat, 
            static_cont=static_cont, 
            labels=label_batch, 
            mask=mask
        )

        
    def build_model_input(self, arr: np.ndarray) -> Dict[str, Union[List[np.ndarray], np.ndarray, None]]:

        """
        Builds model input from prefix or prefix batch (no padding).
        Handles (seq_len, n_features) and (n, seq_len, n_features).
        """

        arr = self.encode(arr)
        arr = self.scale(arr)
    
        feature_to_idx = self.feature_config.feature_to_idx

        dynamic_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.dynamic_categorical_features]
        dynamic_cont_features = [f for f in self.feature_config.feature_order if f in self.feature_config.dynamic_continuous_features]
        static_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.static_categorical_features]
        static_cont_features = [f for f in self.feature_config.feature_order if f in self.feature_config.static_continuous_features]

        # --- Single prefix (2D) ---
        if arr.ndim == 2:
            # Standardise dynamic features to have batch dimension: (1, seq_len) and (1, seq_len, n_feats)
            dynamic_cat = [arr[:, feature_to_idx[f]][None, :] for f in dynamic_cat_features]
            dynamic_cont = arr[:, [feature_to_idx[f] for f in dynamic_cont_features]][None, :, :]
            
            # Standardise static features to have batch dimension: (1,) and (1, n_feats)
            static_cat = [np.array([arr[0, feature_to_idx[f]]]) for f in static_cat_features]
            static_cont = arr[0, [feature_to_idx[f] for f in static_cont_features]][None, :]
    
        # --- Batch processing (3D) ---
        elif arr.ndim == 3:
            dynamic_cat = [arr[:, :, feature_to_idx[f]] for f in dynamic_cat_features]
            dynamic_cont = arr[:, :, [feature_to_idx[f] for f in dynamic_cont_features]]
            
            static_cat = [arr[:, 0, feature_to_idx[f]] for f in static_cat_features]
            static_cont = arr[:, 0, [feature_to_idx[f] for f in static_cont_features]]
            
        else:
            raise ValueError(f"Unsupported array dimension: {arr.ndim}")
    
        return {
            "dynamic_cat": dynamic_cat,
            "dynamic_cont": dynamic_cont,
            "static_cat": static_cat,
            "static_cont": static_cont,
            "mask": None
        }


    def reconstruct_trace_from_model_input(
        self,
        model_input:  Dict[str, Union[List[torch.Tensor], torch.Tensor, None]],
    ) -> np.ndarray:
        
        """
        Reverse build_model_input().
    
        Inputs:
            dynamic_cat : list[(B,T)] or list[(1,T)]
            dynamic_cont: (B,T,Dc)
            static_cat  : list[(B,)] or list[(1,)]
            static_cont : (B,Sc)
    
        Returns:
            trace : (B,T,F) or (T,F)
        """

        dynamic_cat = model_input["dynamic_cat"]
        dynamic_cont = model_input["dynamic_cont"]
        static_cat = model_input["static_cat"]
        static_cont = model_input["static_cont"]
        mask = model_input["mask"]
    
        feature_to_idx = self.feature_config.feature_to_idx
        n_features = len(self.feature_config.feature_order)
    
        dynamic_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.dynamic_categorical_features]
        dynamic_cont_features = [f for f in self.feature_config.feature_order if f in self.feature_config.dynamic_continuous_features]
    
        static_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.static_categorical_features]
        static_cont_features = [f for f in self.feature_config.feature_order if f in self.feature_config.static_continuous_features]
    
        B = dynamic_cont.shape[0]
        T = dynamic_cont.shape[1]
    
        trace = np.zeros((B, T, n_features), dtype=np.float32)
    
        # --- Dynamic categorical ---
        for arr, feature in zip(dynamic_cat, dynamic_cat_features):
            idx = feature_to_idx[feature]
            trace[:, :, idx] = arr
    
        # --- Dynamic continuous ---
        for i, feature in enumerate(dynamic_cont_features):
            idx = feature_to_idx[feature]
            trace[:, :, idx] = dynamic_cont[:, :, i]
    
        # --- Static categorical ---
        for arr, feature in zip(static_cat, static_cat_features):
            idx = feature_to_idx[feature]
    
            trace[:, :, idx] = arr[:, None]      # broadcast across time
    
        # --- Static continuous ---
        for i, feature in enumerate(static_cont_features):
            idx = feature_to_idx[feature]
    
            trace[:, :, idx] = static_cont[:, None, i]

        # --- Zero out padded timesteps ---
        if mask is not None:
            trace *= mask[..., None].astype(trace.dtype)
    
        # remove batch dimension for single trace
        if B == 1:
            trace = trace[0]

        # --- Unscale and decode ---
        trace = self.unscale(trace)
        trace = self.decode(trace)
    
        return trace
    

    # --- Metadata for model configurations ---
    def get_categorical_feature_cardinality(self) -> Dict[str, Dict[str, int]]:
        
        """
        Returns number of classes for each dynamic, static categorical feature in feature_order.
        """
        
        dynamic_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.dynamic_categorical_features]
        static_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.static_categorical_features]

        dynamic_categorical_info = {f: len(self.categorical_encoders[f].classes_) for f in dynamic_cat_features}
        static_categorical_info = {f: len(self.categorical_encoders[f].classes_) for f in static_cat_features}
    
        return {
            "dynamic_categorical_info": dynamic_categorical_info,
            "static_categorical_info": static_categorical_info
        }
      

    def get_embedding_metadata(self, max_dim: int = 50) -> Dict[str, Any]:
        
        """
        Returns embedding metadata for each dynamic, static categorical feature in feature order,
        along with continuous feature counts and target class counts.
        
        Output:
            {
                "dynamic_categorical_info": [(num_categories, emb_dim), ...],
                "static_categorical_info": [(num_categories, emb_dim), ...],
                "n_dynamic_continuous": int,
                "n_static_continuous": int,
                "n_classes": int
            }
        """
        
        dynamic_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.dynamic_categorical_features]
        static_cat_features = [f for f in self.feature_config.feature_order if f in self.feature_config.static_categorical_features]
    
        dynamic_categorical_info = []
        for f in dynamic_cat_features:
            n_categories = len(self.categorical_encoders[f].classes_)
            embedding_dim = min(max_dim, (n_categories + 1) // 2)
            dynamic_categorical_info.append((n_categories, embedding_dim))
    
        static_categorical_info = []
        for f in static_cat_features:
            n_categories = len(self.categorical_encoders[f].classes_)
            embedding_dim = min(max_dim, (n_categories + 1) // 2)
            static_categorical_info.append((n_categories, embedding_dim))
    
        n_dynamic_continuous = len(self.feature_config.dynamic_continuous_features)
        n_static_continuous = len(self.feature_config.static_continuous_features)
        n_classes = len(self.categorical_encoders[self.feature_config.activity_feature].classes_)
    
        return {
            "dynamic_categorical_info": dynamic_categorical_info,
            "static_categorical_info": static_categorical_info,
            "n_dynamic_continuous": n_dynamic_continuous,
            "n_static_continuous": n_static_continuous,
            "n_classes": n_classes
        }
