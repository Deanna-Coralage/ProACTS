import os
import joblib
import warnings

import pandas as pd
import numpy as np

from dataclasses import dataclass, replace

from typing import Tuple, Dict, List, Any



@dataclass(frozen=True)
class FeatureConfig:
    
    """
    Holds all FEATURE METADATA.

    feature_specs: dict of feature definitions.
    For example:
    {
        "time_delta": {
            "type":            "continuous",
            "level":           "event",
            "vary":            True,
            "permitted_range": [0, 480],        ← user_defined OR quantile_derived
            "mad":             45.2,            ← on winsorized (if not robust) original data
            "skewness":        1.2,
            "range_source":    "user_defined",  ← or "quantile_derived"
            "quantile_low":    0.01,           
            "quantile_high":   0.95            
        },
        "resource": {
            "type":              "categorical",
            "level":             "event",
            "vary":              True,
            "valid_categories":  ['R1', 'R2','R3'],
            "categories_source": "user_defined" ← or "data_derived"
        },
        "cost": {
            "type":             "continuous",
            "level":            "static",
            "vary":             False,           ← immutable
            "permitted_range":  [0, 50000],
            "mad":              20.1,
            "skewness":         2.0,
            "range_source":     "data_derived",
            "quantile_low":     0.01,           
            "quantile_high":    0.95   
        },
        "activity": {
            "type":             "categorical",
            "level":            "event",
            "vary":             False,
            "valid_categories": ['A', 'B', 'C'],
            "categories_source":"data_derived"
        }
    }
    """
    
    feature_specs:          Dict[str, Dict[str, Any]]
    
    activity_feature:       str
    feature_order:          Tuple[str, ...]   # Ordered

    
    def __post_init__(self):
        object.__setattr__(self, "feature_to_idx",
            {f: i for i, f in enumerate(self.feature_order)}
        )
        
        object.__setattr__(self, "categorical_features",
            [f for f, cfg in self.feature_specs.items()
            if cfg.get("type", "continuous") == "categorical"]

        )
        object.__setattr__(self, "continuous_features",
            [f for f, cfg in self.feature_specs.items()
            if cfg.get("type", "continuous") == "continuous"]
        )
        
        object.__setattr__(self, "static_features",
            [f for f, cfg in self.feature_specs.items()
            if cfg.get("level", "event") == "case"]
        )
        object.__setattr__(self, "dynamic_features",
            [f for f, cfg in self.feature_specs.items()
            if cfg.get("level", "event") == "event"]
        )
        
        object.__setattr__(self, "static_categorical_features",
            [f for f, cfg in self.feature_specs.items()
            if cfg.get("level", "event") == "case"
            and cfg["type"] == "categorical"]
        )
        object.__setattr__(self, "dynamic_categorical_features",
            [f for f, cfg in self.feature_specs.items()
            if cfg.get("level", "event") == "event"
            and cfg["type"] == "categorical"]
        )

        object.__setattr__(self, "static_continuous_features",
            [f for f, cfg in self.feature_specs.items()
            if cfg.get("level", "event") == "case"
            and cfg["type"] == "continuous"]
        )
        object.__setattr__(self, "dynamic_continuous_features",
            [f for f, cfg in self.feature_specs.items()
            if cfg.get("level", "event") == "event"
            and cfg["type"] == "continuous"]
        )
        
    
    # --- VARYING features (for genetic search) ---
    @property
    def features_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg.get("vary", False)]
        
    @property
    def static_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg.get("level", "event") == "case"
                and cfg.get("vary", False)]
    
    @property
    def dynamic_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg.get("level", "event") == "event"
                and cfg.get("vary", False)]

    @property
    def categorical_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg["type"] == "categorical"
                and cfg.get("vary", False)]

    @property
    def static_categorical_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg["type"] == "categorical"
                and cfg.get("level", "event") == "case"
                and cfg.get("vary", False)]

    @property
    def dynamic_categorical_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg["type"] == "categorical"
                and cfg.get("level", "event") == "event"
                and cfg.get("vary", False)]
    
    @property
    def continuous_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg["type"] == "continuous"
                and cfg.get("vary", False)]

    @property
    def static_continuous_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg["type"] == "continuous"
                and cfg.get("level", "event") == "case"
                and cfg.get("vary", False)]

    @property
    def dynamic_continuous_to_vary(self) -> List[str]:
        return [f for f, cfg in self.feature_specs.items()
                if cfg["type"] == "continuous"
                and cfg.get("level", "event") == "event"
                and cfg.get("vary", False)]

    @property
    def permitted_range(self) -> Dict[str, List[float]]:
        return {f: cfg["permitted_range"]
                for f, cfg in self.feature_specs.items()
                if cfg["type"] == "continuous"}

    @property
    def valid_categories(self) -> Dict[str, List[Any]]:
        return {f: cfg["valid_categories"]
                for f, cfg in self.feature_specs.items()
                if cfg["type"] == "categorical"}

    @property
    def mad(self) -> Dict[str, float]:
        return {f: cfg["mad"]
                for f, cfg in self.feature_specs.items()
                if cfg["type"] == "continuous"
                and "mad" in cfg}

    
    # --- Save ---
    def save(self, path: str = "./pretrained_models/") -> None:

        """ Save feature configs to disk. """

        os.makedirs(path, exist_ok=True)

        meta = {
            "feature_specs": self.feature_specs,
            "activity_feature": self.activity_feature,
            "feature_order": self.feature_order,
        }

        joblib.dump(meta, f"{path}/feature_config_meta.pkl")
        

    # --- Helpers ---
    def get_type(self, feature: str) -> str:
        return self.feature_specs[feature]["type"]

    def is_varying(self, feature: str) -> bool:
        return self.feature_specs[feature].get("vary", False)

    def is_static(self, feature: str) -> bool:
        return self.feature_specs[feature].get("level", "event") == "case"

    def summary(self) -> None:
        feature_w = 30
        type_w = 14
        level_w = 8
        vary_w = 6
        range_w = 40
        src_w = 20
    
        header = (f"{'Feature':<{feature_w}} {'Type':<{type_w}} {'Level':<{level_w}} {'Vary':<{vary_w}} "
                  f"{'Range/Categories':<{range_w}} {'MAD':<10} {'Source':<{src_w}}")

        print(f"  activity_feature:    {self.activity_feature}")
        print(f"  feature_order:       {list(self.feature_order)}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
    
        for feature, cfg in self.feature_specs.items():
            vary = "yes" if cfg.get("vary", False) else "no"
            level = cfg.get("level", "event")
    
            if cfg["type"] == "continuous":
                r = cfg.get("permitted_range", [0, 0])
                mad = cfg.get("mad", float("nan"))
                src = cfg.get("range_source", "")
                range_str = f"[{r[0]:.2f}, {r[1]:.2f}]"
                print(f"{feature:<{feature_w}} {'continuous':<{type_w}} {level:<{level_w}} {vary:<{vary_w}} "
                      f"{range_str:<{range_w}} {mad:<10.4f} {src:<{src_w}}")
            else:
                cats = cfg.get("valid_categories", [])
                src = cfg.get("categories_source", "")
                cats_str = (str(cats[:3])[:-1] + ", ...]"
                            if len(cats) > 3
                            else str(cats))
                print(f"{feature:<{feature_w}} {'categorical':<{type_w}} {level:<{level_w}} {vary:<{vary_w}} "
                      f"{cats_str:<{range_w}} {'N/A':<10} {src:<{src_w}}")

    def with_vary(
        self,
        vary_updates:  Dict[str, bool],
    ) -> "FeatureConfig":
            
        """
        Return a new FeatureConfig with only the 'vary' field updated.
    
        Parameters
        ----------
        vary_updates:
            {"feature_name": True/False}
        """
    
        feature_specs = {
            feature: cfg.copy()
            for feature, cfg in self.feature_specs.items()
        }
    
        for feature, vary in vary_updates.items():
    
            if feature not in feature_specs:
                raise KeyError(f"Unknown feature '{feature}'.")
    
            feature_specs[feature]["vary"] = bool(vary)
    
        return replace(
            self,
            feature_specs=feature_specs,
        )

        
    # --- Constructor ---
    @classmethod
    def from_dataframe(
        cls,
        df:                      pd.DataFrame,
        feature_specs:           Dict[str, Dict[str, Any]],
        activity_feature:        str,
        is_robust:               bool = False,
        default_quantile_low:    float = 0.05,
        default_quantile_high:   float = 0.95,
    ) -> "FeatureConfig":
            
        """
        Build FeatureConfig from training dataframe.
        Per-feature spec can include:
            Continuous:
              "permitted_range": [min, max]  ← user defined, skips quantile computation
              "quantile_low":    float       ← per-feature override of default
              "quantile_high":   float       ← per-feature override of default
              (if none given, uses default_quantile_low / default_quantile_high)
        
            Categorical:
              "valid_categories": [...]      ← user defined original labels
        """

        if len(df) < 100:
            warnings.warn(
                f"Only {len(df)} rows — quantile winsorization may be ineffective. "
                f"Consider tighter quantiles or specifying 'permitted_range' directly in feature_specs.",
                UserWarning
            )
            
        missing = set(feature_specs.keys()) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        data_features = [c for c in df.columns if c in feature_specs]
        df = df[data_features].copy()
        feature_order = tuple(data_features)
            
        feature_defs = {}

        for feature, spec in feature_specs.items():
            feature_def = dict(spec)

            is_varying = spec.get("vary", False)

        
            # --- Continuous feature handling (in original unscaled space) --- 
            if spec["type"] == "continuous":

                feature_data = df[feature].copy().dropna()
                skewness = float(feature_data.skew())
                feature_def["skewness"] = skewness

                # --- Permitted range - only for varying features ---
                if "permitted_range" in spec:
                    q_low_val = float(spec["permitted_range"][0])
                    q_high_val = float(spec["permitted_range"][1])
                    feature_def["range_source"] = "user_defined"
    
                else:
                    q_low = spec.get("quantile_low",  default_quantile_low)
                    q_high = spec.get("quantile_high", default_quantile_high)
    
                    # Auto-tighten right tail for heavily skewed features (if not quantile_high set explicitly)
                    if skewness > 2.0 and "quantile_high" not in spec:
                        q_high = min(q_high, 0.95)
                        warnings.warn(
                            f"Feature '{feature}' is heavily right-skewed (skew={skewness:.2f}). Auto-tightening quantile_high to {q_high}. "
                            f"Override by setting 'quantile_high' or 'permitted_range' in feature_specs.",
                            UserWarning
                        )
    
                    q_low_val = float(feature_data.quantile(q_low))
                    q_high_val = float(feature_data.quantile(q_high))
                    feature_def["quantile_low"] = q_low
                    feature_def["quantile_high"] = q_high
                    feature_def["range_source"] = "quantile_derived"
    
                feature_def["permitted_range"] = [q_low_val, q_high_val]
                
                # --- MAD (for proximity calculations) ---
                if is_robust:
                    # RobustScaler — MAD is already outlier robust, no winsorization needed
                    median = float(np.median(feature_data))
                    mad_val = float(np.median(np.abs(feature_data - median)))
                else:
                    # StandardScaler or unknown — winsorize before MAD, outliers inflate std
                    clipped = feature_data.clip(lower=q_low_val, upper=q_high_val)
                    median = float(np.median(clipped))
                    mad_val = float(np.median(np.abs(clipped - median)))

                feature_def["mad"] = max(mad_val, 1e-8)

        
            # --- Categorical feature handling --- 
            elif spec["type"] == "categorical":

                if "valid_categories" in spec:
                    raw_cats = sorted(spec["valid_categories"])
                    feature_def["valid_categories"] = raw_cats
                    feature_def["categories_source"] = "user_defined"
                else:
                    raw_cats = sorted(df[feature].dropna().unique().tolist())
                    feature_def["valid_categories"] = raw_cats
                    feature_def["categories_source"] = "data_derived"

            # --- Append --- 
            feature_defs[feature] = feature_def
        

        return cls(
            feature_specs=feature_defs,
            activity_feature=activity_feature,
            feature_order=feature_order
        )


    # --- Load ---
    @classmethod
    def load(cls, path: str = "./pretrained_models/") -> "FeatureConfig":

        """ Load feature configs from disk. """

        meta = joblib.load(f"{path}/feature_config_meta.pkl")

        return cls(
            feature_specs=meta["feature_specs"],
            activity_feature=meta["activity_feature"],
            feature_order=tuple(meta["feature_order"])
        )
