import os
import json

import pickle

from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import random

from dataclasses import dataclass

from typing import Optional, Tuple, List, Set, Dict, Any, Sequence

from collections import defaultdict, Counter
from itertools import combinations

from utils.feature_utils import df_to_sequence_array

from config.feature_config import FeatureConfig

from process.engine import ProcessModelConstraintEngine



@dataclass
class ExperimentHandler:

    constraint_engine:  Optional[ProcessModelConstraintEngine] = None

    # --- Experiment Generation ---
    @staticmethod
    def _hamming_diff_positions(
        a: Tuple[str, ...],
        b: Tuple[str, ...],
    ) -> List[int]:
        return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]

    def is_same_branching_set(self, a: str, b: str) -> bool:
        if self.constraint_engine is None:
            return False

        return any(
            a in branch and b in branch
            for branch in self.constraint_engine.branching_sets
        )

    def _filter_flexible_internal_diffs(
        self,
        source_trace,
        source_pattern,
        target_pattern,
        start_pos,
    ):
        raw_diff = self._hamming_diff_positions(source_pattern, target_pattern)
    
        if self.constraint_engine is None:
            return raw_diff
    
        flexible = self.constraint_engine.generate_flexible_constraints(
            trace_activities=source_trace
        )
    
        kept = []
    
        for local_pos in raw_diff:
            abs_pos = start_pos + local_pos
            target_act = target_pattern[local_pos]
    
            allowed = None
            for (s, e), acts in flexible.segments.items():
                if s <= abs_pos <= e:
                    allowed = acts
                    break
    
            # remove if target is already allowed by flexible segment
            if allowed is not None and target_act in allowed:
                continue
    
            kept.append(local_pos)
    
        return kept

    @staticmethod
    def _stratified_sample_cases_by_trace_length(
        candidate_cases,
        sequences,
        max_cases,
        random_state = 42,
    ):
        rng = random.Random(random_state)
    
        cases_by_len = defaultdict(list)
    
        for case_id in candidate_cases:
            cases_by_len[len(sequences[case_id])].append(case_id)
    
        for L in cases_by_len:
            rng.shuffle(cases_by_len[L])
    
        selected = []
    
        lengths = sorted(cases_by_len)
    
        while len(selected) < max_cases and lengths:
            for L in list(lengths):
                if len(selected) >= max_cases:
                    break
    
                if cases_by_len[L]:
                    selected.append(cases_by_len[L].pop(0))
    
                if not cases_by_len[L]:
                    lengths.remove(L)
    
        return selected
        
    def generate_experiments_from_log(
        self,
        df:                         pd.DataFrame,
        case_id_field:              str,
        activity_field:             str,
        sort_field:                 str,
        min_len:                    int = 3,
        max_len:                    int = 15,
        min_support:                int = 20,
        min_alt_support:            int = 20,
        min_desired:                int = 1,
        max_desired:                int = 10,
        max_cases_per_experiment:   int = 10,
        max_templates_per_length:   int = 5,
        max_templates:              int = 50,
        only_branching_conditions:  bool = False,
        save_path:                  Optional[str] = None
    ) -> pd.DataFrame:

        if only_branching_conditions:
            if self.constraint_engine is None or not self.constraint_engine.branching_sets:
                raise ValueError(
                    "A ProcessModelConstraintEngine with branching_sets is required "
                    "when only_branching_conditions=True."
                )

        df_sorted = df.sort_values([case_id_field, sort_field])

        sequences = {
            case_id: tuple(group[activity_field].astype(str).tolist())
            for case_id, group in df_sorted.groupby(case_id_field, sort=False)
        }

        # --- Mine sliding-window patterns ---
        pattern_counter = Counter()
        pattern_cases = defaultdict(list)

        for case_id, seq in sequences.items():
            for L in range(min_len, min(max_len, len(seq)) + 1):
                for start in range(0, len(seq) - L + 1):

                    pattern = seq[start:start + L]
                    key = (L, start, pattern)

                    pattern_counter[key] += 1
                    pattern_cases[key].append(case_id)

        frequent_patterns = [
            (L, start, pattern, support)
            for (L, start, pattern), support in pattern_counter.items()
            if support >= min_support
        ]

        # Shorter windows first
        frequent_patterns = sorted(
            frequent_patterns,
            key=lambda x: (x[0], x[1], -x[3]),
        )

        patterns_by_length = defaultdict(list)

        for L, start, pattern, support in frequent_patterns:
            patterns_by_length[L].append((start, pattern, support))

        rows = []
        template_id = 0
        seen_template_keys = set()

        # --- Compare windows of same length and same start position ---
        for L in sorted(patterns_by_length):

            templates_this_length = 0
            patterns = patterns_by_length[L]

            for (src_start, src_pattern, src_support), (tgt_start, tgt_pattern, tgt_support) in combinations(patterns, 2):

                if templates_this_length >= max_templates_per_length:
                    break

                if template_id >= max_templates:
                    final_df = pd.DataFrame(rows)
                    self._handle_internal_save(final_df, locals(), save_path)
                    return pd.DataFrame(rows)

                if src_start != tgt_start:
                    continue

                for source_pattern, target_pattern, source_support, target_support in [
                    (src_pattern, tgt_pattern, src_support, tgt_support),
                    (tgt_pattern, src_pattern, tgt_support, src_support),
                ]:

                    if templates_this_length >= max_templates_per_length:
                        break

                    if template_id >= max_templates:
                        final_df = pd.DataFrame(rows)
                        self._handle_internal_save(final_df, locals(), save_path)
                        return pd.DataFrame(rows)

                    if target_support < min_alt_support:
                        continue

                    # Use one representative source trace to test flexible filtering
                    rep_case = pattern_cases[(L, src_start, source_pattern)][0]
                    rep_trace = sequences[rep_case]

                    diff_local_positions = self._filter_flexible_internal_diffs(
                        source_trace=rep_trace,
                        source_pattern=source_pattern,
                        target_pattern=target_pattern,
                        start_pos=src_start,
                    )

                    diff_local_positions = [
                        p
                        for p in diff_local_positions
                        if src_start + p != 0
                    ]
                    
                    if not (min_desired <= len(diff_local_positions) <= max_desired):
                        continue
                    
                    valid = True

                    for p in diff_local_positions:
                        src_act = source_pattern[p]
                        tgt_act = target_pattern[p]

                        is_branch_alt = self.is_same_branching_set(src_act, tgt_act)
                        is_frequent_deviation = target_support >= min_alt_support

                        if only_branching_conditions:
                            if not is_branch_alt:
                                valid = False
                                break
                        else:
                            if not (is_branch_alt or is_frequent_deviation):
                                valid = False
                                break

                    if not valid:
                        continue

                    desired_positions = {
                        src_start + p: {target_pattern[p]}
                        for p in diff_local_positions
                    }

                    template_key = tuple(
                        sorted(
                            (abs_pos, source_pattern[abs_pos - src_start], tuple(sorted(allowed)))
                            for abs_pos, allowed in desired_positions.items()
                        )
                    )
                    
                    if template_key in seen_template_keys:
                        continue
                    
                    seen_template_keys.add(template_key)

                    candidate_cases = pattern_cases[
                        (L, src_start, source_pattern)
                    ]

                    # Prefer variety in full trace lengths
                    selected_cases = self._stratified_sample_cases_by_trace_length(
                        candidate_cases=candidate_cases,
                        sequences=sequences,
                        max_cases=max_cases_per_experiment,
                        random_state=42,
                    )

                    for case_id in selected_cases:
                        rows.append({
                            "template_id": template_id,
                            case_id_field: case_id,
                            "trace_length": len(sequences[case_id]),
                            "start_pos": src_start,
                            "length": L,
                            "num_desired": len(diff_local_positions),
                            "source_pattern": source_pattern,
                            "target_pattern": target_pattern,
                            "desired_positions": desired_positions,
                            "source_support": source_support,
                            "target_support": target_support,
                            "num_cases_for_template": len(selected_cases),
                            "experiment_type": (
                                "branching_only"
                                if only_branching_conditions
                                else "window_variant_mixed"
                            ),
                        })

                    templates_this_length += 1
                    template_id += 1

        final_df = pd.DataFrame(rows)
        self._handle_internal_save(final_df, locals(), save_path)
        
        return final_df


    # --- Experiment Selector ---
    def select_random_experiment_instances(
        self,
        exp_df:                 pd.DataFrame,
        *,
        case_id_col:            str,
        n_per_desired:          int = 2,
        desired_values:         Optional[Sequence[int]] = None,
        random_state:           int = 42,
        stratify_trace_length:  bool = True,
    ) -> pd.DataFrame:
        
        """
        Select a small set of factual instances for instance-level plots.
        """
    
        required_columns = {
            "template_id",
            case_id_col,
            "num_desired",
            "trace_length",
        }
    
        missing = required_columns - set(exp_df.columns)
    
        if missing:
            raise KeyError(
                f"Missing required columns: {sorted(missing)}"
            )
    
        # A factual experiment is uniquely identified by template + case.
        candidates = (
            exp_df
            .drop_duplicates(
                subset=[
                    "template_id",
                    case_id_col,
                    "trace_length",
                    "num_desired",
                ]
            )
            .copy()
        )
    
        candidates = candidates.dropna(
            subset=[
                "template_id",
                case_id_col,
                "trace_length",
                "num_desired",
            ]
        )
    
        candidates["num_desired"] = (
            candidates["num_desired"].astype(int)
        )
    
        if desired_values is not None:
            desired_values = {
                int(value)
                for value in desired_values
            }
    
            candidates = candidates[
                candidates["num_desired"].isin(desired_values)
            ].copy()
    
        if candidates.empty:
            raise ValueError(
                "No candidate experiment instances remain after filtering."
            )
    
        rng = np.random.default_rng(random_state)
        selected_groups = []
    
        for num_desired, group in candidates.groupby(
            "num_desired",
            sort=True,
        ):
            group = group.copy()
    
            n_select = min(
                n_per_desired,
                len(group),
            )
    
            if not stratify_trace_length:
                selected_indices = rng.choice(
                    group.index.to_numpy(),
                    size=n_select,
                    replace=False,
                )
    
                selected_groups.append(
                    group.loc[selected_indices]
                )
                continue
    
            # Randomize rows within each trace-length group.
            length_groups = {}
    
            for trace_length, length_df in group.groupby(
                "trace_length",
                sort=True,
            ):
                shuffled_indices = rng.permutation(
                    length_df.index.to_numpy()
                ).tolist()
    
                length_groups[int(trace_length)] = shuffled_indices
    
            available_lengths = list(length_groups.keys())
            rng.shuffle(available_lengths)
    
            selected_indices = []
    
            # Round-robin sampling encourages trace-length diversity.
            while (
                len(selected_indices) < n_select
                and available_lengths
            ):
                for trace_length in list(available_lengths):
                    if len(selected_indices) >= n_select:
                        break
    
                    indices = length_groups[trace_length]
    
                    if indices:
                        selected_indices.append(
                            indices.pop()
                        )
    
                    if not indices:
                        available_lengths.remove(
                            trace_length
                        )
    
            selected_groups.append(
                group.loc[selected_indices]
            )
    
        selected_df = pd.concat(
            selected_groups,
            ignore_index=True,
        )
    
        return (
            selected_df
            .sort_values(
                [
                    "num_desired",
                    "trace_length",
                    "template_id",
                    case_id_col,
                ]
            )
            .reset_index(drop=True)
        )


    # --- Added Helper Method to Package and Save State Safely ---
    def _handle_internal_save(self, df: pd.DataFrame, local_vars: Dict, save_path: Optional[str] = None) -> None:
        
        if save_path is None:
            return

        if not save_path.endswith((".pkl", ".pickle")):
            save_path += ".pkl"

        param_keys = [
            "min_len", "max_len", "min_support", "min_alt_support", 
            "min_desired", "max_desired", "max_cases_per_experiment", 
            "max_templates_per_length", "max_templates", "only_branching_conditions"
        ]
        
        config_params = {k: local_vars[k] for k in param_keys if k in local_vars}

        metadata = {
            "generator_class": self.__class__.__name__,
            "case_id_field": local_vars.get("case_id_field"),
            "activity_field": local_vars.get("activity_field"),
            "sort_field": local_vars.get("sort_field"),
            "parameters": config_params,
            "total_templates_generated": int(df["template_id"].nunique()) if not df.empty else 0,
            "total_experiment_rows": len(df)
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({
                "experiments": df,
                "metadata": metadata
            }, f)

        print(f"Saved dataframe and parameter metadata payload to: {save_path}")

    @staticmethod
    def load(file_path: str) -> Tuple[pd.DataFrame, dict]:

        if not os.path.exists(file_path) and not file_path.endswith((".pkl", ".pickle")):
            if os.path.exists(file_path + ".pkl"):
                file_path += ".pkl"
            elif os.path.exists(file_path + ".pickle"):
                file_path += ".pickle"

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No experimental database payload found at: '{file_path}'. "
                f"Please check your path directory configuration."
            )

        with open(file_path, "rb") as f:
            payload = pickle.load(f)

        df = payload.get("experiments")
        metadata = payload.get("metadata", {})

        total_templates = metadata.get("total_templates_generated", 0)
        print(f"Successfully reloaded {total_templates} experimental templates from: {file_path}")
        
        return df, metadata


    # --- Experiment Runner ---
    @staticmethod
    def _occurrence_at_position(
        trace_activities: Tuple[str, ...],
        pos: int,
    ) -> Tuple[str, int]:
    
        act = trace_activities[pos]
        occurrence = sum(
            1 for x in trace_activities[:pos + 1]
            if x == act
        )
    
        return act, occurrence

    @staticmethod
    def _make_excel_safe(x):
        if hasattr(x, "segments") or isinstance(x, (set, tuple, list, dict)):
            return json.dumps(
                ExperimentHandler._make_json_safe(x)
            )
        return x

    @staticmethod
    def _make_json_safe(obj):
        if hasattr(obj, "segments"):
            return ExperimentHandler._make_json_safe(obj.segments)
    
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {str(k): ExperimentHandler._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ExperimentHandler._make_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj


    @staticmethod
    def _save_experiment_workbook(
        results_df:     pd.DataFrame,
        exp_df:         pd.DataFrame,
        cf_analysis_df: pd.DataFrame, 
        output_path:    str,
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
        # one row per template
        templates_df = (
            exp_df
            .drop_duplicates("template_id")
            .copy()
        )
    
        # make complex columns Excel-safe
        for df_ in [results_df, templates_df, cf_analysis_df]:
            if df_ is not None and not df_.empty:
                for col in df_.columns:
                    df_[col] = df_[col].map(ExperimentHandler._make_excel_safe)
    
        exclude_cols = {"template_id", "row_idx", "cf_id"}
        metric_cols = [
            c for c in results_df.columns
            if results_df[c].dtype.kind in "biufc" and c not in exclude_cols
        ]
    
        summary_df = pd.DataFrame({
            "Metric": metric_cols,
            "Mean": [results_df[c].mean() for c in metric_cols],
            "Std": [results_df[c].std() for c in metric_cols],
            "Min": [results_df[c].min() for c in metric_cols],
            "Max": [results_df[c].max() for c in metric_cols],
        })
    
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)
            
            if cf_analysis_df is not None and not cf_analysis_df.empty:
                cf_analysis_df.to_excel(writer, sheet_name="CF_Fitness_Analysis", index=False)
                
            templates_df.to_excel(writer, sheet_name="Templates", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
        print(f"Saved workbook to: {output_path}")


    @staticmethod
    def _make_matched_conformant_df(
        df:                        pd.DataFrame,
        factual_trace_activities:  Tuple[str, ...],
        target_pattern:            Tuple[str, ...],
        start_pos:                 int,
        factual_trace_len:         int,
        case_id_field:             str,
        activity_field:            str,
        sort_field:                str,
        max_traces:                Optional[int] = None,
    ) -> pd.DataFrame:
    
        if start_pos + len(target_pattern) > factual_trace_len:
            return df.iloc[0:0].copy()
    
        factual_prefix = tuple(map(str, factual_trace_activities[:start_pos]))
        target_pattern = tuple(map(str, target_pattern))
    
        matched = []
    
        df_sorted = df.sort_values([case_id_field, sort_field])
    
        for _, group in df_sorted.groupby(case_id_field, sort=False):
    
            group = group.sort_values(sort_field).reset_index(drop=True)
    
            if len(group) < factual_trace_len:
                continue
    
            acts = tuple(group[activity_field].astype(str).tolist())
    
            # Same prefix before intervention
            if tuple(acts[:start_pos]) != factual_prefix:
                continue
    
            # Target behavior at intervention window
            if tuple(acts[start_pos:start_pos + len(target_pattern)]) != target_pattern:
                continue
    
            matched.append(group.iloc[:factual_trace_len].copy())
    
            if max_traces is not None and len(matched) >= max_traces:
                break
    
        if not matched:
            return df.iloc[0:0].copy()
    
        return pd.concat(matched, ignore_index=True)


    def run_experiment_df(
        self,
        cf_method,
        technique:              str,
        exp_df:                 pd.DataFrame,
        df:                     pd.DataFrame,
        feature_config:         FeatureConfig,
        case_id_field:          str,
        sort_field:             str,
        output_dir:             str = "./generated_experiments_results/",
        max_conformant_traces:  Optional[int] = None,
    ) -> pd.DataFrame:
    
        os.makedirs(output_dir, exist_ok=True)
    
        result_rows = []
        all_cf_analysis_rows = []
        global_row_counter = 0

        train_arr_cache = {}
    
        template_groups = list(
            exp_df.groupby(
                ["template_id", "trace_length"],
                sort=False,
            )
        )
    
        with tqdm(
            total=len(exp_df),
            desc="Experiments",
            unit="case",
            dynamic_ncols=True,
        ) as pbar:
    
            for (template_id, grouped_trace_length), template_df in template_groups:
    
                trace_length = int(grouped_trace_length)
    
                # Build training prefixes once per trace length/template group
                if trace_length not in train_arr_cache:
                    train_arr_cache[trace_length] = df_to_sequence_array(
                        df=df,
                        case_id_field=case_id_field,
                        sort_field=sort_field,
                        feature_config=feature_config,
                        prefix_len=trace_length,
                    )
        
                train_arr = train_arr_cache[trace_length]
    
                for _, exp in template_df.iterrows():
    
                    try:
                        case_id = exp[case_id_field]
    
                        trace_df = (
                            df[df[case_id_field] == case_id]
                            .sort_values(sort_field)
                            .iloc[:trace_length]
                            .copy()
                        )
    
                        if len(trace_df) < trace_length:
                            tqdm.write(
                                f"Skipping template {template_id}, "
                                f"case {case_id}: trace is shorter than "
                                f"{trace_length} events."
                            )
                            continue
    
                        trace_arr = df_to_sequence_array(
                            df=trace_df,
                            case_id_field=case_id_field,
                            sort_field=sort_field,
                            feature_config=feature_config,
                            prefix_len=trace_length,
                        )[0]
    
                        trace_activities = tuple(
                            trace_df[feature_config.activity_feature]
                            .astype(str)
                            .tolist()
                        )
    
                        conformant_df = self._make_matched_conformant_df(
                            df=df,
                            factual_trace_activities=trace_activities,
                            target_pattern=exp["target_pattern"],
                            start_pos=int(exp["start_pos"]),
                            factual_trace_len=trace_length,
                            case_id_field=case_id_field,
                            activity_field=feature_config.activity_feature,
                            sort_field=sort_field,
                            max_traces=max_conformant_traces,
                        )
    
                        if conformant_df.empty:
                            tqdm.write(
                                f"Skipping template {template_id}, "
                                f"case {case_id}, trace length {trace_length}: "
                                f"no conformant traces."
                            )
                            continue
    
                        conformant_arr = df_to_sequence_array(
                            df=conformant_df,
                            case_id_field=case_id_field,
                            sort_field=sort_field,
                            feature_config=feature_config,
                            prefix_len=trace_length,
                        )
    
                        desired_positions = exp["desired_positions"]
    
                        desired_constraints = (
                            self.constraint_engine.generate_desired_constraints(
                                trace_activities=trace_activities,
                                user_specs={
                                    self._occurrence_at_position(
                                        trace_activities,
                                        int(pos),
                                    ): allowed
                                    for pos, allowed
                                    in desired_positions.items()
                                },
                            )
                        )
    
                        desired_pos_set = set(
                            desired_constraints.get_constrained_positions()
                        )
    
                        if not desired_pos_set:
                            tqdm.write(
                                f"Skipping template {template_id}, "
                                f"case {case_id}: no desired positions were resolved."
                            )
                            continue
    
                        flexible_constraints = (
                            self.constraint_engine.generate_flexible_constraints(
                                trace_activities=trace_activities,
                            )
                        )
    
                        num_desired = len(desired_pos_set)
                        num_flexible = len(
                            flexible_constraints.get_constrained_positions()
                        )
    
                        last_desired_pos = max(desired_pos_set)
    
                        change_allowed_positions = [
                            pos
                            for pos in range(last_desired_pos + 1)
                            if pos not in desired_pos_set
                        ]
    
                        cf_output = cf_method.search(
                            trace=trace_arr,
                            train_arr=train_arr,
                            conformant_arr=conformant_arr,
                            change_allowed_positions=change_allowed_positions,
                            desired_constraints=desired_constraints,
                            flexible_constraints=flexible_constraints,
                        )
    
                        eval_result = cf_output.get("eval", {})
                        fitness_components = cf_output.get(
                            "fitness_components",
                            {},
                        )
    
                        result_row = {
                            "template_id": template_id,
                            case_id_field: case_id,
                            "trace_length": trace_length,
                            "num_desired": num_desired,
                            "num_flexible": num_flexible,
                        }
    
                        if isinstance(eval_result, dict):
                            result_row.update(eval_result)
    
                        if isinstance(fitness_components, dict):
                            component_row = {
                                f"best_cf_{component}": (
                                    float(values[0])
                                    if hasattr(values, "__len__")
                                    else float(values)
                                )
                                for component, values
                                in fitness_components.items()
                            }
                            result_row.update(component_row)
    
                        if (
                            isinstance(fitness_components, dict)
                            and fitness_components
                        ):
                            first_component = next(
                                iter(fitness_components.values())
                            )
    
                            num_individuals = (
                                len(first_component)
                                if hasattr(first_component, "__len__")
                                else 1
                            )
    
                            for cf_idx in range(num_individuals):
    
                                analysis_row = {
                                    "row_idx": global_row_counter,
                                    "template_id": template_id,
                                    case_id_field: case_id,
                                    "trace_length": trace_length,
                                    "num_desired": num_desired,
                                    "num_flexible": num_flexible,
                                }
    
                                for component, values in (
                                    fitness_components.items()
                                ):
                                    analysis_row[component] = (
                                        float(values[cf_idx])
                                        if hasattr(values, "__len__")
                                        else float(values)
                                    )
    
                                all_cf_analysis_rows.append(analysis_row)
                                global_row_counter += 1
    
                        result_rows.append(result_row)
    
                    finally:
                        # Runs even when the current case uses `continue`
                        pbar.update(1)
    
        results_df = pd.DataFrame(result_rows)
        cf_analysis_df = pd.DataFrame(all_cf_analysis_rows)
    
        ExperimentHandler._save_experiment_workbook(
            results_df=results_df,
            exp_df=exp_df,
            cf_analysis_df=cf_analysis_df,
            output_path=os.path.join(
                output_dir,
                f"{technique}_experiment_results.xlsx",
            ),
        )
    
        return results_df
