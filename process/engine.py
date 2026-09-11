import os
import json

from dataclasses import dataclass, field

from itertools import product

from typing import Set, List, Dict, Tuple, Optional

from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.petri_net.obj import PetriNet, Marking

from process.constraint import ProcessConstraint



@dataclass(frozen=True)
class ProcessModelConstraintEngine:

    petri_net:          Optional[PetriNet] = None
    initial_marking:    Optional[Marking] = None
    final_marking:      Optional[Marking] = None
    process_tree:       Optional[ProcessTree] = None

    min_segment_len:    int = 2

    parallel_sets:      List[Set[str]] = field(default_factory=list, init=False)
    branching_sets:     List[Set[str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        tree = self.process_tree

        if tree is None and self.petri_net is not None:
            tree = self._convert_petri_net_to_process_tree()

        if tree is None:
            object.__setattr__(self, "parallel_sets", [])
            object.__setattr__(self, "branching_sets", [])
            return

        parallel_sets, branching_sets = self._extract_constraints_from_process_tree(tree)

        object.__setattr__(self, "process_tree", tree)
        object.__setattr__(self, "parallel_sets", parallel_sets)
        object.__setattr__(self, "branching_sets", branching_sets)
        

    def _convert_petri_net_to_process_tree(self) -> ProcessTree:
        if self.initial_marking is None or self.final_marking is None:
            raise ValueError(
                "Petri net to process tree conversion requires initial_marking and final_marking."
            )

        try:
            from pm4py.objects.conversion.wf_net import converter as wf_net_converter

            return wf_net_converter.apply(
                self.petri_net,
                self.initial_marking,
                self.final_marking,
            )

        except Exception as e:
            raise ValueError(
                "Could not convert Petri net to process tree. "
                "This usually means the Petri net is not block-structured. "
                "Prefer passing the original process tree from Inductive Miner."
            ) from e

    
    def _leaf_labels(self, node: ProcessTree) -> Set[str]:
        labels = set()

        if node.label is not None:
            labels.add(str(node.label))

        for child in node.children:
            labels.update(self._leaf_labels(child))

        return labels

    
    def _extract_constraints_from_process_tree(
        self,
        tree: ProcessTree,
    ) -> Tuple[List[Set[str]], List[Set[str]]]:

        parallel_sets = []
        branching_sets = []

        def visit(node: ProcessTree):

            if node.operator == Operator.PARALLEL:
                acts = self._leaf_labels(node)
                if len(acts) > 1:
                    parallel_sets.append(acts)

            elif node.operator == Operator.XOR:
                acts = self._leaf_labels(node)
                if len(acts) > 1:
                    branching_sets.append(acts)

            for child in node.children:
                visit(child)

        visit(tree)

        return parallel_sets, branching_sets

    
    def save(self, path: str = "./pretrained_models/") -> None:
        """
        Save extracted process-model constraints.
        """
    
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, "process_model_constraints.json")
    
        data = {
            "min_segment_len": self.min_segment_len,
            "parallel_sets": [sorted(list(s)) for s in self.parallel_sets],
            "branching_sets": [sorted(list(s)) for s in self.branching_sets],
        }
    
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


    @classmethod
    def load(
        cls, path: str = "./pretrained_models/",
    ) -> "ProcessModelConstraintEngine":
    
        filepath = os.path.join(path, "process_model_constraints.json")
    
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        engine = cls(
            petri_net=None,
            initial_marking=None,
            final_marking=None,
            process_tree=None,
            min_segment_len=data["min_segment_len"],
        )
    
        object.__setattr__(
            engine,
            "parallel_sets",
            [set(s) for s in data["parallel_sets"]],
        )
    
        object.__setattr__(
            engine,
            "branching_sets",
            [set(s) for s in data["branching_sets"]],
        )
    
        return engine


    # --- Extract constraints ---
    def generate_flexible_constraints(
        self,
        trace_activities: Tuple[str, ...],
    ) -> ProcessConstraint:
        
        """Generate FlexibleConstraint from process-tree parallel blocks.

        A segment is flexible if consecutive activities in the trace belong
        to the same extracted parallel activity set.
        """
        
        flexible_constraints = ProcessConstraint(segments={})
        n = len(trace_activities)

        for allowed in self.parallel_sets:
            start: Optional[int] = None

            for i in range(n):
                act = trace_activities[i]

                if act in allowed:
                    if start is None:
                        start = i
                else:
                    if start is not None:
                        end = i - 1
                        if (end - start + 1) >= self.min_segment_len:
                            segment_activities = list(trace_activities[start : end + 1])
                    
                            try:
                                flexible_constraints.insert(start, end, segment_activities)
                            except ValueError:
                                # Overlapping parallel sets on identical boundaries fallback:
                                # Force append by merging or bypass to keep processing intact
                                pass
                                
                        start = None

            if start is not None:
                end = n - 1
                if (end - start + 1) >= self.min_segment_len:
                    segment_activities = list(trace_activities[start : end + 1])
                    
                    try:
                        flexible_constraints.insert(start, end, segment_activities)
                    except ValueError:
                        pass

        return flexible_constraints


    @staticmethod
    def _find_occurrence_index(
        trace_activities:   Tuple[str, ...], 
        target_act:         str, 
        occurrence:         int
    ) -> Optional[int]:
        
        """Locates the absolute index position of a specific activity occurrence (1-based)."""
        
        count = 0
        for idx, act in enumerate(trace_activities):
            if act == target_act:
                count += 1
                if count == occurrence:
                    return idx
        return None


    def generate_desired_constraints(
        self,
        trace_activities: Tuple[str, ...],
        user_specs: Dict[Tuple[str, int], Set[str]],
    ) -> ProcessConstraint:
        
        """Generates desired constraint tracking matrices mapping to user instructions.
        
        Natively identifies strictly contiguous user rule indices first, then
        validates the combinatorial flexibility of only those touching blocks.
        """

        base_trace = list(trace_activities)
    
        # Resolve user specifications to precise absolute trace positions
        resolved = []
        for (target_act, occurrence), desired_acts in user_specs.items():
            pos = self._find_occurrence_index(
                trace_activities=tuple(base_trace),
                target_act=target_act,
                occurrence=occurrence,
            )
            if pos is not None:
                resolved.append((pos, desired_acts))
    
        if not resolved:
            return ProcessConstraint()
    
        resolved.sort(key=lambda x: x[0])
        
        # Segment the rules into separate contiguous groups
        contiguous_groups: List[List[Tuple[int, Set[str]]]] = []
        current_group = [resolved[0]]
        
        for item in resolved[1:]:
            prev_pos = current_group[-1][0]
            curr_pos = item[0]
            
            if curr_pos == prev_pos + 1:
                current_group.append(item)
            else:
                contiguous_groups.append(current_group)
                current_group = [item]
        contiguous_groups.append(current_group)
        
        # Maps (start, end) -> Set of unique valid activities across all paths
        aggregated_segments: Dict[Tuple[int, int], Set[str]] = {}
        
        # Process each contiguous group independently
        for group in contiguous_groups:
            start = group[0][0]
            end = group[-1][0]
            
            # Isolated single rule that has no touching neighbors -> Standalone staging
            if len(group) == 1:
                key = (start, end)
                if key not in aggregated_segments:
                    aggregated_segments[key] = set()
                for act in group[0][1]:
                    aggregated_segments[key].add(act)
                continue
                
            # Multiple back-to-back user rules exist -> Evaluate their combination flexibility
            group_options = [sorted(list(acts)) for _, acts in group]
            
            for combo in product(*group_options):
                is_fully_flexible = True
                
                for idx in range(len(combo) - 1):
                    current_act = combo[idx]
                    next_act = combo[idx + 1]
                    
                    link_flexible = False
                    for allowed_parallel_set in self.parallel_sets:
                        if current_act in allowed_parallel_set and next_act in allowed_parallel_set:
                            link_flexible = True
                            break
                            
                    if not link_flexible:
                        is_fully_flexible = False
                        break
                        
                # Path A: The chosen combination is perfectly parallel across the whole chain
                if is_fully_flexible:
                    key = (start, end)
                    if key not in aggregated_segments:
                        aggregated_segments[key] = set()
                    for act in combo:
                        aggregated_segments[key].add(act)
                        
                # Path B: The chosen combination breaks parallel rules -> Unpack to unique standalone coordinates
                else:
                    for item_idx, pos_act in enumerate(combo):
                        item_pos = group[item_idx][0]
                        item_key = (item_pos, item_pos)
                        
                        if item_key not in aggregated_segments:
                            aggregated_segments[item_key] = set()
                        aggregated_segments[item_key].add(pos_act)
                            
        master_desired_constraints = ProcessConstraint(segments={})
        for (s, e), acts_set in aggregated_segments.items():
            try:
                master_desired_constraints.insert(s, e, list(acts_set))
            except ValueError:
                pass
                
        return master_desired_constraints
