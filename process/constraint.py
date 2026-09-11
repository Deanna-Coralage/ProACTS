from dataclasses import dataclass, field

from typing import Set, List, Dict, Tuple, Optional



@dataclass
class ProcessConstraint:
    
    """
    Maps a positional span (start, end) to a list of allowed parallel activities.
    Partially overlapping segments (e.g., (2,5) and (4,7)) can co-exist, but 
    identical boundaries with differing content will raise a ValueError.
    """
    
    segments: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)

    def __post_init__(self):
        self.segments = {k: list(v) for k, v in self.segments.items()}
        self._validate_and_sort()

    def _validate_and_sort(self) -> None:
        if not self.segments:
            return

        sorted_keys = sorted(self.segments.keys())
        sorted_segments = {k: self.segments[k] for k in sorted_keys}

        object.__setattr__(self, "segments", sorted_segments)
        

    def insert(
        self, start: int, end: int, allowed: List[str], new_segment: bool = False
    ) -> None:
            
        """Dynamically inserts a new parallel segment into the constraint tracking pool.

        Extends an existing segment if boundaries match AND the incoming content
        is included (subset/superset relation) within the existing pool, unless
        new_segment is True.

        If new_segment is True or content does not match, it spawns an
        independent overlapping tracking window.
        """
        
        if start > end:
            raise ValueError(
                f"Invalid bounds: start index ({start}) cannot be greater than end index ({end})."
            )

        allowed_list = list(allowed)
        new_key = (start, end)

        if new_key in self.segments:
            if new_segment:
                raise ValueError(
                    f"Duplicate boundary error: Cannot force a new segment at {new_key}. "
                    f"This window is already registered in the constraint engine pool."
                )

            incoming_set = set(allowed_list)
            existing_set = set(self.segments[new_key])

            if incoming_set.issubset(existing_set) or existing_set.issubset(incoming_set):
                existing = self.segments[new_key].copy()
            
                for act in allowed_list:
                    if act in existing:
                        existing.remove(act)
                    else:
                        self.segments[new_key].append(act)
            
                return
            else:
                raise ValueError(
                    f"Content mismatch error: A segment at boundary {new_key} already exists, "
                    f"but the incoming activities do not satisfy a subset/superset relationship "
                    f"with the existing pool, and duplicate keys are forbidden."
                )

        self.segments[new_key] = allowed_list
        self._validate_and_sort()

    
    def get_constrained_positions(self) -> List[int]:
        
        indices: Set[int] = set()
        for start, end in self.segments.keys():
            for pos in range(start, end + 1):
                indices.add(pos)
        return sorted(indices)

    
    def __repr__(self) -> str:
        if not self.segments:
            return "No Constraints"

        lines = []
        for (start, end), allowed in self.segments.items():
            acts = "[" + ", ".join(sorted(allowed)) + "]"
            lines.append(f"{start}:{end} - {acts}")
        return ", ".join(lines)

    def __str__(self) -> str:
        return self.__repr__()
        

    def check_constrained_position(
        self, current_pos: int, activity: str
    ) -> Tuple[bool, bool, Optional[Tuple[int, int]]]:
        
        """Evaluates an activity against a given step index by scanning all
        overlapping segments that cover the current position.
        """
        
        is_constrained_zone = False

        for key in list(self.segments.keys()):
            start, end = key

            # Clean up dead windows as the chronological timeline moves past them
            if current_pos > end:
                del self.segments[key]
                continue

            if start <= current_pos <= end:
                is_constrained_zone = True
                allowed = self.segments[key]

                if activity in allowed:
                    allowed.remove(activity)

                    if not allowed:
                        del self.segments[key]

                    return True, True, (start, end)    # (is_constrained=True, constraint_ok=True)
    
        return is_constrained_zone, False, None
