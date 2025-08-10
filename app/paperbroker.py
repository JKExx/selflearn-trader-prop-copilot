from dataclasses import dataclass

@dataclass
class Position:
    side: int = 0  # -1 short, 0 flat, 1 long
    entry: float = 0.0
    size: float = 0.0
