from enum import Enum
from dataclasses import dataclass

class SemaphorState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

@dataclass
class Semaphor:
    state: SemaphorState
    timer: int  # seconds
