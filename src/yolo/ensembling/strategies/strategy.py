from enum import Enum


class EnsembleStrategy(Enum):
    FUSION = 1
    AVERAGE = 2
    MAJORITY_VOTING = 3
