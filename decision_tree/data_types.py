from typing import TypedDict, Optional
from numpy import ndarray

class BestSplit(TypedDict):
    feature: Optional[int]
    threshold: Optional[float]
    left_dataset: Optional[ndarray]
    right_dataset: Optional[ndarray]
    gain: float