from typing import Optional, List, TypedDict, Tuple, Literal
from numpy import ndarray

LinearForwardCache = Tuple[ndarray, ndarray, ndarray]
ActivationCache = ndarray
Cache = Tuple[LinearForwardCache, ActivationCache]
Activation = Literal["sigmoid", "relu"]