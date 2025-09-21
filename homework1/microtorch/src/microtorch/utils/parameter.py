from dataclasses import dataclass

import numpy as np


@dataclass
class Parameter:
    name  : str
    value : np.ndarray
    grad  : np.ndarray
