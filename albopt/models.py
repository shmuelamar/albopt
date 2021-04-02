from dataclasses import dataclass, field

import numpy as np


@dataclass
class Image:
    name: str
    filename: str = field(repr=False)
    img: np.ndarray = field(repr=False)
    hidden_vector: np.array = field(default=None, repr=False)
    ratio: float = 0.0
