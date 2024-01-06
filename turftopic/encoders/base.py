from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class ExternalEncoder(ABC):
    @abstractmethod
    def encode(sentences: Iterable[str]) -> np.ndarray:
        pass
