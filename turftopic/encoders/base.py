from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np


class ExternalEncoder(ABC):
    """Base class for external encoder models."""

    @abstractmethod
    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        """Encodes sentences into an embedding matrix.

        Parameters
        ----------
        sentences: Iterable[str]
            Sentences to get embeddings for.

        Returns
        -------
        ndarray of shape (n_docs, n_dimensions)
            Embedding matrix.
        """
        pass
