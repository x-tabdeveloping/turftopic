from typing import Callable, Optional, TypedDict

import numpy as np


class TopicData(TypedDict):
    corpus: list[str]
    vocab: np.ndarray
    document_term_matrix: np.ndarray
    document_topic_matrix: np.ndarray
    topic_term_matrix: np.ndarray
    document_representation: np.ndarray
    transform: Optional[Callable]
    topic_names: list[str]
