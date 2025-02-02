from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np

from turftopic.container import TopicContainer


@dataclass
class Figures:
    figure_names: list[str]


class TopicData(Mapping, TopicContainer):
    """Contains data about topic inference on a corpus.
    Can be used with multiple convenience and interpretation utilities.

    Parameters
    ----------
    vocab: ndarray of shape (n_vocab,)
        Array of all words in the vocabulary of the topic model.
    document_term_matrix: ndarray of shape (n_documents, n_vocab)
        Bag-of-words document representations.
        Elements of the matrix are word importances/frequencies for given documents.
    document_topic_matrix: ndarray of shape (n_documents, n_topics)
        Topic importances for each document.
    topic_term_matrix: ndarray of shape (n_topics, n_vocab)
        Importances of each term for each topic in a matrix.
    document_representation: ndarray of shape (n_documents, n_dimensions)
        Embedded representations for documents.
        Can also be a sparse BoW matrix for classical models.
    topic_names: list of str
        Names or topic descriptions inferred for topics by the model.
    classes: np.ndarray, default None
        Topic IDs that might be different from 0-n_topics.
        (For instance if you have an outlier topic, which is labelled -1)
    corpus: list of str, default None
        The corpus on which inference was run. Can be None.
    transform: (list[str]) -> ndarray, default None
        Function that transforms documents to document-topic matrices.
        Can be None in the case of transductive models.
    """

    def __init__(
        self,
        *,
        vocab: np.ndarray,
        document_term_matrix: np.ndarray,
        document_topic_matrix: np.ndarray,
        topic_term_matrix: np.ndarray,
        document_representation: np.ndarray,
        topic_names: list[str],
        classes: Optional[np.ndarray] = None,
        corpus: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        self.corpus = corpus
        self.vocab = vocab
        self.document_term_matrix = document_term_matrix
        self.document_topic_matrix = document_topic_matrix
        self.topic_term_matrix = topic_term_matrix
        self.document_representation = document_representation
        self.transform = transform
        self.topic_names_ = topic_names
        self.classes = classes
        for key, value in kwargs:
            setattr(self, key, value)
        self._attributes = [
            "corpus",
            "vocab",
            "document_term_matrix",
            "topic_term_matrix",
            "document_topic_matrix",
            "document_representation",
            "transform",
            "topic_names",
            *kwargs.keys(),
        ]

    @property
    def components_(self) -> np.ndarray:
        return self.topic_term_matrix

    @property
    def classes_(self) -> np.ndarray:
        if self.classes is None:
            raise AttributeError("Topic model does not have classes_")
        else:
            return self.classes

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, newvalue):
        return setattr(self, key, newvalue)

    def __len__(self):
        return len(self._attributes)

    def __iter__(self):
        return iter(self._attributes)

    def get_vocab(self) -> np.ndarray:
        return self.vocab

    def visualize(self, **kwargs):
        try:
            import topicwizard
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "topicwizard is not installed on your system, you can install it by running pip install topic-wizard."
            )
        return topicwizard.visualize(topic_data=self, **kwargs)

    @property
    def figures(self):
        try:
            import topicwizard.figures
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "topicwizard is not installed on your system, you can install it by running pip install topic-wizard."
            )

        # Skip Group figures
        figure_names = [
            figure_name
            for figure_name in topicwizard.figures.__all__
            if not figure_name.startswith("group")
        ]
        module = Figures(figure_names)
        for figure_name in figure_names:
            figure_fn = getattr(topicwizard.figures, figure_name)
            figure_fn = partial(figure_fn, topic_data=self)
            setattr(
                module,
                figure_name,
                figure_fn,
            )
        return module

    @classmethod
    def from_disk(cls, path: str | Path):
        """Loads TopicData object from disk with Joblib.

        Parameters
        ----------
        path: str or Path
            Path to load the data from, e.g. "topic_data.joblib"
        """
        path = Path(path)
        data = joblib.load(path)
        return cls(**data)

    def to_disk(self, path: str | Path):
        """Saves TopicData object to disk.

        Parameters
        ----------
        path: str or Path
            Path to save the data to, e.g. "topic_data.joblib"
        """
        path = Path(path)
        joblib.dump({**self}, path)
