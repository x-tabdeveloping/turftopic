from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import joblib
import numpy as np
from PIL import Image
from rich.console import Console
from rich.tree import Tree

from turftopic.container import TopicContainer

if TYPE_CHECKING:
    from turftopic.hierarchical import TopicNode


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
    topic_names: list of str, default None
        Names or topic descriptions inferred for topics by the model.
    classes: np.ndarray, default None
        Topic IDs that might be different from 0-n_topics.
        (For instance if you have an outlier topic, which is labelled -1)
    corpus: list of str, default None
        The corpus on which inference was run. Can be None.
    transform: (list[str]) -> ndarray, default None
        Function that transforms documents to document-topic matrices.
        Can be None in the case of transductive models.
    time_bin_edges: list[datetime], default None
        Edges of the time bins in a dynamic topic model.
    temporal_components: np.ndarray (n_slices, n_topics, n_vocab), default None
        Topic-term importances over time. Only relevant for dynamic topic models.
    temporal_importance: np.ndarray (n_slices, n_topics), default None
        Topic strength signal over time. Only relevant for dynamic topic models.
    has_negative_side: bool, default False
        Indicates whether the topic model's components are supposed to be interpreted in both directions.
        e.g. in SemanticSignalSeparation, one is supposed to look at highest, but also lowest ranking words.
        This is in contrast to KeyNMF for instance, where only positive word importance should be considered.
    hierarchy: TopicNode, default None
        Optional topic hierarchy for models that support hierarchical topic modeling.
    images: list[ImageRepr], default None
        Images the model has been fit on
    top_images: list[list[Image]], default None
        Top images discovered by the topic model.
    negative_images: list[list[Image]], default None
        Lowest ranking images discivered by the topic model.
        (Only relevant with models like S^3)
    """

    def __init__(
        self,
        *,
        vocab: np.ndarray,
        document_term_matrix: np.ndarray,
        document_topic_matrix: np.ndarray,
        topic_term_matrix: np.ndarray,
        document_representation: np.ndarray,
        topic_names: Optional[list[str]] = None,
        classes: Optional[np.ndarray] = None,
        corpus: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        time_bin_edges: Optional[list[datetime]] = None,
        temporal_components: Optional[np.ndarray] = None,
        temporal_importance: Optional[np.ndarray] = None,
        has_negative_side: bool = False,
        hierarchy: Optional[TopicNode] = None,
        images: Optional[list[str | Image.Image]] = None,
        top_images: Optional[list[list[Image.Image]]] = None,
        negative_images: Optional[list[list[Image.Image]]] = None,
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
        self.time_bin_edges = time_bin_edges
        self.temporal_components = temporal_components
        self.temporal_importance = temporal_importance
        self.hierarchy = hierarchy
        self._has_negative_side = has_negative_side
        self.top_images = top_images
        self.negative_images = negative_images
        self.images = images
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
            "time_bin_edges",
            "temporal_components",
            "temporal_importance",
            "has_negative_side",
            "hierarchy",
            *kwargs.keys(),
        ]

    @property
    def components_(self) -> np.ndarray:
        return self.topic_term_matrix

    @property
    def temporal_components_(self) -> np.ndarray:
        if self.temporal_components is None:
            raise AttributeError(
                "Topic data does not contain dynamic information."
            )
        return self.temporal_components

    @property
    def temporal_importance_(self) -> np.ndarray:
        if self.temporal_importance is None:
            raise AttributeError(
                "Topic data does not contain dynamic information."
            )
        return self.temporal_importance

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

    def __str__(self):
        console = Console()
        with console.capture() as capture:
            tree = Tree("TopicData")
            for key, value in self.items():
                if value is None:
                    continue
                if hasattr(value, "shape"):
                    text = f"{key} {value.shape}"
                elif hasattr(value, "__len__"):
                    text = f"{key} ({len(value)})"
                else:
                    text = key
                tree.add(text)
            console.print(tree)
        return capture.get()

    def __repr__(self):
        return str(self)

    def visualize_topicwizard(self, **kwargs):
        """Opens the topicwizard web app with which you can interactively investigate your model.
        See [topicwizard's documentation](https://github.com/x-tabdeveloping/topicwizard) for more detail.
        """
        try:
            import topicwizard
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "topicwizard is not installed on your system, you can install it by running pip install turftopic[topic-wizard]."
            )
        return topicwizard.visualize(topic_data=self, **kwargs)

    @property
    def figures(self):
        """Container object for topicwizard figures that can be generated from this TopicData object.
        You can use any of the interactive figures from the [Figures API](https://x-tabdeveloping.github.io/topicwizard/figures.html) in topicwizard.

        For instance:
        ```python
        topic_data.figures.topic_barcharts()
        # or
        topic_data.figures.topic_wordclouds()
        ```
        See [topicwizard's documentation](https://github.com/x-tabdeveloping/topicwizard) for more detail.
        """
        try:
            import topicwizard.figures
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "topicwizard is not installed on your system, you can install it by running pip install turftopic[topic-wizard]."
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

    @property
    def has_negative_side(self) -> bool:
        return self._has_negative_side
