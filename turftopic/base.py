from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from turftopic.data import TopicData
from turftopic.encoders import ExternalEncoder


def remove_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


Encoder = Union[ExternalEncoder, SentenceTransformer]


class ContextualModel(ABC, TransformerMixin, BaseEstimator):
    """Base class for contextual topic models in Turftopic."""

    def get_topics(
        self, top_k: int = 10
    ) -> List[Tuple[Any, List[Tuple[str, float]]]]:
        """Returns high-level topic representations in form of the top K words
        in each topic.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.

        Returns
        -------
        list[tuple]
            List of topics. Each topic is a tuple of
            topic ID and the top k words.
            Top k words are a list of (word, word_importance) pairs.
        """
        n_topics = self.components_.shape[0]
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(n_topics))
        highest = np.argpartition(-self.components_, top_k)[:, :top_k]
        vocab = self.get_vocab()
        top = []
        score = []
        for component, high in zip(self.components_, highest):
            importance = component[high]
            high = high[np.argsort(-importance)]
            score.append(component[high])
            top.append(vocab[high])
        topics = []
        for topic, words, scores in zip(classes, top, score):
            topic_data = (topic, list(zip(words, scores)))
            topics.append(topic_data)
        return topics

    def print_topics(self, top_k: int = 10, show_scores: bool = False):
        """Pretty prints topics in the model in a table.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        """
        topics = self.get_topics(top_k)
        table = Table(show_lines=True)
        table.add_column("Topic ID", style="blue", justify="right")
        table.add_column(
            f"Top {top_k} Words",
            justify="left",
            style="magenta",
            max_width=100,
        )
        for topic_id, terms in topics:
            if show_scores:
                concat_words = ", ".join(
                    [f"{word}({importance:.2f})" for word, importance in terms]
                )
            else:
                concat_words = ", ".join([word for word, importance in terms])
            table.add_row(f"{topic_id}", f"{concat_words}")
        console = Console()
        console.print(table)

    def print_highest_ranking_documents(
        self, topic_id, raw_documents, document_topic_matrix=None, top_k=5
    ):
        """Pretty prints the highest ranking documents in a topic.

        Parameters
        ----------
        topic_id: int
            ID of the topic to display.
        raw_documents: list of str
            List of documents to consider.
        document_topic_matrix: ndarray of shape (n_topics, n_topics), optional
            Document topic matrix to use. This is useful for transductive methods,
            as they cannot infer topics from text.
        top_k: int, default 5
            Top K documents to show.
        """
        if document_topic_matrix is None:
            try:
                document_topic_matrix = self.transform(raw_documents)
            except AttributeError:
                raise ValueError(
                    "Transductive methods cannot "
                    "infer topical content in documents.\n"
                    "Please pass a document_topic_matrix."
                )
        try:
            topic_id = list(self.classes_).index(topic_id)
        except AttributeError:
            pass
        kth = min(top_k, document_topic_matrix.shape[0] - 1)
        highest = np.argpartition(-document_topic_matrix[:, topic_id], kth)[
            :kth
        ]
        highest = highest[
            np.argsort(-document_topic_matrix[highest, topic_id])
        ]
        scores = document_topic_matrix[highest, topic_id]
        table = Table(show_lines=True)
        table.add_column(
            "Document", justify="left", style="magenta", max_width=100
        )
        table.add_column("Score", style="blue", justify="right")
        for document_id, score in zip(highest, scores):
            doc = raw_documents[document_id]
            doc = remove_whitespace(doc)
            if len(doc) > 300:
                doc = doc[:300] + "..."
            table.add_row(doc, f"{score:.2f}")
        console = Console()
        console.print(table)

    def print_topic_distribution(
        self, text=None, topic_dist=None, top_k: int = 10
    ):
        """Pretty prints topic distribution in a document.

        Parameters
        ----------
        text: str, optional
            Text to infer topic distribution for.
        topic_dist: ndarray of shape (n_topics), optional
            Already inferred topic distribution for the text.
            This is useful for transductive methods,
            as they cannot infer topics from text.
        top_k: int, default 10
            Top K topics to show.
        """
        if topic_dist is None:
            if text is None:
                raise ValueError(
                    "You should either pass a text or a distribution."
                )
            try:
                topic_dist = self.transform([text])
            except AttributeError:
                raise ValueError(
                    "Transductive methods cannot "
                    "infer topical content in documents.\n"
                    "Please pass a topic distribution."
                )
        topic_dist = np.squeeze(np.asarray(topic_dist))
        topic_desc = self.get_topics(top_k=4)
        topic_names = []
        for topic_id, terms in topic_desc:
            concat_words = "_".join([word for word, importance in terms])
            topic_names.append(f"{topic_id}_{concat_words}")
        highest = np.argsort(-topic_dist)[:top_k]
        table = Table()
        table.add_column("Topic name", justify="left", style="magenta")
        table.add_column("Score", justify="right", style="blue")
        for ind in highest:
            score = topic_dist[ind]
            table.add_row(topic_names[ind], f"{score:.2f}")
        console = Console()
        console.print(table)

    def encode_documents(self, raw_documents: Iterable[str]) -> np.ndarray:
        """Encodes documents with the sentence encoder of the topic model.

        Parameters
        ----------
        raw_documents: iterable of str
            Textual documents to encode.

        Return
        ------
        ndarray of shape (n_documents, n_dimensions)
            Matrix of document embeddings.
        """
        return self.encoder_.encode(raw_documents)

    @abstractmethod
    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fits model and infers topic importances for each document.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        y: None
            Ignored, exists for sklearn compatibility.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.

        Returns
        -------
        ndarray of shape (n_documents, n_topics)
            Document-topic matrix.
        """
        pass

    def fit(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        """Fits model on the given corpus.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        y: None
            Ignored, exists for sklearn compatibility.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.
        """
        self.fit_transform(raw_documents, y, embeddings)
        return self

    def get_vocab(self) -> np.ndarray:
        """Get vocabulary of the model.

        Returns
        -------
        ndarray of shape (n_vocab)
            All terms in the vocabulary.
        """
        return self.vectorizer.get_feature_names_out()

    def get_feature_names_out(self) -> np.ndarray:
        """Get topic ids.

        Returns
        -------
        ndarray of shape (n_topics)
            IDs for each output feature of the model.
            This is useful, since some models have outlier
            detection, and this gets -1 as ID, instead of
            its index.
        """
        n_topics = self.components_.shape[0]
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(n_topics))
        return np.asarray(classes)

    def prepare_topic_data(
        self,
        corpus: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> TopicData:
        """Produces topic inference data for a given corpus, that can be then used and reused.
        Exists to allow visualizations out of the box with topicwizard.

        Parameters
        ----------
        corpus: list of str
            Documents to infer topical content for.
        embeddings: ndarray of shape (n_documents, n_dimensions)
            Embeddings of documents.

        Returns
        -------
        TopicData
            Information about topical inference in a dictionary.
        """
        if embeddings is None:
            embeddings = self.encode_documents(corpus)
        try:
            document_topic_matrix = self.transform(corpus, embeddings=embeddings)
        except AttributeError, NotFittedError:
            document_topic_matrix = self.fit_transform(corpus, embeddings=embeddings)
        dtm = self.vectorizer.transform(corpus) # type: ignore
        res: TopicData = {
            "corpus": corpus,
            "document_term_matrix": dtm,
            "vocab": self.get_vocab(),
            "document_topic_matrix": document_topic_matrix,
            "document_representation": embeddings,
            "topic_term_matrix": self.components_, # type: ignore
            "transform": getattr(self, "transform", None),
        }
        return res
