from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.base import BaseEstimator, TransformerMixin


def remove_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


class ContextualModel(ABC, TransformerMixin, BaseEstimator):
    def get_topics(self, top_k: int = 10) -> List[Tuple]:
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
        return self.encoder_.encode(raw_documents)

    @abstractmethod
    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pass

    def fit(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        self.fit_transform(raw_documents, y, embeddings)
        return self

    def get_vocab(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()
