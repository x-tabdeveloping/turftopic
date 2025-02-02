from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from rich.console import Console
from rich.table import Table

from turftopic.namers.base import TopicNamer
from turftopic.utils import export_table


def remove_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


@dataclass
class Figures:
    figure_names: list[str]


class TopicData(Mapping):
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
        self.topic_names = topic_names
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

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, newvalue):
        return setattr(self, key, newvalue)

    def __len__(self):
        return len(self._attributes)

    def __iter__(self):
        return iter(self._attributes)

    @property
    def has_negative_side(self) -> bool:
        return np.any(self.components < 0)

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
        top = []
        score = []
        for component, high in zip(self.components_, highest):
            importance = component[high]
            high = high[np.argsort(-importance)]
            score.append(component[high])
            top.append(self.vocab[high])
        topics = []
        for topic, words, scores in zip(classes, top, score):
            topic_data = (topic, list(zip(words, scores)))
            topics.append(topic_data)
        return topics

    def _top_terms(
        self, top_k: int = 10, positive: bool = True
    ) -> list[list[str]]:
        terms = []
        for component in self.components_:
            lowest = np.argpartition(component, top_k)[:top_k]
            lowest = lowest[np.argsort(component[lowest])]
            highest = np.argpartition(-component, top_k)[:top_k]
            highest = highest[np.argsort(-component[highest])]
            if not positive:
                terms.append(list(self.vocab[lowest]))
            else:
                terms.append(list(self.vocab[highest]))
        return terms

    def _rename_automatic(self, namer: TopicNamer) -> list[str]:
        self.topic_names = namer.name_topics(self._top_terms())
        return self.topic_names

    def _topics_table(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: Optional[bool] = None,
    ) -> list[list[str]]:
        if show_negative is None:
            show_negative = self.has_negative_side
        columns = ["Topic ID"]
        if getattr(self, "topic_names_", None):
            columns.append("Topic Name")
        columns.append("Highest Ranking")
        if show_negative:
            columns.append("Lowest Ranking")
        rows = []
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(self.components_.shape[0]))
        for i_topic, (topic_id, component) in enumerate(
            zip(classes, self.components_)
        ):
            highest = np.argpartition(-component, top_k)[:top_k]
            highest = highest[np.argsort(-component[highest])]
            lowest = np.argpartition(component, top_k)[:top_k]
            lowest = lowest[np.argsort(component[lowest])]
            if show_scores:
                concat_positive = ", ".join(
                    [
                        f"{word}({importance:.2f})"
                        for word, importance in zip(
                            self.vocab[highest], component[highest]
                        )
                    ]
                )
                concat_negative = ", ".join(
                    [
                        f"{word}({importance:.2f})"
                        for word, importance in zip(
                            self.vocab[lowest], component[lowest]
                        )
                    ]
                )
            else:
                concat_positive = ", ".join(
                    [word for word in self.vocab[highest]]
                )
                concat_negative = ", ".join(
                    [word for word in self.vocab[lowest]]
                )
            row = [f"{topic_id}"]
            if getattr(self, "topic_names_", None):
                row.append(self.topic_names_[i_topic])
            row.append(f"{concat_positive}")
            if show_negative:
                row.append(concat_negative)
            rows.append(row)
        return [columns, *rows]

    def print_topics(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: Optional[bool] = None,
    ):
        """Pretty prints topics in the model in a table.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        show_negative: bool, default None
            Indicates whether the most negative terms should also be displayed.
            By default this gets inferred from the model parameters.
        """
        columns, *rows = self._topics_table(top_k, show_scores, show_negative)
        table = Table(show_lines=True)
        for column in columns:
            if column == "Highest Ranking":
                table.add_column(
                    column, justify="left", style="magenta", max_width=100
                )
            elif column == "Lowest Ranking":
                table.add_column(
                    column, justify="left", style="red", max_width=100
                )
            elif column == "Topic ID":
                table.add_column(column, style="blue", justify="right")
            else:
                table.add_column(column)
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)

    def export_topics(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: Optional[bool] = None,
        format: str = "csv",
    ) -> str:
        """Exports top K words from topics in a table in a given format.
        Returns table as a pure string.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        show_negative: bool, default None
            Indicates whether the most negative terms should also be displayed.
            By default this gets inferred from the models' parameters.
        format: 'csv', 'latex' or 'markdown'
            Specifies which format should be used.
            'csv', 'latex' and 'markdown' are supported.
        """
        table = self._topics_table(
            top_k, show_scores, show_negative=show_negative
        )
        return export_table(table, format=format)

    def _representative_docs(
        self,
        topic_id,
        top_k=5,
        show_negative: Optional[bool] = None,
    ) -> list[list[str]]:
        if show_negative is None:
            show_negative = self.has_negative_side
        if self.classes is not None:
            topic_id = list(self.classes).index(topic_id)
        if self.corpus is None:
            raise TypeError(
                "TopicData object does not contain the corpus, can't print representative documents."
            )
        kth = min(top_k, self.document_topic_matrix.shape[0] - 1)
        highest = np.argpartition(
            -self.document_topic_matrix[:, topic_id], kth
        )[:kth]
        highest = highest[
            np.argsort(-self.document_topic_matrix[highest, topic_id])
        ]
        scores = self.document_topic_matrix[highest, topic_id]
        columns = []
        columns.append("Document")
        columns.append("Score")
        rows = []
        for document_id, score in zip(highest, scores):
            doc = self.corpus[document_id]
            doc = remove_whitespace(doc)
            if len(doc) > 300:
                doc = doc[:300] + "..."
            rows.append([doc, f"{score:.2f}"])
        if show_negative:
            rows.append(["...", ""])
            lowest = np.argpartition(
                self.document_topic_matrix[:, topic_id], kth
            )[:kth]
            lowest = lowest[
                np.argsort(self.document_topic_matrix[lowest, topic_id])
            ]
            lowest = lowest[::-1]
            scores = self.document_topic_matrix[lowest, topic_id]
            for document_id, score in zip(lowest, scores):
                doc = self.raw_documents[document_id]
                doc = remove_whitespace(doc)
                if len(doc) > 300:
                    doc = doc[:300] + "..."
                rows.append([doc, f"{score:.2f}"])
        return [columns, *rows]

    def print_representative_documents(
        self,
        topic_id,
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: bool = None,
    ):
        """Pretty prints the highest ranking documents in a topic.

        Parameters
        ----------
        topic_id: int
            ID of the topic to display.
        raw_documents: list of str
            List of documents to consider.
        document_topic_matrix: ndarray of shape (n_documents, n_topics), optional
            Document topic matrix to use. This is useful for transductive methods,
            as they cannot infer topics from text.
        top_k: int, default 5
            Top K documents to show.
        show_negative: bool, default None
            Indicates whether lowest ranking documents should also be shown.
            By default this gets inferred from model parameters.
        """
        columns, *rows = self._representative_docs(
            topic_id,
            raw_documents,
            document_topic_matrix,
            top_k,
            show_negative,
        )
        table = Table(show_lines=True)
        table.add_column(
            "Document", justify="left", style="magenta", max_width=100
        )
        table.add_column("Score", style="blue", justify="right")
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)

    def export_representative_documents(
        self,
        topic_id,
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: Optional[bool] = None,
        format: str = "csv",
    ):
        """Exports the highest ranking documents in a topic as a text table.

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
        show_negative: bool, default False
            Indicates whether lowest ranking documents should also be shown.
        format: 'csv', 'latex' or 'markdown'
            Specifies which format should be used.
            'csv', 'latex' and 'markdown' are supported.
        """
        table = self._highest_ranking_docs(
            topic_id,
            raw_documents,
            document_topic_matrix,
            top_k,
            show_negative,
        )
        return export_table(table, format=format)

    def rename_topics(
        self, names: Union[list[str], dict[int, str], TopicNamer]
    ) -> None:
        """Rename topics in a model manually or automatically, using a namer.

        Examples:
        ```python
        model.rename_topics(["Automobiles", "Telephones"])
        # Or:
        model.rename_topics({-1: "Outliers", 2: "Christianity"})
        # Or:
        namer = OpenAITopicNamer()
        model.rename_topics(namer)
        ```

        Parameters
        ----------
        names: list[str] or dict[int,str]
            Should be a list of topic names, or a mapping of topic IDs to names.
        """
        if isinstance(names, TopicNamer):
            self._rename_automatic(names)
        elif isinstance(names, dict):
            topic_names = self.topic_names
            for topic_id, topic_name in names.items():
                try:
                    topic_id = list(self.classes_).index(topic_id)
                except AttributeError:
                    pass
                topic_names[topic_id] = topic_name
            self.topic_names_ = topic_names
        else:
            names = list(names)
            n_given = len(names)
            n_topics = self.components_.shape[0]
            if n_topics != n_given:
                raise ValueError(
                    f"Number of topics ({n_topics}) doesn't match the length of the given topic name list ({n_given})."
                )
            self.topic_names_ = names

    def _topic_distribution(
        self, text=None, top_k: int = 10
    ) -> list[list[str]]:
        try:
            topic_dist = self.transform([text])
        except (AttributeError, TypeError):
            raise ValueError(
                "Transductive methods cannot "
                "infer topical content in documents."
            )
        topic_dist = np.squeeze(np.asarray(topic_dist))
        topic_desc = self.get_topics(top_k=4)
        topic_names = []
        for topic_id, terms in topic_desc:
            concat_words = "_".join([word for word, importance in terms])
            topic_names.append(f"{topic_id}_{concat_words}")
        highest = np.argsort(-topic_dist)[:top_k]
        columns = []
        columns.append("Topic name")
        columns.append("Score")
        rows = []
        for ind in highest:
            score = topic_dist[ind]
            rows.append([topic_names[ind], f"{score:.2f}"])
        return [columns, *rows]

    def print_topic_distribution(self, text=None, top_k: int = 10):
        """Pretty prints topic distribution in a document.

        Parameters
        ----------
        text: str, optional
            Text to infer topic distribution for.
        top_k: int, default 10
            Top K topics to show.
        """
        columns, *rows = self._topic_distribution(text, top_k)
        table = Table()
        table.add_column("Topic name", justify="left", style="magenta")
        table.add_column("Score", justify="right", style="blue")
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)

    def export_topic_distribution(
        self, text=None, top_k: int = 10, format="csv"
    ) -> str:
        """Exports topic distribution as a text table.

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
        format: 'csv', 'latex' or 'markdown'
            Specifies which format should be used.
            'csv', 'latex' and 'markdown' are supported.
        """
        table = self._topic_distribution(text, top_k)
        return export_table(table, format=format)

    def run_topicwizard(self, **kwargs):
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
