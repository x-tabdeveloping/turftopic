import itertools
import warnings
from abc import ABC
from datetime import datetime
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

from turftopic.namers.base import TopicNamer
from turftopic.utils import export_table


def remove_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


class TopicContainer(ABC):
    """Base class for classes that contain topical information."""

    @property
    def has_negative_side(self) -> bool:
        return np.any(self.components_ < 0)

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

    def _top_terms(
        self, top_k: int = 10, positive: bool = True
    ) -> list[list[str]]:
        terms = []
        vocab = self.get_vocab()
        for component in self.components_:
            lowest = np.argpartition(component, top_k)[:top_k]
            lowest = lowest[np.argsort(component[lowest])]
            highest = np.argpartition(-component, top_k)[:top_k]
            highest = highest[np.argsort(-component[highest])]
            if not positive:
                terms.append(list(vocab[lowest]))
            else:
                terms.append(list(vocab[highest]))
        return terms

    def get_top_words(
        self, top_k: int = 10, positive: bool = True
    ) -> list[list[str]]:
        """Returns list of top words for each topic.

        Parameters
        ----------
        top_k: int, default 10
            Number of words to return.
        positive: bool, default True
            Indicates whether the highest
            or lowest scoring terms should be returned.
        """
        return self._top_terms(top_k, positive)

    def get_top_documents(
        self,
        raw_documents=None,
        document_topic_matrix=None,
        top_k: int = 10,
        positive: bool = True,
    ) -> list[list[str]]:
        """Returns list of top documents for each topic.

        Parameters
        ----------
        top_k: int, default 10
            Number of documents to return per topic.
        positive: bool, default True
            Indicates whether the highest
            or lowest scoring documents should be returned.
        """
        docs = []
        raw_documents = raw_documents or getattr(self, "corpus", None)
        if raw_documents is None:
            raise ValueError(
                "No corpus was passed, can't search for representative documents."
            )
        document_topic_matrix = document_topic_matrix or getattr(
            self, "document_topic_matrix", None
        )
        if document_topic_matrix is None:
            try:
                document_topic_matrix = self.transform(raw_documents)
            except AttributeError:
                raise ValueError(
                    "Transductive methods cannot "
                    "infer topical content in documents.\n"
                    "Please pass a document_topic_matrix."
                )
        for topic_doc_vec in document_topic_matrix.T:
            if positive:
                topic_doc_vec = -topic_doc_vec
            highest = np.argsort(topic_doc_vec)[:top_k]
            docs.append([raw_documents[i_doc] for i_doc in highest])
        return docs

    def get_top_images(self, top_k: int = True, positive: bool = True):
        """Returns list of top images for each topic.

        Parameters
        ----------
        top_k: int, default 10
            Number of images to return.
        positive: bool, default True
            Indicates whether the highest
            or lowest scoring images should be returned.
        """
        if not hasattr(self, "top_images"):
            raise ValueError(
                "Model either has not been fit or was fit without images. top_images property missing."
            )
        if (not positive) and not hasattr(self, "negative_images"):
            raise ValueError(
                "Model either has not been fit or was fit without images. top_images property missing."
            )
        top_images = self.top_images if positive else self.negative_images
        ims = []
        for topic_images in top_images:
            if len(topic_images) < top_k:
                warnings.warn(
                    "Number of images stored in the topic model is smaller than the specified top_k, returning all that the model has."
                )
            ims.append(topic_images[:top_k])
        return ims

    def _rename_automatic(self, namer: TopicNamer) -> list[str]:
        self.topic_names_ = namer.name_topics(self._top_terms())
        return self.topic_names_

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
        vocab = self.get_vocab()
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
                            vocab[highest], component[highest]
                        )
                    ]
                )
                concat_negative = ", ".join(
                    [
                        f"{word}({importance:.2f})"
                        for word, importance in zip(
                            vocab[lowest], component[lowest]
                        )
                    ]
                )
            else:
                concat_positive = ", ".join([word for word in vocab[highest]])
                concat_negative = ", ".join([word for word in vocab[lowest]])
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
        show_negative: bool, default False
            Indicates whether the most negative terms should also be displayed.
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
        show_negative: bool, default False
            Indicates whether the most negative terms should also be displayed.
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
        raw_documents=None,
        document_topic_matrix=None,
        top_k=5,
        show_negative: Optional[bool] = None,
    ) -> list[list[str]]:
        if show_negative is None:
            show_negative = self.has_negative_side
        raw_documents = (
            raw_documents
            if raw_documents is not None
            else getattr(self, "corpus", None)
        )
        if raw_documents is None:
            raise ValueError(
                "No corpus was passed, can't search for representative documents."
            )
        document_topic_matrix = (
            document_topic_matrix
            if document_topic_matrix is not None
            else getattr(self, "document_topic_matrix", None)
        )
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
        columns = []
        columns.append("Document")
        columns.append("Score")
        rows = []
        for document_id, score in zip(highest, scores):
            doc = raw_documents[document_id]
            doc = remove_whitespace(doc)
            if len(doc) > 300:
                doc = doc[:300] + "..."
            rows.append([doc, f"{score:.2f}"])
        if show_negative:
            rows.append(["...", ""])
            lowest = np.argpartition(document_topic_matrix[:, topic_id], kth)[
                :kth
            ]
            lowest = lowest[
                np.argsort(document_topic_matrix[lowest, topic_id])
            ]
            lowest = lowest[::-1]
            scores = document_topic_matrix[lowest, topic_id]
            for document_id, score in zip(lowest, scores):
                doc = raw_documents[document_id]
                doc = remove_whitespace(doc)
                if len(doc) > 300:
                    doc = doc[:300] + "..."
                rows.append([doc, f"{score:.2f}"])
        return [columns, *rows]

    def print_representative_documents(
        self,
        topic_id,
        raw_documents=None,
        document_topic_matrix=None,
        top_k=5,
        show_negative: Optional[bool] = None,
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
        show_negative: bool, default False
            Indicates whether lowest ranking documents should also be shown.
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
        raw_documents=None,
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
        table = self._representative_docs(
            topic_id,
            raw_documents,
            document_topic_matrix,
            top_k,
            show_negative,
        )
        return export_table(table, format=format)

    @property
    def topic_names(self) -> list[str]:
        """Names of the topics based on the highest scoring 4 terms."""
        topic_names = getattr(self, "topic_names_", None)
        if topic_names is not None:
            return list(topic_names)
        topic_desc = self.get_topics(top_k=4)
        names = []
        for topic_id, terms in topic_desc:
            concat_words = "_".join([word for word, importance in terms])
            names.append(f"{topic_id}_{concat_words}")
        return names

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
        self, text=None, topic_dist=None, top_k: int = 10
    ) -> list[list[str]]:
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
        highest = np.argsort(-topic_dist)[:top_k]
        columns = []
        columns.append("Topic name")
        columns.append("Score")
        rows = []
        for ind in highest:
            score = topic_dist[ind]
            rows.append([self.topic_names[ind], f"{score:.2f}"])
        return [columns, *rows]

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
        columns, *rows = self._topic_distribution(text, topic_dist, top_k)
        table = Table()
        table.add_column("Topic name", justify="left", style="magenta")
        table.add_column("Score", justify="right", style="blue")
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)

    def export_topic_distribution(
        self, text=None, topic_dist=None, top_k: int = 10, format="csv"
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
        table = self._topic_distribution(text, topic_dist, top_k)
        return export_table(table, format=format)

    def topics_df(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: Optional[bool] = None,
    ):
        """Extracts topics into a pandas dataframe.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        show_negative: bool, default False
            Indicates whether the most negative terms should also be displayed.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "You need to pip install pandas to be able to use dataframes."
            )
        columns, *rows = self._topics_table(top_k, show_scores, show_negative)
        return pd.DataFrame(rows, columns=columns)

    def representative_documents_df(
        self,
        topic_id,
        raw_documents=None,
        document_topic_matrix=None,
        top_k=5,
        show_negative: Optional[bool] = None,
    ):
        """Collects highest ranking documents in a topic to a dataframe.

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
        show_negative: bool, default False
            Indicates whether lowest ranking documents should also be shown.
        """
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "You need to pip install pandas to be able to use dataframes."
            )
        if show_negative is None:
            show_negative = self.has_negative_side
        raw_documents = raw_documents or getattr(self, "corpus", None)
        if raw_documents is None:
            raise ValueError(
                "No corpus was passed, can't search for representative documents."
            )
        document_topic_matrix = document_topic_matrix or getattr(
            self, "document_topic_matrix", None
        )
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
        columns = [["Document", "Score"]]
        rows = []
        for document_id, score in zip(highest, scores):
            doc = raw_documents[document_id]
            rows.append([doc, score])
        if show_negative:
            lowest = np.argpartition(document_topic_matrix[:, topic_id], kth)[
                :kth
            ]
            lowest = lowest[
                np.argsort(document_topic_matrix[lowest, topic_id])
            ]
            lowest = lowest[::-1]
            scores = document_topic_matrix[lowest, topic_id]
            for document_id, score in zip(lowest, scores):
                doc = raw_documents[document_id]
                rows.append([doc, score])
        return pd.DataFrame(rows, columns=columns)

    def topic_distribution_df(
        self, text=None, topic_dist=None, top_k: int = 10
    ):
        """Extracts topic distribution into a dataframe.

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
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "You need to pip install pandas to be able to use dataframes."
            )
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
        highest = np.argsort(-topic_dist)[:top_k]
        columns = []
        columns.append("Topic name")
        columns.append("Score")
        rows = []
        for ind in highest:
            score = topic_dist[ind]
            rows.append([self.topic_names[ind], score])
        return pd.DataFrame(rows, columns=columns)

    def get_time_slices(self) -> list[tuple[datetime, datetime]]:
        """Returns starting and ending datetime of
        each timeslice in the model."""
        bins = getattr(self, "time_bin_edges", None)
        if bins is None:
            raise AttributeError(
                "Topic model is not dynamic, time_bin_edges attribute is missing."
            )
        res = []
        for i_bin, slice_end in enumerate(bins[1:]):
            res.append((bins[i_bin], slice_end))
        return res

    def get_topics_over_time(
        self, top_k: int = 10
    ) -> list[list[tuple[Any, list[tuple[str, float]]]]]:
        """Returns high-level topic representations in form of the top K words
        in each topic.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.

        Returns
        -------
        list[list[tuple]]
            List of topics over each time slice in the dynamic model.
            Each time slice is a list of topics.
            Each topic is a tuple of topic ID and the top k words.
            Top k words are a list of (word, word_importance) pairs.
        """
        temporal_components = getattr(self, "temporal_components_", None)
        if temporal_components is None:
            raise AttributeError(
                "Topic model is not dynamic, temporal_components_ attribute is missing."
            )
        n_topics = temporal_components.shape[1]
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(n_topics))
        res = []
        for components in temporal_components:
            highest = np.argpartition(-components, top_k)[:, :top_k]
            vocab = self.get_vocab()
            top = []
            score = []
            for component, high in zip(components, highest):
                importance = component[high]
                high = high[np.argsort(-importance)]
                score.append(component[high])
                top.append(vocab[high])
            topics = []
            for topic, words, scores in zip(classes, top, score):
                topic_data = (topic, list(zip(words, scores)))
                topics.append(topic_data)
            res.append(topics)
        return res

    def _topics_over_time(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        date_format: str = "%Y %m %d",
    ) -> list[list[str]]:
        temporal_components = getattr(self, "temporal_components_", None)
        if temporal_components is None:
            raise AttributeError(
                "Topic model is not dynamic, temporal_components_ attribute is missing."
            )
        temporal_importance = getattr(self, "temporal_importance_", None)
        if temporal_components is None:
            raise AttributeError(
                "Topic model is not dynamic, temporal_importance_ attribute is missing."
            )
        slices = self.get_time_slices()
        slice_names = []
        for start_dt, end_dt in slices:
            start_str = start_dt.strftime(date_format)
            end_str = end_dt.strftime(date_format)
            slice_names.append(f"{start_str} - {end_str}")
        n_topics = temporal_components.shape[1]
        try:
            topic_names = self.topic_names
        except AttributeError:
            topic_names = [f"Topic {i}" for i in range(n_topics)]
        columns = []
        rows = []
        columns.append("Time Slice")
        for topic in topic_names:
            columns.append(topic)
        for slice_name, components, weights in zip(
            slice_names, temporal_components, temporal_importance
        ):
            fields = []
            fields.append(slice_name)
            vocab = self.get_vocab()
            for component, weight in zip(components, weights):
                if np.all(component == 0) or np.all(np.isnan(component)):
                    fields.append("Topic not present.")
                    continue
                if weight < 0:
                    component = -component
                top = np.argpartition(-component, top_k)[:top_k]
                importance = component[top]
                top = top[np.argsort(-importance)]
                top = top[importance != 0]
                scores = component[top]
                words = vocab[top]
                if show_scores:
                    concat_words = ", ".join(
                        [
                            f"{word}({importance:.2f})"
                            for word, importance in zip(words, scores)
                        ]
                    )
                else:
                    concat_words = ", ".join([word for word in words])
                fields.append(concat_words)
            rows.append(fields)
        return [columns, *rows]

    def print_topics_over_time(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        date_format: str = "%Y %m %d",
    ):
        """Pretty prints topics in the model in a table.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        """
        columns, *rows = self._topics_over_time(
            top_k, show_scores, date_format
        )
        table = Table(show_lines=True)
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)

    def export_topics_over_time(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        date_format: str = "%Y %m %d",
        format="csv",
    ) -> str:
        """Pretty prints topics in the model in a table.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        format: 'csv', 'latex' or 'markdown'
            Specifies which format should be used.
            'csv', 'latex' and 'markdown' are supported.
        """
        table = self._topics_over_time(top_k, show_scores, date_format)
        return export_table(table, format=format)

    def topics_over_time_df(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        format="csv",
    ):
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "You need to pip install pandas to be able to use dataframes."
            )

        def parse_time_slice(slice: str) -> tuple[datetime, datetime]:
            date_format = "%Y %m %d"
            start_date, end_date = slice.split(" - ")
            return datetime.strptime(
                start_date, date_format
            ), datetime.strptime(end_date, date_format)

        columns, *rows = self._topics_over_time(top_k, show_scores)
        df = pd.DataFrame(rows, columns=columns)
        df["Time Slice"] = df["Time Slice"].map(parse_time_slice)
        return df

    def plot_topics_over_time(
        self,
        top_k: int = 6,
        color_discrete_sequence: Optional[Iterable[str]] = None,
        color_discrete_map: Optional[dict[str, str]] = None,
    ):
        """Displays topics over time in the fitted dynamic model on a dynamic HTML figure.

        > You will need to `pip install plotly` to use this method.

        Parameters
        ----------
        top_k: int, default 6
            Number of top words per topic to display on the figure.
        color_discrete_sequence: Iterable[str], default None
            Color palette to use in the plot.
            Example:

            ```python
            import plotly.express as px
            model.plot_topics_over_time(color_discrete_sequence=px.colors.qualitative.Light24)
            ```

        color_discrete_map: dict[str, str], default None
            Topic names mapped to the colors that should
            be associated with them.

        Returns
        -------
        go.Figure
            Plotly graph objects Figure, that can be displayed or exported as
            HTML or static image.
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        temporal_components = getattr(self, "temporal_components_", None)
        if temporal_components is None:
            raise AttributeError(
                "Topic model is not dynamic, temporal_components_ attribute is missing."
            )
        temporal_importance = getattr(self, "temporal_importance_", None)
        if temporal_components is None:
            raise AttributeError(
                "Topic model is not dynamic, temporal_importance_ attribute is missing."
            )
        if color_discrete_sequence is not None:
            topic_colors = itertools.cycle(color_discrete_sequence)
        elif color_discrete_map is not None:
            topic_colors = [
                color_discrete_map[topic_name]
                for topic_name in self.topic_names
            ]
        else:
            topic_colors = px.colors.qualitative.Dark24
        fig = go.Figure()
        vocab = self.get_vocab()
        n_topics = temporal_components.shape[1]
        try:
            topic_names = self.topic_names
        except AttributeError:
            topic_names = [f"Topic {i}" for i in range(n_topics)]
        for trace_color, (i_topic, topic_imp_t) in zip(
            itertools.cycle(topic_colors), enumerate(temporal_importance.T)
        ):
            component_over_time = temporal_components[:, i_topic, :]
            name_over_time = []
            for component, importance in zip(component_over_time, topic_imp_t):
                if importance < 0:
                    component = -component
                top = np.argpartition(-component, top_k)[:top_k]
                values = component[top]
                if np.all(values == 0) or np.all(np.isnan(values)):
                    name_over_time.append("<not present>")
                    continue
                top = top[np.argsort(-values)]
                name_over_time.append(", ".join(vocab[top]))
            times = self.time_bin_edges[:-1]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=topic_imp_t,
                    mode="markers+lines",
                    text=name_over_time,
                    name=topic_names[i_topic],
                    hovertemplate="<b>%{text}</b>",
                    marker=dict(
                        line=dict(width=2, color="black"),
                        size=14,
                        color=trace_color,
                    ),
                    line=dict(width=3),
                )
            )
        fig.update_layout(
            template="plotly_white",
            hoverlabel=dict(font_size=16, bgcolor="white"),
            hovermode="x",
            font=dict(family="Roboto Mono"),
        )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5)
        fig.update_xaxes(title="Time Slice Start")
        fig.update_yaxes(title="Topic Importance")
        return fig

    @staticmethod
    def _image_grid(
        images: list[Image.Image],
        final_size=(1200, 1200),
        grid_size: tuple[int, int] = (4, 4),
    ):
        grid_img = Image.new("RGB", final_size, (255, 255, 255))
        cell_width = final_size[0] // grid_size[0]
        cell_height = final_size[1] // grid_size[1]
        n_rows, n_cols = grid_size
        for idx, img in enumerate(images[: n_rows * n_cols]):
            img = img.resize(
                (cell_width, cell_height), resample=Image.Resampling.LANCZOS
            )
            x_offset = (idx % grid_size[0]) * cell_width
            y_offset = (idx // grid_size[1]) * cell_height
            grid_img.paste(img, (x_offset, y_offset))
        return grid_img

    def plot_topics_with_images(self, n_cols: int = 3, grid_size: int = 4):
        """Plots the most important images for each topic, along with keywords.

        Note that you will need to `pip install plotly` to use plots in Turftopic.

        Parameters
        ----------
        n_cols: int, default 3
            Number of columns you want to have in the grid of topics.
        grid_size: int, default 4
            The square root of the number of images you want to display for a given topic.
            For instance if grid_size==4, all topics will have 16 images displayed,
            since the joint image will have 4 columns and 4 rows.

        Returns
        -------
        go.Figure
            Plotly figure containing top images and keywords for topics.
        """
        if not hasattr(self, "top_images"):
            raise ValueError(
                "Model either has not been fit or was fit without images. top_images property missing."
            )
        try:
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        negative_images = getattr(self, "negative_images", None)
        if negative_images is not None:
            # If the model has negative images, it should display them side by side with the positive ones.
            n_components = self.components_.shape[0]
            fig = go.Figure()
            width, height = 1200, 1200
            scale_factor = 0.25
            w, h = width * scale_factor, height * scale_factor
            padding = 10
            figure_height = (h + padding) * n_components
            figure_width = (w + padding) * 2
            fig = fig.add_trace(
                go.Scatter(
                    x=[0, figure_width],
                    y=[0, figure_height],
                    mode="markers",
                    marker_opacity=0,
                )
            )
            vocab = self.get_vocab()
            for i, component in enumerate(self.components_):
                positive = vocab[np.argsort(-component)[:7]]
                negative = vocab[np.argsort(component)[:7]]
                pos_image = self._image_grid(
                    self.top_images[i],
                    (width, height),
                    grid_size=(grid_size, grid_size),
                )
                neg_image = self._image_grid(
                    self.negative_images[i],
                    (width, height),
                    grid_size=(grid_size, grid_size),
                )
                x0 = 0
                y0 = (h + padding) * (n_components - i)
                fig = fig.add_layout_image(
                    dict(
                        x=x0,
                        sizex=w,
                        y=y0,
                        sizey=h,
                        xref="x",
                        yref="y",
                        opacity=1.0,
                        layer="below",
                        sizing="stretch",
                        source=pos_image,
                    ),
                )
                fig.add_annotation(
                    x=(w / 2),
                    y=(h + padding) * (n_components - i) - (h / 2),
                    text="<b> " + "<br> ".join(positive),
                    font=dict(
                        size=16,
                        family="Roboto Mono",
                        color="white",
                    ),
                    bgcolor="rgba(0,0,255, 0.5)",
                )
                x0 = (w + padding) * 1
                fig = fig.add_layout_image(
                    dict(
                        x=x0,
                        sizex=w,
                        y=y0,
                        sizey=h,
                        xref="x",
                        yref="y",
                        opacity=1.0,
                        layer="below",
                        sizing="stretch",
                        source=neg_image,
                    ),
                )
                fig.add_annotation(
                    x=(w + padding) + (w / 2),
                    y=(h + padding) * (n_components - i) - (h / 2),
                    text="<b> " + "<br> ".join(negative),
                    font=dict(
                        size=16,
                        family="Times New Roman",
                        color="white",
                    ),
                    bgcolor="rgba(255,0,0, 0.5)",
                )
            fig = fig.update_xaxes(visible=False, range=[0, figure_width])
            fig = fig.update_yaxes(
                visible=False,
                range=[0, figure_height],
                # the scaleanchor attribute ensures that the aspect ratio stays constant
                scaleanchor="x",
            )
            fig = fig.update_layout(
                width=figure_width,
                height=figure_height,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            )
            return fig
        else:
            fig = go.Figure()
            width, height = 1200, 1200
            scale_factor = 0.25
            w, h = width * scale_factor, height * scale_factor
            padding = 10
            n_components = self.components_.shape[0]
            n_rows = n_components // n_cols + int(bool(n_components % n_cols))
            figure_height = (h + padding) * n_rows
            figure_width = (w + padding) * n_cols
            fig = fig.add_trace(
                go.Scatter(
                    x=[0, figure_width],
                    y=[0, figure_height],
                    mode="markers",
                    marker_opacity=0,
                )
            )
            vocab = self.get_vocab()
            for i, component in enumerate(self.components_):
                col = i % n_cols
                row = i // n_cols
                top_7 = vocab[np.argsort(-component)[:7]]
                images = self.top_images[i]
                image = self._image_grid(
                    images, (width, height), grid_size=(grid_size, grid_size)
                )
                x0 = (w + padding) * col
                y0 = (h + padding) * (n_rows - row)
                fig = fig.add_layout_image(
                    dict(
                        x=x0,
                        sizex=w,
                        y=y0,
                        sizey=h,
                        xref="x",
                        yref="y",
                        opacity=1.0,
                        layer="below",
                        sizing="stretch",
                        source=image,
                    ),
                )
                fig.add_annotation(
                    x=(w + padding) * col + (w / 2),
                    y=(h + padding) * (n_rows - row) - (h / 2),
                    text="<b> " + "<br> ".join(top_7),
                    font=dict(
                        size=16,
                        family="Times New Roman",
                        color="white",
                    ),
                    bgcolor="rgba(0,0,0, 0.5)",
                )
            fig = fig.update_xaxes(visible=False, range=[0, figure_width])
            fig = fig.update_yaxes(
                visible=False,
                range=[0, figure_height],
                # the scaleanchor attribute ensures that the aspect ratio stays constant
                scaleanchor="x",
            )
            fig = fig.update_layout(
                width=figure_width,
                height=figure_height,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
            )
            return fig

    def plot_multimodal_topics(
        self,
        top_k: int = 10,
        grid_size: int = 4,
        raw_documents=None,
        document_topic_matrix=None,
    ):
        """Plots all multimodal topics in a model along with top documents individually,
        and provides a slider to switch between them.

        Parameters
        ----------
        top_k: int = 10
            Number of top words and documents to display.
        grid_size: int, default 4
            The square root of the number of images you want to display for a given topic.
            For instance if grid_size==4, all topics will have 16 images displayed,
            since the joint image will have 4 columns and 4 rows.
        raw_documents: list of str, optional
            List of documents to consider.
        document_topic_matrix: ndarray of shape (n_documents, n_topics), optional
            Document topic matrix to use. This is useful for transductive methods,
            as they cannot infer topics from text.

        """
        if not hasattr(self, "top_images"):
            raise ValueError(
                "Model either has not been fit or was fit without images. top_images property missing."
            )
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        negative_images = getattr(self, "negative_images", None)
        negative_topics = (
            self.get_top_words(top_k=top_k, positive=False)
            if negative_images is not None
            else None
        )
        specs = [{"type": "image"}, {"type": "table"}]
        if negative_images is not None:
            specs.append({"type": "image"})
        fig = make_subplots(
            rows=1,
            cols=2 if negative_images is None else 3,
            specs=[specs],
            shared_yaxes=True,
            shared_xaxes=True,
        )
        width, height = 1200, 1200
        topics = self.get_top_words(top_k=top_k)
        n_topics = len(topics)
        annotations = []
        for i, topic in enumerate(topics):
            images = self.top_images[i]
            image = TopicContainer._image_grid(
                images, (width, height), grid_size=(grid_size, grid_size)
            )
            trace = px.imshow(image).data[0]
            trace.visible = False
            fig.add_trace(trace, col=1, row=1)
            annt = dict(
                x=width / 2,
                y=height / 2,
                text="<b> " + "<br> ".join(topic),
                font=dict(
                    size=16,
                    family="Roboto Mono",
                    color="white",
                ),
                bgcolor="rgba(0,0,255, 0.5)",
                xref="x",
                yref="y",
            )
            annotations.append(annt)
        if negative_topics is not None:
            for i, negative_topic in enumerate(negative_topics):
                images = negative_images[i]
                image = TopicContainer._image_grid(
                    images, (width, height), grid_size=(grid_size, grid_size)
                )
                trace = px.imshow(image).data[0]
                trace.visible = False
                fig.add_trace(trace, col=3, row=1)
                annotations.append(
                    dict(
                        x=width / 2,
                        y=height / 2,
                        text="<b> " + "<br> ".join(negative_topic),
                        font=dict(
                            size=16,
                            family="Roboto Mono",
                            color="white",
                        ),
                        bgcolor="rgba(255,0,0, 0.5)",
                        xref="x2",
                        yref="y2",
                    )
                )
        fig = fig.add_annotation(**annotations[0])
        if negative_images is not None:
            fig.add_annotation(**annotations[n_topics])
        classes = getattr(self, "classes_", np.arange(n_topics))
        for i, topic_id in enumerate(classes):
            header, *cells = self._representative_docs(
                topic_id,
                raw_documents=raw_documents,
                document_topic_matrix=document_topic_matrix,
                top_k=top_k,
                show_negative=negative_images is not None,
            )
            # Transposing cells
            cells = [list(column) for column in zip(*cells)]
            fig.add_trace(
                go.Table(
                    columnorder=[1, 2],
                    columnwidth=[400, 80],
                    header=dict(
                        values=header,
                        fill_color="white",
                        line=dict(color="black", width=4),
                        font=dict(
                            family="Roboto Mono", color="black", size=20
                        ),
                    ),
                    cells=dict(
                        values=cells,
                        fill_color="white",
                        align="left",
                        line=dict(color="black", width=2),
                        font=dict(
                            family="Roboto Mono", color="black", size=16
                        ),
                        height=40,
                    ),
                    visible=False,
                ),
                col=2,
                row=1,
            )
        fig.data[0].visible = True
        fig.data[n_topics].visible = True
        if negative_images is not None:
            fig.data[n_topics * 2].visible = True
        fig = fig.update_layout(
            margin={"l": 0, "r": 0, "t": 40, "b": 20},
            template="plotly_white",
            font=dict(family="Roboto Mono"),
        )
        fig = fig.update_xaxes(visible=False)
        fig = fig.update_yaxes(visible=False)
        steps = []
        n_traces = n_topics * 2 if negative_images is None else n_topics * 3
        for i, name in enumerate(self.topic_names):
            _annt = [annotations[i]]
            if negative_topics is not None:
                _annt.append(annotations[n_topics + i])
            step = dict(
                method="update",
                label=name,
                args=[
                    {"visible": [False] * n_traces},
                    {
                        "title": "Topic: " + name,
                        "annotations": _annt,
                    },
                ],
            )
            step["args"][0]["visible"][i] = True
            step["args"][0]["visible"][n_topics + i] = True
            if negative_images is not None:
                step["args"][0]["visible"][n_topics * 2 + i] = True
            steps.append(step)
        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Topic: "},
                pad={"t": 50, "b": 20, "r": 40, "l": 40},
                steps=steps,
            )
        ]
        fig = fig.update_layout(sliders=sliders)
        return fig
