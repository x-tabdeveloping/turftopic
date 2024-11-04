import json
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
from huggingface_hub import HfApi
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from turftopic.data import TopicData
from turftopic.encoders import ExternalEncoder
from turftopic.namers.base import TopicNamer
from turftopic.serialization import create_readme, get_package_versions
from turftopic.utils import export_table


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

    def _rename_automatic(self, namer: TopicNamer) -> list[str]:
        self.topic_names_ = namer.name_topics(self._top_terms())
        return self.topic_names_

    def _topics_table(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: bool = False,
    ) -> list[list[str]]:
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
        show_negative: bool = False,
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
        show_negative: bool = False,
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
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: bool = False,
    ) -> list[list[str]]:
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
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: bool = False,
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
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: bool = False,
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
            document_topic_matrix = self.transform(
                corpus, embeddings=embeddings
            )
        except (AttributeError, NotFittedError):
            document_topic_matrix = self.fit_transform(
                corpus, embeddings=embeddings
            )
        dtm = self.vectorizer.transform(corpus)  # type: ignore
        res: TopicData = {
            "corpus": corpus,
            "document_term_matrix": dtm,
            "vocab": self.get_vocab(),
            "document_topic_matrix": document_topic_matrix,
            "document_representation": embeddings,
            "topic_term_matrix": self.components_,  # type: ignore
            "transform": getattr(self, "transform", None),
            "topic_names": self.topic_names,
        }
        return res

    def to_disk(self, out_dir: Union[Path, str]):
        """Persists model to directory on your machine.

        Parameters
        ----------
        out_dir: Path | str
            Directory to save the model to.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        package_versions = get_package_versions()
        with out_dir.joinpath("package_versions.json").open("w") as ver_file:
            ver_file.write(json.dumps(package_versions))
        joblib.dump(self, out_dir.joinpath("model.joblib"))

    def push_to_hub(self, repo_id: str):
        """Uploads model to HuggingFace Hub

        Parameters
        ----------
        repo_id: str
            Repository to upload the model to.
        """
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            readme_path = Path(tmp_dir).joinpath("README.md")
            with readme_path.open("w") as readme_file:
                readme_file.write(create_readme(self, repo_id))
            self.to_disk(tmp_dir)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                repo_type="model",
            )
