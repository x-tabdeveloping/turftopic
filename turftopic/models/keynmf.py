from datetime import datetime
from typing import Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel, Encoder
from turftopic.data import TopicData
from turftopic.dynamic import DynamicTopicModel
from turftopic.models._keynmf import KeywordExtractor, KeywordNMF


class KeyNMF(ContextualModel, DynamicTopicModel):
    """Extracts keywords from documents based on semantic similarity of
    term encodings to document encodings.
    Topics are then extracted with non-negative matrix factorization from
    keywords' proximity to documents.

    ```python
    from turftopic import KeyNMF

    corpus: list[str] = ["some text", "more text", ...]

    model = KeyNMF(10, top_n=10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int
        Number of topics.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    top_n: int, default 25
        Number of keywords to extract for each document.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    """

    def __init__(
        self,
        n_components: int,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        top_n: int = 25,
        random_state: Optional[int] = None,
    ):
        self.random_state = random_state
        self.n_components = n_components
        self.top_n = top_n
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = vectorizer
        self.model = KeywordNMF(
            n_components=n_components, seed=random_state, top_n=self.top_n
        )
        self.extractor = KeywordExtractor(
            top_n=self.top_n, vectorizer=self.vectorizer, encoder=self.encoder_
        )

    def extract_keywords(
        self,
        batch_or_document: Union[str, list[str]],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[dict[str, float]]:
        """Extracts keywords from a document or a batch of documents.

        Parameters
        ----------
        batch_or_document: str | list[str]
            A single document or a batch of documents.
        embeddings: ndarray, optional
            Precomputed document embeddings.
        """
        if isinstance(batch_or_document, str):
            batch_or_document = [batch_or_document]
        return self.extractor.batch_extract_keywords(
            batch_or_document, embeddings=embeddings
        )

    def fit_transform(
        self,
        raw_documents=None,
        y=None,
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
    ) -> np.ndarray:
        """Fits topic model and returns topic importances for documents.

        Parameters
        ----------
        raw_documents: iterable of str, optional
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.
        keywords: list[dict[str, float]], optional
            Precomputed keyword dictionaries.

        Returns
        -------
        ndarray of shape (n_dimensions, n_topics)
            Document-topic matrix.
        """
        console = Console()
        with console.status("Running KeyNMF") as status:
            if keywords is None:
                status.update("Extracting keywords")
                keywords = self.extract_keywords(
                    raw_documents, embeddings=embeddings
                )
                console.log("Keyword extraction done.")
            status.update("Decomposing with NMF")
            try:
                doc_topic_matrix = self.model.transform(keywords)
            except (NotFittedError, AttributeError):
                doc_topic_matrix = self.model.fit_transform(keywords)
                self.components_ = self.model.components
            console.log("Model fitting done.")
        return doc_topic_matrix

    def fit(
        self,
        raw_documents=None,
        y=None,
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
    ) -> np.ndarray:
        """Fits topic model and returns topic importances for documents.

        Parameters
        ----------
        raw_documents: iterable of str, optional
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.
        keywords: list[dict[str, float]], optional
            Precomputed keyword dictionaries.
        """
        self.fit_transform(raw_documents, y, embeddings, keywords)
        return self

    def get_vocab(self) -> np.ndarray:
        return np.array(self.model.index_to_key)

    def transform(
        self,
        raw_documents=None,
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
    ) -> np.ndarray:
        """Infers topic importances for new documents based on a fitted model.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.
        keywords: list[dict[str, float]], optional
            Precomputed keyword dictionaries.

        Returns
        -------
        ndarray of shape (n_dimensions, n_topics)
            Document-topic matrix.
        """
        if keywords is None and raw_documents is None:
            raise ValueError(
                "You have to pass either keywords or raw_documents."
            )
        if keywords is None:
            keywords = self.extract_keywords(
                list(raw_documents), embeddings=embeddings
            )
        return self.model.transform(keywords)

    def partial_fit(
        self,
        raw_documents: Optional[list[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
    ):
        """Online fits KeyNMF on a batch of documents.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.
        keywords: list[dict[str, float]], optional
            Precomputed keyword dictionaries.
        """
        if keywords is None and raw_documents is None:
            raise ValueError(
                "You have to pass either keywords or raw_documents."
            )
        if keywords is None:
            keywords = self.extract_keywords(
                raw_documents, embeddings=embeddings
            )
        self.model.partial_fit(keywords)
        self.components_ = self.model.components
        return self

    def prepare_topic_data(
        self,
        corpus: list[str],
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
    ) -> TopicData:
        if keywords is None and corpus is None:
            raise ValueError(
                "You have to pass either keywords or raw_documents."
            )
        console = Console()
        with console.status("Running KeyNMF") as status:
            if embeddings is None:
                embeddings = self.encode_documents(corpus)
            if keywords is None:
                status.update("Extracting keywords")
                keywords = self.extract_keywords(corpus, embeddings=embeddings)
                console.log("Keyword extraction done.")
            if (corpus is not None) and (len(keywords) != len(corpus)):
                raise ValueError(
                    "length of keywords is not the same as length of the corpus"
                )
            status.update("Decomposing with NMF")
            try:
                doc_topic_matrix = self.model.transform(keywords)
            except (NotFittedError, AttributeError):
                doc_topic_matrix = self.model.fit_transform(keywords)
                self.components_ = self.model.components
            console.log("Model fitting done.")
            document_term_matrix = self.model.vectorize(keywords)
        res: TopicData = {
            "corpus": corpus,
            "document_term_matrix": document_term_matrix,
            "vocab": self.get_vocab(),
            "document_topic_matrix": doc_topic_matrix,
            "document_representation": embeddings,
            "topic_term_matrix": self.components_,  # type: ignore
            "transform": getattr(self, "transform", None),
            "topic_names": self.topic_names,
        }
        return res

    def fit_transform_dynamic(
        self,
        raw_documents=None,
        timestamps: Optional[list[datetime]] = None,
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
        bins: Union[int, list[datetime]] = 10,
    ) -> np.ndarray:
        if timestamps is None:
            raise TypeError(
                "You have to pass timestamps when fitting a dynamic model."
            )
        if keywords is None and raw_documents is None:
            raise ValueError(
                "You have to pass either keywords or raw_documents."
            )
        if keywords is None:
            keywords = self.extract_keywords(
                raw_documents, embeddings=embeddings
            )
        time_labels, self.time_bin_edges = self.bin_timestamps(
            timestamps, bins
        )
        doc_topic_matrix = self.model.fit_transform_dynamic(
            keywords, time_labels, self.time_bin_edges
        )
        self.temporal_importance_ = (
            self.model.temporal_importance_.T
            / self.model.temporal_importance_.sum(axis=1)
        ).T
        self.temporal_components_ = self.model.temporal_components
        self.components_ = self.model.components
        return doc_topic_matrix

    def partial_fit_dynamic(
        self,
        raw_documents=None,
        timestamps: Optional[list[datetime]] = None,
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
        bins: Union[int, list[datetime]] = 10,
    ):
        """Online fits Dynamic KeyNMF on a batch of documents.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.
        keywords: list[dict[str, float]], optional
            Precomputed keyword dictionaries.
        timestamps: list[datetime], optional
            List of timestamps for the batch.
        bins: list[datetime]
            Explicit time bin edges for the dynamic model.
        """
        if timestamps is None:
            raise TypeError(
                "You have to pass timestamps when fitting a dynamic model."
            )
        if keywords is None and raw_documents is None:
            raise ValueError(
                "You have to pass either keywords or raw_documents."
            )
        time_bin_edges = getattr(self, "time_bin_edges", None)
        if time_bin_edges is None:
            if isinstance(bins, int):
                raise TypeError(
                    "You have to pass explicit time bins (list of time bin edges) when partial "
                    "fitting KeyNMF, at least at the first call."
                )
            else:
                self.time_bin_edges = bins
        time_labels, self.time_bin_edges = self.bin_timestamps(
            timestamps, self.time_bin_edges
        )
        if keywords is None:
            keywords = self.extract_keywords(
                raw_documents, embeddings=embeddings
            )
        self.model.partial_fit_dynamic(
            keywords, time_labels, self.time_bin_edges
        )
        self.temporal_importance_ = (
            self.model.temporal_importance_.T
            / self.model.temporal_importance_.sum(axis=1)
        ).T
        self.temporal_components_ = self.model.temporal_components
        self.components_ = self.model.components
        return self
