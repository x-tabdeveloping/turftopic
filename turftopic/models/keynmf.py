import warnings
from datetime import datetime
from typing import Literal, Optional, Union

import numpy as np
import scipy.sparse as spr
from PIL import Image
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from turftopic.base import ContextualModel, Encoder
from turftopic.data import TopicData
from turftopic.dynamic import DynamicTopicModel
from turftopic.encoders.multimodal import MultimodalEncoder
from turftopic.hierarchical import DivisibleTopicNode
from turftopic.models._keynmf import KeywordNMF, SBertKeywordExtractor
from turftopic.models.wnmf import weighted_nmf
from turftopic.multimodal import (ImageRepr, MultimodalEmbeddings,
                                  MultimodalModel)
from turftopic.vectorizers.default import default_vectorizer


class KeyNMF(ContextualModel, DynamicTopicModel, MultimodalModel):
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
    metric: "cosine" or "dot", default "cosine"
        Similarity metric to use for keyword extraction.
    seed_phrase: str, default None
        Describes an aspect of the corpus that the model should explore.
        It can be a free-text query, such as
        "Christian Denominations: Protestantism and Catholicism"
    cross_lingual: bool, default False
        Indicates whether KeyNMF should match terms across languages.
        This is useful when you have a corpus containing multiple languages.
    term_match_threshold: float, default 0.9
        Cosine similarity threshold for matching terms across languages.
    """

    def __init__(
        self,
        n_components: int,
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        top_n: int = 25,
        random_state: Optional[int] = None,
        metric: Literal["cosine", "dot"] = "cosine",
        seed_phrase: Optional[str] = None,
        cross_lingual: bool = False,
        term_match_threshold: float = 0.9,
    ):
        self.random_state = random_state
        self.n_components = n_components
        self.top_n = top_n
        self.metric = metric
        self.encoder = encoder
        self._has_custom_vectorizer = vectorizer is not None
        if isinstance(encoder, str):
            if (
                encoder == "sentence-transformers/all-MiniLM-L6-v2"
            ) and cross_lingual:
                warnings.warn(
                    "all-MiniLM is incompatible with cross-lingual transfer, using paraphrase-multilingual-MiniLM-L12-v2."
                )
                encoder = "paraphrase-multilingual-MiniLM-L12-v2"
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        self.validate_encoder()
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.model = KeywordNMF(
            n_components=n_components, seed=random_state, top_n=self.top_n
        )
        self.extractor = SBertKeywordExtractor(
            top_n=self.top_n,
            vectorizer=self.vectorizer,
            encoder=self.encoder_,
            metric=self.metric,
        )
        self.seed_phrase = seed_phrase
        self.seed_embedding = None
        if self.seed_phrase is not None:
            self.seed_embedding = self.encoder_.encode([self.seed_phrase])[0]
        self.cross_lingual = cross_lingual
        self.term_match_threshold = term_match_threshold

    def extract_keywords(
        self,
        batch_or_document: Union[str, list[str]],
        embeddings: Optional[np.ndarray] = None,
        fitting: bool = True,
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
        keywords = self.extractor.batch_extract_keywords(
            batch_or_document,
            embeddings=embeddings,
            seed_embedding=self.seed_embedding,
            fitting=fitting,
        )
        if self.cross_lingual:
            keywords = self.extractor.match_terms(
                keywords, threshold=self.term_match_threshold
            )
        return keywords

    def vectorize(
        self,
        raw_documents=None,
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
    ) -> spr.csr_array:
        """Creates document-term-matrix from documents."""
        if keywords is None:
            keywords = self.extract_keywords(
                raw_documents, embeddings=embeddings
            )
        return self.model.vectorize(keywords)

    def divide_topic(
        self,
        node: DivisibleTopicNode,
        n_subtopics: int,
    ) -> list[DivisibleTopicNode]:
        document_term_matrix = getattr(self, "document_term_matrix", None)
        if document_term_matrix is None:
            raise ValueError(
                "document_term_matrix is needed for computing hierarchies. Perhaps you fitted the model online?"
            )
        dtm = document_term_matrix
        subtopics = []
        weight = node.document_topic_vector
        subcomponents, sub_doc_topic = weighted_nmf(
            dtm, weight, n_subtopics, self.random_state, max_iter=200
        )
        subcomponents = subcomponents * np.log(
            1 + subcomponents.mean() / (subcomponents.sum(axis=0) + 1)
        )
        subcomponents = normalize(subcomponents, axis=1, norm="l2")
        for i, component, doc_topic_vector in zip(
            range(n_subtopics), subcomponents, sub_doc_topic.T
        ):
            sub = DivisibleTopicNode(
                self,
                path=(*node.path, i),
                word_importance=component,
                document_topic_vector=doc_topic_vector,
                children=None,
            )
            subtopics.append(sub)
        return subtopics

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
        if self.seed_phrase is not None and keywords is not None:
            warnings.warn(
                "Seed phrase is specified, but keyword matrix is pre-computed. The seed phrase will be ignored. Note that this is not a problem if you already calculated the keyword matrix using the seed phrase."
            )
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
        self.document_topic_matrix = doc_topic_matrix
        self.document_term_matrix = self.model.vectorize(keywords)
        self.hierarchy = DivisibleTopicNode.create_root(
            self, self.components_, self.document_topic_matrix
        )
        return doc_topic_matrix

    def fit_transform_multimodal(
        self,
        raw_documents: list[str],
        images: list[ImageRepr],
        y=None,
        embeddings: Optional[MultimodalEmbeddings] = None,
    ) -> np.ndarray:
        self.validate_embeddings(embeddings)
        console = Console()
        self.multimodal_embeddings = embeddings
        with console.status("Fitting model") as status:
            if self.multimodal_embeddings is None:
                status.update("Encoding documents")
                self.multimodal_embeddings = self.encode_multimodal(
                    raw_documents, images
                )
                console.log("Documents encoded.")
            status.update("Extracting keywords")
            document_keywords = self.extract_keywords(
                raw_documents,
                embeddings=self.multimodal_embeddings["document_embeddings"],
            )
            image_keywords = self.extract_keywords(
                raw_documents,
                embeddings=self.multimodal_embeddings["image_embeddings"],
            )
            console.log("Keyword extraction done.")
            status.update("Decomposing with NMF")
            try:
                doc_topic_matrix = self.model.transform(document_keywords)
            except (NotFittedError, AttributeError):
                doc_topic_matrix = self.model.fit_transform(document_keywords)
                self.components_ = self.model.components
            console.log("Model fitting done.")
            status.update("Transforming images")
            self.image_topic_matrix = self.model.transform(image_keywords)
            self.top_images: list[list[Image.Image]] = self.collect_top_images(
                images, self.image_topic_matrix
            )
            console.log("Images transformed")
        self.document_topic_matrix = doc_topic_matrix
        self.document_term_matrix = self.model.vectorize(document_keywords)
        self.hierarchy = DivisibleTopicNode.create_root(
            self, self.components_, self.document_topic_matrix
        )
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
        self.fit_transform(
            raw_documents, y, embeddings=embeddings, keywords=keywords
        )
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
        if self.seed_phrase is not None and keywords is not None:
            warnings.warn(
                "Seed phrase is specified, but keyword matrix is pre-computed. The seed phrase will be ignored. Note that this is not a problem if you already calculated the keyword matrix using the seed phrase."
            )
        if keywords is None and raw_documents is None:
            raise ValueError(
                "You have to pass either keywords or raw_documents."
            )
        if keywords is None:
            keywords = self.extract_keywords(
                list(raw_documents),
                embeddings=embeddings,
                fitting=False,
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
        if self.seed_phrase is not None and keywords is not None:
            warnings.warn(
                "Seed phrase is specified, but keyword matrix is pre-computed. The seed phrase will be ignored. Note that this is not a problem if you already calculated the keyword matrix using the seed phrase."
            )
        if self.cross_lingual:
            raise ValueError(
                "Cross-lingual online topic modeling is yet to be implemented in KeyNMF."
            )
        if not self._has_custom_vectorizer:
            self.vectorizer = CountVectorizer(stop_words="english")
            self._has_custom_vectorizer = True
        min_df = self.vectorizer.min_df
        max_df = self.vectorizer.max_df
        if (min_df != 1) or (max_df != 1.0):
            warnings.warn(
                f"""When applying partial fitting, the vectorizer is fitted batch-wise in KeyNMF.
            You have a vectorizer with min_df={min_df}, and max_df={max_df}.
            If you continue with these settings, all tokens might get filtered out.
            We recommend setting min_df=1 and max_df=1.0 for online fitting.
            `model = KeyNMF(10, vectorizer=CountVectorizer(min_df=1, max_df=1.0)`
            """
            )
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
        corpus: Optional[list[str]],
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
    ) -> TopicData:
        if keywords is None and corpus is None:
            raise ValueError(
                "You have to pass either keywords or raw_documents."
            )
        if self.seed_phrase is not None and keywords is not None:
            warnings.warn(
                "Seed phrase is specified, but keyword matrix is pre-computed. The seed phrase will be ignored. Note that this is not a problem if you already calculated the keyword matrix using the seed phrase."
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
        self.document_topic_matrix = doc_topic_matrix
        self.document_term_matrix = document_term_matrix
        self.hierarchy = DivisibleTopicNode.create_root(
            self, self.components_, self.document_topic_matrix
        )
        res = TopicData(
            corpus=corpus,
            document_term_matrix=document_term_matrix,
            vocab=self.get_vocab(),
            document_topic_matrix=doc_topic_matrix,
            document_representation=embeddings,
            topic_term_matrix=self.components_,  # type: ignore
            transform=getattr(self, "transform", None),
            topic_names=self.topic_names,
            hierarchy=getattr(self, "hierarchy", None),
        )
        return res

    def prepare_dynamic_topic_data(
        self,
        corpus: Optional[list[str]],
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
        keywords: Optional[list[dict[str, float]]] = None,
    ):
        if self.seed_phrase is not None and keywords is not None:
            warnings.warn(
                "Seed phrase is specified, but keyword matrix is pre-computed. The seed phrase will be ignored. Note that this is not a problem if you already calculated the keyword matrix using the seed phrase."
            )
        if ((corpus is not None) and (keywords is not None)) and (
            len(keywords) != len(corpus)
        ):
            raise ValueError(
                "Length of keywords is not the same as length of the corpus"
            )
        if (corpus is None) and (keywords is None):
            raise TypeError(
                "You have to pass keywords or a corpus, but both are None."
            )
        if embeddings is None:
            embeddings = self.encode_documents(corpus)
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
                doc_topic_matrix = self.fit_transform_dynamic(
                    corpus,
                    embeddings=embeddings,
                    keywords=keywords,
                    bins=bins,
                    timestamps=timestamps,
                )
            console.log("Model fitting done.")
            document_term_matrix = self.model.vectorize(keywords)
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(self.components_.shape[0]))
        res = TopicData(
            corpus=corpus,
            document_term_matrix=document_term_matrix,
            vocab=self.get_vocab(),
            document_topic_matrix=doc_topic_matrix,
            document_representation=embeddings,
            topic_term_matrix=self.components_,  # type: ignore
            transform=getattr(self, "transform", None),
            topic_names=self.topic_names,
            classes=classes,
            temporal_components=self.temporal_components_,
            temporal_importance=self.temporal_importance_,
            time_bin_edges=self.time_bin_edges,
        )
        return res

    def fit_transform_dynamic(
        self,
        raw_documents=None,
        timestamps: Optional[list[datetime]] = None,
        embeddings: Optional[np.ndarray] = None,
        keywords: Optional[list[dict[str, float]]] = None,
        bins: Union[int, list[datetime]] = 10,
    ) -> np.ndarray:
        if self.seed_phrase is not None and keywords is not None:
            warnings.warn(
                "Seed phrase is specified, but keyword matrix is pre-computed. The seed phrase will be ignored. Note that this is not a problem if you already calculated the keyword matrix using the seed phrase."
            )
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
        self.document_topic_matrix = doc_topic_matrix
        self.document_term_matrix = self.model.vectorize(keywords)
        self.hierarchy = DivisibleTopicNode.create_root(
            self, self.components_, self.document_topic_matrix
        )
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
        if self.seed_phrase is not None and keywords is not None:
            warnings.warn(
                "Seed phrase is specified, but keyword matrix is pre-computed. The seed phrase will be ignored. Note that this is not a problem if you already calculated the keyword matrix using the seed phrase."
            )
        if self.cross_lingual:
            raise ValueError(
                "Cross-lingual online topic modeling is yet to be implemented in KeyNMF."
            )
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
