from collections import OrderedDict
from typing import Literal, Union

import numpy as np

from turftopic.base import ContextualModel, Encoder
from turftopic.encoders.multimodal import MultimodalEncoder
from turftopic.late import LateModel, Offsets, RaggedArray
from turftopic.spans.base import SpanExtractor

Seeds = tuple[list[str], list[str]]


class ConceptVectorProjection(ContextualModel):
    """Concept Vector Projection model from [Lyngbæk et al. (2025)](https://doi.org/10.63744/nVu1Zq5gRkuD)
    Can be used to project document embeddings onto a difference projection vector between positive and negative seed phrases.
    The primary use case is sentiment analysis, and continuous sentiment scores,
    especially for languages where dedicated models are not available.

    Parameters
    ----------
    seeds: (list[str], list[str]) or list of (str, (list[str], list[str]))
        If you want to project to a single concept, then
        a tuple of (list of negative terms, list of positive terms). <br>
        If there are multiple concepts, they should be specified as (name, Seeds) tuples in a list.
        Alternatively, seeds can be an OrderedDict with the names of the concepts being the keys,
        and the tuples of negative and positive seeds as the values.
    encoder: str or SentenceTransformer
        Model to produce document representations, paraphrase-multilingual-mpnet-base-v2 is the default
        per Lyngbæk et al. (2025).
    trf_kwargs: dict, default None
        Keyword arguments to apply when loading the Encoder model.
    encode_kwargs: dict, default None
        Keyword arguments to apply encoding documents with the encoder.
    """

    def __init__(
        self,
        seeds: Union[Seeds, list[tuple[str, Seeds]], OrderedDict[str, Seeds]],
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        trf_kwargs=None,
        encode_kwargs=None,
    ):
        self.seeds = seeds
        if isinstance(seeds, OrderedDict):
            self._seeds = seeds
        elif (
            (len(seeds) == 2)
            and (isinstance(seeds, tuple))
            and (isinstance(seeds[0][0], str))
        ):
            self._seeds = OrderedDict([("default", seeds)])
        else:
            self._seeds = OrderedDict(seeds)
        self.encoder = encoder
        self.trf_kwargs = trf_kwargs
        self.encode_kwargs = encode_kwargs
        self.load_encoder()
        self.classes_ = np.array([name for name in self._seeds])
        self.concept_matrix_ = []
        for _, (positive, negative) in self._seeds.items():
            positive_emb = self.encode_documents(list(positive))
            negative_emb = self.encode_documents(list(negative))
            cv = np.mean(positive_emb, axis=0) - np.mean(negative_emb, axis=0)
            self.concept_matrix_.append(cv / np.linalg.norm(cv))
        self.concept_matrix_ = np.stack(self.concept_matrix_)

    def get_feature_names_out(self):
        """Returns concept names in an array."""
        return self.classes_

    def fit_transform(self, raw_documents=None, y=None, embeddings=None):
        """Project documents onto the concept vectors.

        Parameters
        ----------
        raw_documents: list[str] or None
            List of documents to project to the concept vectors.
        embeddings: ndarray of shape (n_documents, n_dimensions)
            Document embeddings (has to be created with the same encoder as the concept vectors.)

        Returns
        -------
        document_concept_matrix: ndarray of shape (n_documents, n_dimensions)
            Prevalance of each concept in each document.
        """
        if (raw_documents is None) and (embeddings is None):
            raise ValueError(
                "Either embeddings or raw_documents has to be passed, both are None."
            )
        if embeddings is None:
            embeddings = self.encode_documents(list(raw_documents))
        return embeddings @ self.concept_matrix_.T

    def transform(self, raw_documents=None, embeddings=None):
        """Project documents onto the concept vectors.

        Parameters
        ----------
        raw_documents: list[str] or None
            List of documents to project to the concept vectors.
        embeddings: ndarray of shape (n_documents, n_dimensions)
            Document embeddings (has to be created with the same encoder as the concept vectors.)

        Returns
        -------
        document_concept_matrix: ndarray of shape (n_documents, n_dimensions)
            Prevalance of each concept in each document.
        """
        return self.fit_transform(raw_documents, embeddings=embeddings)


class LateCVP(LateModel):
    """Late-interaction (multi-vector) Concept Vector Projection model from [Lyngbæk et al. (2025)](https://doi.org/10.63744/nVu1Zq5gRkuD)
    Can be used to project document embeddings onto a difference projection vector between positive and negative seed phrases.
    The primary use case is sentiment analysis, and continuous sentiment scores,
    especially for languages where dedicated models are not available.

    Parameters
    ----------
    seeds: (list[str], list[str]) or list of (str, (list[str], list[str]))
        If you want to project to a single concept, then
        a tuple of (list of negative terms, list of positive terms). <br>
        If there are multiple concepts, they should be specified as (name, Seeds) tuples in a list.
        Alternatively, seeds can be an OrderedDict with the names of the concepts being the keys,
        and the tuples of negative and positive seeds as the values.
    encoder: str or SentenceTransformer
        Model to produce document representations, paraphrase-multilingual-mpnet-base-v2 is the default
        per Lyngbæk et al. (2025).
    span_extractor: SpanExtractor, default None
        Object that extracts arbitrary spans from a piece of text.
        These spans will then be used to pool token embeddings.
        This could for instance be a SentenceSeparator.
    include_edge_tokens: bool, default True
        Indicates whether the tokens on the edges of spans (ones that go over the span boundary)
        should be included in the span embedding.
        Ignored when not using a span extractor.
    window_size: int, default None
        Size of the sliding window to average tokens over.
        If None, documents will be represented at a token level.
    step_size: int, default None
        Step size of the window.
        If (step_size == None) or (step_size == window_size), then windows are separate.
        If step_size < window_size, windows will overlap.
        If step_size > window_size, there will be gaps between the windows.
        In this case, we throw a warning, as this is probably unintended behaviour.
    pooling: Callable, default None
        Indicates whether and how to pool document-topic matrices.
        If None, multi-vector topic proportions are returned in a ragged array.
        If Callable, multiple vectors are averaged with the callable in each document.
        You could for example take the mean by specifying `pooling=np.nanmean`.
    trf_kwargs: dict, default None
        Keyword arguments to apply when loading the Encoder model.
    encode_kwargs: dict, default None
        Keyword arguments to apply encoding documents with the encoder.
    """

    def __init__(
        self,
        seeds: Union[Seeds, list[tuple[str, Seeds]], OrderedDict[str, Seeds]],
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        span_extractor: SpanExtractor | Literal["sentence"] | None = None,
        include_edge_tokens: bool = True,
        window_size: int | None = None,
        step_size: int | None = None,
        pooling=None,
        trf_kwargs=None,
        encode_kwargs=None,
    ):
        self.seeds = seeds
        if isinstance(seeds, OrderedDict):
            self._seeds = seeds
        elif (
            (len(seeds) == 2)
            and (isinstance(seeds, tuple))
            and (isinstance(seeds[0][0], str))
        ):
            self._seeds = OrderedDict([("default", seeds)])
        else:
            self._seeds = OrderedDict(seeds)
        self.encoder = encoder
        self.trf_kwargs = trf_kwargs
        self.encode_kwargs = encode_kwargs
        self.load_encoder()
        self.classes_ = np.array([name for name in self._seeds])
        self.concept_matrix_ = []
        self.window_size = window_size
        self.span_extractor = span_extractor
        self.include_edge_tokens = include_edge_tokens
        self.step_size = step_size
        self.pooling = pooling
        for _, (positive, negative) in self._seeds.items():
            positive_emb = self.encode_documents(list(positive))
            negative_emb = self.encode_documents(list(negative))
            cv = np.mean(positive_emb, axis=0) - np.mean(negative_emb, axis=0)
            self.concept_matrix_.append(cv / np.linalg.norm(cv))
        self.concept_matrix_ = np.stack(self.concept_matrix_)

    def get_feature_names_out(self):
        """Returns concept names in an array."""
        return self.classes_

    def fit_transform(
        self,
        raw_documents=None,
        embeddings: np.ndarray | None = None,
        offsets: Offsets | None = None,
    ) -> RaggedArray | np.ndarray:
        if (embeddings is None) or (offsets is None):
            embeddings, offsets = self.encode_late(raw_documents)
        doc_topic_matrix = [emb @ self.concept_matrix_.T for emb in embeddings]
        return self.pool(doc_topic_matrix, offsets)

    def transform(
        self,
        raw_documents=None,
        embeddings: np.ndarray | None = None,
        offsets: Offsets | None = None,
    ) -> RaggedArray | np.ndarray:
        if (embeddings is None) or (offsets is None):
            embeddings, offsets = self.encode_late(raw_documents)
        doc_topic_matrix = [emb @ self.concept_matrix_.T for emb in embeddings]
        return self.pool(doc_topic_matrix, offsets)
