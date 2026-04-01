from collections import OrderedDict
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from turftopic.base import Encoder
from turftopic.encoders.multimodal import MultimodalEncoder

Seeds = tuple[list[str], list[str]]


class ConceptVectorProjection(BaseEstimator, TransformerMixin):
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
    """

    def __init__(
        self,
        seeds: Union[Seeds, list[tuple[str, Seeds]], OrderedDict[str, Seeds]],
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
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
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        self.classes_ = np.array([name for name in self._seeds])
        self.concept_matrix_ = []
        for _, (positive, negative) in self._seeds.items():
            positive_emb = self.encoder_.encode(positive)
            negative_emb = self.encoder_.encode(negative)
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
            embeddings = self.encoder_.encode(raw_documents)
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
