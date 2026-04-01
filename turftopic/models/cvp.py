from collections import OrderedDict
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from turftopic.base import Encoder
from turftopic.encoders.multimodal import MultimodalEncoder


class ConceptVectorProjection(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        seeds: (
            tuple[list[str], list[str]]
            | list[tuple[[str, tuple[list[str], list[str]]]]]
        ),
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.seeds = seeds
        if (
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
        return self.classes_

    def fit_transform(self, raw_documents=None, y=None, embeddings=None):
        if (raw_documents is None) and (embeddings is None):
            raise ValueError(
                "Either embeddings or raw_documents has to be passed, both are None."
            )
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        return embeddings @ self.concept_matrix_.T

    def transform(self, raw_documents=None, embeddings=None):
        return self.fit_transform(raw_documents, embeddings=embeddings)
