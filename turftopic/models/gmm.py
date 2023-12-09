from typing import Literal, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from turftopic.base import ContextualModel
from turftopic.soft_ctf_idf import soft_ctf_idf


class MixtureTopicModel(ContextualModel):
    def __init__(
        self,
        n_components: int,
        encoder: Union[
            SentenceTransformer, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        prior: Literal["dirichlet", "dirichlet_process", None] = None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.prior = prior
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = CountVectorizer(min_df=10)
        else:
            self.vectorizer = vectorizer
        if self.prior is not None:
            self.gmm_ = BayesianGaussianMixture(
                n_components=n_components,
                weight_concentration_prior_type="dirichlet_distribution"
                if self.prior == "dirichlet"
                else "dirichlet_process",
            )
        else:
            self.gmm_ = GaussianMixture(n_components)

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        document_term_matrix = self.vectorizer.fit_transform(raw_documents)
        self.gmm_.fit(embeddings)
        document_topic_matrix = self.gmm_.predict_proba(embeddings)
        self.components_ = soft_ctf_idf(
            document_topic_matrix, document_term_matrix
        )
        self.weights_ = self.gmm_.weights_
        return document_topic_matrix

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        return self.gmm_.predict_proba(embeddings)
