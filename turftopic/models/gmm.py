from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
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
        weight_prior: Literal[
            "dirichlet", "dirichlet_process", None
        ] = "dirichlet",
        gamma: Optional[float] = None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.weight_prior = weight_prior
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = CountVectorizer(min_df=10)
        else:
            self.vectorizer = vectorizer
        if self.weight_prior is not None:
            self.gmm_ = BayesianGaussianMixture(
                n_components=n_components,
                weight_concentration_prior_type="dirichlet_distribution"
                if self.weight_prior == "dirichlet"
                else "dirichlet_process",
                weight_concentration_prior=gamma,
            )
        else:
            self.gmm_ = GaussianMixture(n_components)

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        console = Console()
        with console.status("Fitting model") as status:
            if embeddings is None:
                status.update("Encoding documents")
                embeddings = self.encoder_.encode(raw_documents)
                console.log("Documents encoded.")
            status.update("Extracting terms.")
            document_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            status.update("Fitting mixture model.")
            self.gmm_.fit(embeddings)
            console.log("Mixture model fitted.")
            status.update("Estimating term importances.")
            document_topic_matrix = self.gmm_.predict_proba(embeddings)
            self.components_ = soft_ctf_idf(
                document_topic_matrix, document_term_matrix
            )
            self.weights_ = self.gmm_.weights_
            console.log("Model fitting done.")
        return document_topic_matrix

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        return self.gmm_.predict_proba(embeddings)
