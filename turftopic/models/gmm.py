from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from turftopic.base import ContextualModel, Encoder
from turftopic.soft_ctf_idf import soft_ctf_idf
from turftopic.vectorizer import default_vectorizer


class GMM(ContextualModel):
    """Multivariate Gaussian Mixture Model over document embeddings.
    Models topics as mixture components.

    ```python
    from turftopic import GMM

    corpus: list[str] = ["some text", "more text", ...]

    model = GMM(10, weight_prior="dirichlet_process").fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int
        Number of topics. If you're using priors on the weight,
        feel free to overshoot with this value.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    weight_prior: 'dirichlet', 'dirichlet_process' or None, default 'dirichlet'
        Prior to impose on component weights, if None,
        maximum likelihood is optimized with expectation maximization,
        otherwise variational inference is used.
    gamma: float, default None
        Concentration parameter of the symmetric prior.
        By default 1/n_components is used.
        Ignored when weight_prior is None.

    Attributes
    ----------
    weights_: ndarray of shape (n_components)
        Weights of the different mixture components.
    """

    def __init__(
        self,
        n_components: int,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        weight_prior: Literal["dirichlet", "dirichlet_process", None] = None,
        gamma: Optional[float] = None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.weight_prior = weight_prior
        self.gamma = gamma
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
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
        """Infers topic importances for new documents based on a fitted model.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.

        Returns
        -------
        ndarray of shape (n_dimensions, n_topics)
            Document-topic matrix.
        """
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        return self.gmm_.predict_proba(embeddings)
