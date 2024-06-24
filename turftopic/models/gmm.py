from datetime import datetime
from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.pipeline import Pipeline, make_pipeline

from turftopic.base import ContextualModel, Encoder
from turftopic.dynamic import DynamicTopicModel
from turftopic.feature_importance import soft_ctf_idf
from turftopic.vectorizer import default_vectorizer


class GMM(ContextualModel, DynamicTopicModel):
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
    dimensionality_reduction: TransformerMixin, default None
        Optional dimensionality reduction step before GMM is run.
        This is recommended for very large datasets with high dimensionality,
        as the number of parameters grows vast in the model otherwise.
        We recommend using PCA, as it is a linear solution, and will likely
        result in Gaussian components.
        For even larger datasets you can use IncrementalPCA to reduce
        memory load.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.

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
        dimensionality_reduction: Optional[TransformerMixin] = None,
        weight_prior: Literal["dirichlet", "dirichlet_process", None] = None,
        gamma: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.weight_prior = weight_prior
        self.gamma = gamma
        self.random_state = random_state
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.dimensionality_reduction = dimensionality_reduction
        if self.weight_prior is not None:
            mixture = BayesianGaussianMixture(
                n_components=n_components,
                weight_concentration_prior_type=(
                    "dirichlet_distribution"
                    if self.weight_prior == "dirichlet"
                    else "dirichlet_process"
                ),
                weight_concentration_prior=gamma,
                random_state=self.random_state,
            )
        else:
            mixture = GaussianMixture(
                n_components, random_state=self.random_state
            )
        if dimensionality_reduction is not None:
            self.gmm_ = make_pipeline(dimensionality_reduction, mixture)
        else:
            self.gmm_ = mixture

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
            console.log("Model fitting done.")
        return document_topic_matrix

    @property
    def weights_(self) -> np.ndarray:
        if isinstance(self.gmm_, Pipeline):
            model = self.gmm_.steps[-1][1]
        else:
            model = self.gmm_
        return model.weights_

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

    def fit_transform_dynamic(
        self,
        raw_documents,
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
    ):
        time_labels, self.time_bin_edges = self.bin_timestamps(
            timestamps, bins
        )
        if hasattr(self, "components_"):
            doc_topic_matrix = self.transform(
                raw_documents, embeddings=embeddings
            )
        else:
            doc_topic_matrix = self.fit_transform(
                raw_documents, embeddings=embeddings
            )
        document_term_matrix = self.vectorizer.transform(raw_documents)
        n_comp, n_vocab = self.components_.shape
        n_bins = len(self.time_bin_edges) - 1
        self.temporal_components_ = np.zeros(
            (n_bins, n_comp, n_vocab), dtype=document_term_matrix.dtype
        )
        self.temporal_importance_ = np.zeros((n_bins, n_comp))
        for i_timebin in np.unique(time_labels):
            topic_importances = doc_topic_matrix[time_labels == i_timebin].sum(
                axis=0
            )
            # Normalizing
            topic_importances = topic_importances / topic_importances.sum()
            components = soft_ctf_idf(
                doc_topic_matrix[time_labels == i_timebin],
                document_term_matrix[time_labels == i_timebin],  # type: ignore
            )
            self.temporal_components_[i_timebin] = components
            self.temporal_importance_[i_timebin] = topic_importances
        return doc_topic_matrix
