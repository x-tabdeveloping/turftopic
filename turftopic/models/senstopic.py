from datetime import datetime
from functools import partial
from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from turftopic._datamapplot import build_datamapplot
from turftopic.base import ContextualModel, Encoder
from turftopic.dynamic import DynamicTopicModel
from turftopic.encoders.multimodal import MultimodalEncoder
from turftopic.models._snmf import SNMF
from turftopic.multimodal import (
    ImageRepr,
    MultimodalEmbeddings,
    MultimodalModel,
)
from turftopic.optimization import (
    optimize_n_components,
)
from turftopic.vectorizers.default import default_vectorizer

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)


def bic_snmf(
    n_components: int, sparsity: float, X, random_state: int = 42
) -> float:
    decomp = SNMF(
        n_components=n_components,
        sparsity=sparsity,
        random_state=42,
        verbose=False,
        progress_bar=False,
    )
    doc_topic = decomp.fit_transform(X)
    rss = np.square(decomp.reconstruction_err_)
    n_docs, n_dims = X.shape
    # BIC1 from https://pmc.ncbi.nlm.nih.gov/articles/PMC9181460/
    bic1 = np.log(rss) + n_components * (
        (n_docs + n_dims) / (n_docs * n_dims)
    ) * np.log((n_docs * n_dims) / (n_docs + n_dims))
    return bic1


class SensTopic(ContextualModel, DynamicTopicModel, MultimodalModel):
    """Semi-nonnegative Semantic Signal Separation.

    ```python
    from turftopic import SensTopic

    corpus: list[str] = ["some text", "more text", ...]

    model = SensTopic(10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int, default "auto"
        Number of topics.
        If "auto", the number of topics is determined using BIC.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    max_iter: int, default 200
        Maximum number of iterations for S-NMF.
    feature_importance: "axial", "angular" or "combined", default "combined"
        Defines whether the word's position on an axis ('axial'), it's angle to the axis ('angular')
        or their combination ('combined') should determine the word's importance for a topic.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    sparsity: float, default 1
        L1 penalty applied to document-topic proportions.
        Higher values push the model to assign fewer topics to a single document,
        while lower values will distribute topics across documents.
    """

    def __init__(
        self,
        n_components: Union[int, Literal["auto"]] = "auto",
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        max_iter: int = 200,
        feature_importance: Literal[
            "axial", "angular", "combined"
        ] = "combined",
        random_state: Optional[int] = None,
        sparsity: float = 1,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.feature_importance = feature_importance
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        self.validate_encoder()
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.max_iter = max_iter
        self.random_state = random_state
        self.sparsity = sparsity

    def estimate_components(
        self, feature_importance: Literal["axial", "angular", "combined"]
    ) -> np.ndarray:
        """Reestimates components based on the chosen feature_importance method."""
        if feature_importance == "axial":
            self.components_ = self.axial_components_
        elif feature_importance == "angular":
            self.components_ = self.angular_components_
        elif feature_importance == "combined":
            self.components_ = (
                np.square(self.axial_components_) * self.angular_components_
            )
        if hasattr(self, "axial_temporal_components_"):
            if feature_importance == "axial":
                self.temporal_components_ = self.axial_temporal_components_
            elif feature_importance == "angular":
                self.temporal_components_ = self.angular_temporal_components_
            elif feature_importance == "combined":
                self.temporal_components_ = (
                    np.square(self.axial_temporal_components_)
                    * self.angular_temporal_components_
                )
        return self.components_

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        console = Console()
        self.embeddings = embeddings
        with console.status("Fitting model") as status:
            if self.embeddings is None:
                status.update("Encoding documents")
                self.embeddings = self.encoder_.encode(raw_documents)
                console.log("Documents encoded.")
            if self.n_components == "auto":
                status.update("Finding the number of components.")
                self.n_components_ = optimize_n_components(
                    partial(
                        bic_snmf, X=self.embeddings, sparsity=self.sparsity
                    ),
                    min_n=1,
                    verbose=True,
                )
                console.log("N components set at: " + str(self.n_components_))
            else:
                self.n_components_ = self.n_components
            self.decomposition = SNMF(
                self.n_components_,
                max_iter=self.max_iter,
                sparsity=self.sparsity,
                random_state=self.random_state,
            )
            status.update("Decomposing embeddings")
            doc_topic = self.decomposition.fit_transform(self.embeddings, y=y)
            console.log("Decomposition done.")
            status.update("Extracting terms.")
            vocab = self.vectorizer.fit(raw_documents).get_feature_names_out()
            console.log("Term extraction done.")
            if getattr(self, "vocab_embeddings", None) is None:
                status.update("Encoding vocabulary")
                self.vocab_embeddings = self.encoder_.encode(vocab)
            if self.vocab_embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError(
                    NOT_MATCHING_ERROR.format(
                        n_dims=self.embeddings.shape[1],
                        n_word_dims=self.vocab_embeddings.shape[1],
                    )
                )
            console.log("Vocabulary encoded.")
            status.update("Estimating term importances")
            vocab_topic = self.decomposition.transform(self.vocab_embeddings)
            self.axial_components_ = vocab_topic.T
            if self.feature_importance == "axial":
                self.components_ = self.axial_components_
            elif self.feature_importance == "angular":
                self.components_ = self.angular_components_
            elif self.feature_importance == "combined":
                self.components_ = (
                    np.square(self.axial_components_)
                    * self.angular_components_
                )
            self.top_documents = self.get_top_documents(
                raw_documents, document_topic_matrix=doc_topic
            )
            self.document_topic_matrix = doc_topic
            console.log("Model fitting done.")
        return doc_topic

    def fit_transform_multimodal(
        self,
        raw_documents: list[str],
        images: list[ImageRepr],
        y=None,
        embeddings: Optional[MultimodalEmbeddings] = None,
    ) -> np.ndarray:
        self.validate_embeddings(embeddings)
        console = Console()
        self.images = images
        self.multimodal_embeddings = embeddings
        with console.status("Fitting model") as status:
            if self.multimodal_embeddings is None:
                status.update("Encoding documents")
                self.multimodal_embeddings = self.encode_multimodal(
                    raw_documents, images
                )
                console.log("Documents encoded.")
            self.embeddings = self.multimodal_embeddings["document_embeddings"]
            if self.n_components == "auto":
                status.update("Finding the number of components.")
                self.n_components_ = optimize_n_components(
                    partial(
                        bic_snmf, X=self.embeddings, sparsity=self.sparsity
                    ),
                    min_n=1,
                    verbose=True,
                )
                console.log("N components set at: " + str(self.n_components_))
            else:
                self.n_components_ = self.n_components
            self.decomposition = SNMF(
                self.n_components_,
                max_iter=self.max_iter,
                sparsity=self.sparsity,
                random_state=self.random_state,
            )
            status.update("Decomposing embeddings")
            doc_topic = self.decomposition.fit_transform(self.embeddings, y=y)
            console.log("Decomposition done.")
            status.update("Extracting terms.")
            vocab = self.vectorizer.fit(raw_documents).get_feature_names_out()
            console.log("Term extraction done.")
            status.update("Encoding vocabulary")
            self.vocab_embeddings = self.encode_documents(vocab)
            if self.vocab_embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError(
                    NOT_MATCHING_ERROR.format(
                        n_dims=self.embeddings.shape[1],
                        n_word_dims=self.vocab_embeddings.shape[1],
                    )
                )
            console.log("Vocabulary encoded.")
            status.update("Estimating term importances")
            vocab_topic = self.decomposition.transform(self.vocab_embeddings)
            self.axial_components_ = vocab_topic.T
            if self.feature_importance == "axial":
                self.components_ = self.axial_components_
            elif self.feature_importance == "angular":
                self.components_ = self.angular_components_
            elif self.feature_importance == "combined":
                self.components_ = (
                    np.square(self.axial_components_)
                    * self.angular_components_
                )
            console.log("Model fitting done.")
            status.update("Transforming images")
            self.image_topic_matrix = self.transform(
                [], embeddings=self.multimodal_embeddings["image_embeddings"]
            )
            self.top_images = self.collect_top_images(
                images, self.image_topic_matrix
            )
            self.top_documents = self.get_top_documents(
                raw_documents, document_topic_matrix=doc_topic
            )
            console.log("Images transformed")
        return doc_topic

    def fit_transform_dynamic(
        self,
        raw_documents,
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
    ) -> np.ndarray:
        document_topic_matrix = self.fit_transform(
            raw_documents, embeddings=embeddings
        )
        time_labels, self.time_bin_edges = self.bin_timestamps(
            timestamps, bins
        )
        n_comp, n_vocab = self.components_.shape
        n_bins = len(self.time_bin_edges) - 1
        self.axial_temporal_components_ = np.full(
            (n_bins, n_comp, n_vocab),
            np.nan,
            dtype=self.components_.dtype,
        )
        self.temporal_importance_ = np.zeros((n_bins, n_comp))
        # doc_topic = np.dot(X, self.components_.T)
        for i_timebin in np.unique(time_labels):
            topic_importances = document_topic_matrix[
                time_labels == i_timebin
            ].mean(axis=0)
            self.temporal_importance_[i_timebin, :] = topic_importances
            t_doc_topic = document_topic_matrix[time_labels == i_timebin]
            t_embeddings = self.embeddings[time_labels == i_timebin]
            t_components = self.decomposition.fit_timeslice(
                t_embeddings, t_doc_topic
            )
            ax_t = np.maximum(
                self.vocab_embeddings @ np.linalg.pinv(t_components), 0
            )
            self.axial_temporal_components_[i_timebin, :, :] = ax_t.T
        self.estimate_components(self.feature_importance)
        return document_topic_matrix

    @property
    def angular_components_(self):
        """Reweights words based on their angle in ICA-space to the axis
        base vectors.
        """
        if not hasattr(self, "axial_components_"):
            raise NotFittedError("Model has not been fitted yet.")
        word_vectors = self.axial_components_.T
        n_topics = self.axial_components_.shape[0]
        axis_vectors = np.eye(n_topics)
        cosine_components = cosine_similarity(axis_vectors, word_vectors)
        return cosine_components

    @property
    def angular_temporal_components_(self):
        """Reweights words based on their angle in ICA-space to the axis
        base vectors in a dynamic model.
        """
        if not hasattr(self, "axial_temporal_components_"):
            raise NotFittedError("Model has not been fitted dynamically.")
        components = []
        for axial_components in self.axial_temporal_components_:
            word_vectors = axial_components.T
            n_topics = axial_components.shape[0]
            axis_vectors = np.eye(n_topics)
            cosine_components = cosine_similarity(axis_vectors, word_vectors)
            components.append(cosine_components)
        return np.stack(components)

    def plot_topic_decay(self):
        try:
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        doc_topic = self.document_topic_matrix
        topic_proportions = []
        for dt in doc_topic:
            sum_dt = dt.sum()
            if sum_dt > 0:
                dt /= sum_dt
            dt = -np.sort(-dt)
            topic_proportions.append(dt)
        topic_proportions = np.stack(topic_proportions)
        med_prop = np.median(topic_proportions, axis=0)
        upper = np.quantile(topic_proportions, 0.975, axis=0)
        lower = np.quantile(topic_proportions, 0.025, axis=0)
        fig = go.Figure(
            [
                go.Scatter(
                    name="Median",
                    x=np.arange(self.n_components_),
                    y=med_prop,
                    mode="lines",
                    line=dict(color="rgb(31, 119, 180)"),
                ),
                go.Scatter(
                    name="Upper Bound",
                    x=np.arange(self.n_components_),
                    y=upper,
                    mode="lines",
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False,
                ),
                go.Scatter(
                    name="Lower Bound",
                    x=np.arange(self.n_components_),
                    y=lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode="lines",
                    fillcolor="rgba(68, 68, 68, 0.3)",
                    fill="tonexty",
                    showlegend=False,
                ),
            ]
        )
        fig = fig.update_layout(
            template="plotly_white",
            xaxis_title="Topic Rank",
            yaxis_title="Topic Proportion",
            title="Topic Decay",
            font=dict(family="Merriweather", size=16),
        )
        return fig

    def plot_components(
        self, hover_text: Optional[list[str]] = None, **kwargs
    ):
        """Creates an interactive browser plot of the topics in your data using plotly.

        Parameters
        ----------
        hover_text: list of str, optional
            Text to show when hovering over a document.

        Returns
        -------
        plot
            Interactive datamap plot, you can call the `.show()` method to
            display it in your default browser or save it as static HTML using `.write_html()`.
        """
        doc_topic = self.document_topic_matrix
        coords = TSNE(2, metric="cosine").fit_transform(doc_topic)
        labels = np.argmax(doc_topic, axis=1)
        print(np.unique_counts(labels))
        topics_present = np.sort(np.unique(labels))
        names = [self.topic_names[i] for i in topics_present]
        if getattr(self, "topic_descriptions", None) is not None:
            desc = [self.topic_descriptions[i] for i in topics_present]
        else:
            desc = None
        all_words = self.get_top_words()
        keywords = [all_words[i] for i in topics_present]
        fig = build_datamapplot(
            coords,
            labels=labels,
            topic_names=names,
            top_words=keywords,
            hover_text=hover_text,
            topic_descriptions=desc,
            classes=topics_present,
            # Boundaries are unlikely to be very clear
            cluster_boundary_polygons=False,
        )
        return fig

    def plot_components_datamapplot(
        self, hover_text: Optional[list[str]] = None, **kwargs
    ):
        """Creates an interactive browser plot of the topics in your data using datamapplot.

        Parameters
        ----------
        hover_text: list of str, optional
            Text to show when hovering over a document.

        Returns
        -------
        plot
            Interactive datamap plot, you can call the `.show()` method to
            display it in your default browser or save it as static HTML using `.write_html()`.
        """
        doc_topic = self.document_topic_matrix
        coords = TSNE(2, metric="cosine").fit_transform(doc_topic)
        labels = np.argmax(doc_topic, axis=1)
        print(np.unique_counts(labels))
        topics_present = np.sort(np.unique(labels))
        names = [self.topic_names[i] for i in topics_present]
        if getattr(self, "topic_descriptions", None) is not None:
            desc = [self.topic_descriptions[i] for i in topics_present]
        else:
            desc = None
        all_words = self.get_top_words()
        keywords = [all_words[i] for i in topics_present]
        fig = build_datamapplot(
            coords,
            labels=labels,
            topic_names=names,
            top_words=keywords,
            hover_text=hover_text,
            topic_descriptions=desc,
            classes=topics_present,
            # Boundaries are unlikely to be very clear
            cluster_boundary_polygons=False,
        )
        return fig
