import warnings
from datetime import datetime
from typing import Literal, Optional, Union

import numpy as np
from PIL import Image
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.decomposition import FastICA
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from turftopic.base import ContextualModel, Encoder
from turftopic.dynamic import DynamicTopicModel
from turftopic.encoders.multimodal import MultimodalEncoder
from turftopic.multimodal import (
    ImageRepr,
    MultimodalEmbeddings,
    MultimodalModel,
)
from turftopic.namers.base import TopicNamer
from turftopic.vectorizers.default import default_vectorizer

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)


class SemanticSignalSeparation(
    ContextualModel, DynamicTopicModel, MultimodalModel
):
    """Separates the embedding matrix into 'semantic signals' with
    component analysis methods.
    Topics are assumed to be dimensions of semantics.

    ```python
    from turftopic import SemanticSignalSeparation

    corpus: list[str] = ["some text", "more text", ...]

    model = SemanticSignalSeparation(10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int, default 10
        Number of topics.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    decomposition: TransformerMixin, default None
        Custom decomposition method to use.
        Can be an instance of FastICA or PCA, or basically any dimensionality
        reduction method. Has to have `fit_transform` and `fit` methods.
        If not specified, FastICA is used.
    max_iter: int, default 200
        Maximum number of iterations for ICA.
    feature_importance: "axial", "angular" or "combined", default "combined"
        Defines whether the word's position on an axis ('axial'), it's angle to the axis ('angular')
        or their combination ('combined') should determine the word's importance for a topic.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    """

    def __init__(
        self,
        n_components: int = 10,
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        decomposition: Optional[TransformerMixin] = None,
        max_iter: int = 200,
        feature_importance: Literal[
            "axial", "angular", "combined"
        ] = "combined",
        random_state: Optional[int] = None,
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
        if decomposition is None:
            self.decomposition = FastICA(
                n_components, max_iter=max_iter, random_state=random_state
            )
        else:
            self.decomposition = decomposition

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

    @property
    def has_negative_side(self) -> bool:
        return False

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
            status.update("Decomposing embeddings")
            if isinstance(self.decomposition, FastICA) and (y is not None):
                warnings.warn(
                    "y is specified but decomposition method is FastICA, which can't use labels. y will be ignored. Use a metric learning method for semi-supervised S^3."
                )
            doc_topic = self.decomposition.fit_transform(self.embeddings, y=y)
            console.log("Decomposition done.")
            status.update("Extracting terms.")
            vocab = self.vectorizer.fit(raw_documents).get_feature_names_out()
            console.log("Term extraction done.")
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
            status.update("Decomposing embeddings")
            if isinstance(self.decomposition, FastICA) and (y is not None):
                warnings.warn(
                    "Supervisory signal is specified but decomposition method is FastICA. y will be ignored. Use a metric learning method for supervised S^3."
                )
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
            self.negative_images = self.collect_top_images(
                images, self.image_topic_matrix, negative=True
            )
            console.log("Images transformed")
        return doc_topic

    def plot_topics_with_images(self, n_columns: int = 3, grid_size: int = 4):
        try:
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        fig = go.Figure()
        width, height = 1200, 1200
        scale_factor = 0.25
        w, h = width * scale_factor, height * scale_factor
        padding = 10
        figure_height = (h + padding) * self.n_components
        figure_width = (w + padding) * 2
        fig = fig.add_trace(
            go.Scatter(
                x=[0, figure_width],
                y=[0, figure_height],
                mode="markers",
                marker_opacity=0,
            )
        )
        vocab = self.get_vocab()
        for i, component in enumerate(self.components_):
            positive = vocab[np.argsort(-component)[:7]]
            negative = vocab[np.argsort(component)[:7]]
            pos_image = self._image_grid(
                self.top_images[i],
                (width, height),
                grid_size=(grid_size, grid_size),
            )
            neg_image = self._image_grid(
                self.negative_images[i],
                (width, height),
                grid_size=(grid_size, grid_size),
            )
            x0 = 0
            y0 = (h + padding) * (self.n_components - i)
            fig = fig.add_layout_image(
                dict(
                    x=x0,
                    sizex=w,
                    y=y0,
                    sizey=h,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=pos_image,
                ),
            )
            fig.add_annotation(
                x=(w / 2),
                y=(h + padding) * (self.n_components - i) - (h / 2),
                text="<b> " + "<br> ".join(positive),
                font=dict(
                    size=16,
                    family="Times New Roman",
                    color="white",
                ),
                bgcolor="rgba(0,0,255, 0.5)",
            )
            x0 = (w + padding) * 1
            fig = fig.add_layout_image(
                dict(
                    x=x0,
                    sizex=w,
                    y=y0,
                    sizey=h,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=neg_image,
                ),
            )
            fig.add_annotation(
                x=(w + padding) + (w / 2),
                y=(h + padding) * (self.n_components - i) - (h / 2),
                text="<b> " + "<br> ".join(negative),
                font=dict(
                    size=16,
                    family="Times New Roman",
                    color="white",
                ),
                bgcolor="rgba(255,0,0, 0.5)",
            )
        fig = fig.update_xaxes(visible=False, range=[0, figure_width])
        fig = fig.update_yaxes(
            visible=False,
            range=[0, figure_height],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x",
        )
        fig = fig.update_layout(
            width=figure_width,
            height=figure_height,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        return fig

    def _rename_automatic(self, namer: TopicNamer) -> list[str]:
        """Names topics with a topic namer in the model.

        Parameters
        ----------
        namer: TopicNamer
            A Topic namer model to name topics with.

        Returns
        -------
        list[str]
            List of topic names.
        """
        positive_names = namer.name_topics(self._top_terms())
        negative_names = namer.name_topics(self._top_terms(positive=False))
        names = []
        for positive, negative in zip(positive_names, negative_names):
            names.append(f"{positive}/{negative}")
        self.topic_names_ = names
        return self.topic_names_

    def refit_transform(
        self,
        n_components: Optional[int] = None,
        max_iter: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Refits model with the given parameters.
        This is significantly faster than fitting a new model from scratch.

        Parameters
        ----------
        n_components: int, default None
            Number of topics.
        max_iter: int, default None
            Maximum number of iterations for ICA.
        random_state: int, default None
            Random state to use so that results are exactly reproducible.

        Returns
        -------
        ndarray of shape (n_documents, n_topics)
            Document-topic matrix.
        """
        self.n_components = n_components
        self.topic_names_ = None
        n_components = (
            n_components if n_components is not None else self.n_components
        )
        max_iter = max_iter if max_iter is not None else self.max_iter
        random_state = (
            random_state if random_state is not None else self.random_state
        )
        self.decomposition = FastICA(
            n_components, max_iter=max_iter, random_state=random_state
        )
        console = Console()
        with console.status("Refitting model") as status:
            status.update("Decomposing embeddings")
            doc_topic = self.decomposition.fit_transform(self.embeddings)
            console.log("Decomposition done.")
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
        whitened_embeddings = np.copy(self.embeddings)
        if getattr(self.decomposition, "whiten"):
            whitened_embeddings -= self.decomposition.mean_
        # doc_topic = np.dot(X, self.components_.T)
        for i_timebin in np.unique(time_labels):
            topic_importances = document_topic_matrix[
                time_labels == i_timebin
            ].mean(axis=0)
            self.temporal_importance_[i_timebin, :] = topic_importances
            t_doc_topic = document_topic_matrix[time_labels == i_timebin]
            t_embeddings = whitened_embeddings[time_labels == i_timebin]
            linreg = LinearRegression().fit(t_embeddings, t_doc_topic)
            self.axial_temporal_components_[i_timebin, :, :] = np.dot(
                self.vocab_embeddings, linreg.coef_.T
            ).T
        self.estimate_components(self.feature_importance)
        return document_topic_matrix

    def refit_transform_dynamic(
        self,
        timestamps: list[datetime],
        bins: Union[int, list[datetime]] = 10,
        n_components: Optional[int] = None,
        max_iter: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Refits $S^3$ to be a dynamic model."""
        document_topic_matrix = self.refit_transform(
            n_components=n_components,
            max_iter=max_iter,
            random_state=random_state,
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
        whitened_embeddings = np.copy(self.embeddings)
        if getattr(self.decomposition, "whiten"):
            whitened_embeddings -= self.decomposition.mean_
        # doc_topic = np.dot(X, self.components_.T)
        for i_timebin in np.unique(time_labels):
            topic_importances = document_topic_matrix[
                time_labels == i_timebin
            ].mean(axis=0)
            self.temporal_importance_[i_timebin, :] = topic_importances
            t_doc_topic = document_topic_matrix[time_labels == i_timebin]
            t_embeddings = whitened_embeddings[time_labels == i_timebin]
            linreg = LinearRegression().fit(t_embeddings, t_doc_topic)
            self.axial_temporal_components_[i_timebin, :, :] = np.dot(
                self.vocab_embeddings, linreg.coef_.T
            ).T
        self.estimate_components(self.feature_importance)
        return document_topic_matrix

    def refit(
        self,
        n_components: Optional[int] = None,
        max_iter: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Refits model with the given parameters.
        This is significantly faster than fitting a new model from scratch.

        Parameters
        ----------
        n_components: int, default None
            Number of topics.
        max_iter: int, default None
            Maximum number of iterations for ICA.
        random_state: int, default None
            Random state to use so that results are exactly reproducible.

        Returns
        -------
        Refitted model.
        """
        self.refit_transform(n_components, max_iter, random_state)
        return self

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
        return self.decomposition.transform(embeddings)

    def print_topics(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        show_negative: bool = True,
    ):
        super().print_topics(top_k, show_scores, show_negative)

    def export_topics(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        show_negative: bool = True,
        format: str = "csv",
    ) -> str:
        return super().export_topics(top_k, show_scores, show_negative, format)

    def print_representative_documents(
        self,
        topic_id,
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: bool = True,
    ):
        super().print_representative_documents(
            topic_id,
            raw_documents,
            document_topic_matrix,
            top_k,
            show_negative,
        )

    def export_representative_documents(
        self,
        topic_id,
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: bool = True,
        format: str = "csv",
    ):
        return super().export_representative_documents(
            topic_id,
            raw_documents,
            document_topic_matrix,
            top_k,
            show_negative,
            format,
        )

    def concept_compass(
        self, topic_x: Union[int, str], topic_y: Union[str, int]
    ):
        """[DEPRECATED] will be removed in version 1.0.0.
        See plot_concept_compass().
        """
        warnings.warn(
            "concept_compass() is deprecated and will be removed in version 1.0.0. Use plot_concept_compass() instead."
        )
        return self.plot_concept_compass(topic_x, topic_y)

    def plot_concept_compass(
        self, topic_x: Union[int, str], topic_y: Union[str, int]
    ):
        """Display a compass of concepts along two semantic axes.
        In order for the plot to be concise and readable, terms are randomly selected on
        a grid of the two topics.

        Parameters
        ----------
        topic_x: int or str
            Index or name of the topic to display on the X axis.
        topic_y: int or str
            Index or name of the topic to display on the Y axis.

        Returns
        -------
        go.Figure
            Plotly interactive plot of the concept compass.
        """
        try:
            import plotly.express as px
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        if isinstance(topic_x, str):
            try:
                topic_x = list(self.topic_names).index(topic_x)
            except ValueError as e:
                raise ValueError(
                    f"{topic_x} is not a valid topic name or index."
                ) from e
        if isinstance(topic_y, str):
            try:
                topic_y = list(self.topic_names).index(topic_y)
            except ValueError as e:
                raise ValueError(
                    f"{topic_y} is not a valid topic name or index."
                ) from e
        x = self.axial_components_[topic_x]
        y = self.axial_components_[topic_y]
        vocab = self.get_vocab()
        points = np.array(list(zip(x, y)))
        xx, yy = np.meshgrid(
            np.linspace(np.min(x), np.max(x), 20),
            np.linspace(np.min(y), np.max(y), 20),
        )
        coords = np.array(list(zip(np.ravel(xx), np.ravel(yy))))
        coords = coords + np.random.default_rng(0).normal(
            [0, 0], [0.1, 0.1], size=coords.shape
        )
        dist = euclidean_distances(coords, points)
        idxs = np.argmin(dist, axis=1)
        fig = px.scatter(
            x=x[idxs],
            y=y[idxs],
            text=vocab[idxs],
            template="plotly_white",
        )
        fig = fig.update_traces(
            mode="text", textfont_color="black", marker=dict(color="black")
        ).update_layout(
            xaxis_title=f"{self.topic_names[topic_x]}",
            yaxis_title=f"{self.topic_names[topic_y]}",
            font=dict(family="Roboto Mono"),
        )
        fig = fig.update_layout(
            font=dict(family="Roboto Mono", color="black", size=21),
            margin=dict(l=5, r=5, t=5, b=5),
        )
        fig = fig.add_hline(y=0, line_color="black", line_width=4)
        fig = fig.add_vline(x=0, line_color="black", line_width=4)
        return fig

    def plot_image_compass(
        self, topic_x: Union[str, int], topic_y: Union[str, int]
    ):
        try:
            import plotly.express as px
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        top_images = getattr(self, "top_images", None)
        if top_images is None:
            raise ValueError(
                "Topic model is not multimodal. Can't plot image compass."
            )
        if isinstance(topic_x, str):
            try:
                topic_x = list(self.topic_names).index(topic_x)
            except ValueError as e:
                raise ValueError(
                    f"{topic_x} is not a valid topic name or index."
                ) from e
        if isinstance(topic_y, str):
            try:
                topic_y = list(self.topic_names).index(topic_y)
            except ValueError as e:
                raise ValueError(
                    f"{topic_y} is not a valid topic name or index."
                ) from e
        x = self.image_topic_matrix[:, topic_x]
        y = self.image_topic_matrix[:, topic_y]
        points = np.array(list(zip(x, y)))
        xx, yy = np.meshgrid(
            np.linspace(np.min(x), np.max(x), 8),
            np.linspace(np.min(y), np.max(y), 8),
        )
        coords = np.array(list(zip(np.ravel(xx), np.ravel(yy))))
        dist = euclidean_distances(coords, points)
        idxs = np.argmin(dist, axis=1)
        fig = px.scatter(
            x=x[idxs],
            y=y[idxs],
            template="plotly_white",
        )
        sizex = (max(x) - min(x)) / 10
        sizey = (max(y) - min(y)) / 10
        for i in np.unique(idxs):
            fig.add_layout_image(
                dict(
                    name=f"image{i}",
                    source=self.images[i],
                    x=x[i],
                    y=y[i],
                    xref="x",
                    yref="y",
                    xanchor="right",
                    yanchor="top",
                    layer="above",
                    sizex=sizex,
                    sizey=sizey,
                )
            )
        fig = fig.update_traces(
            mode="markers", textfont_color="black", marker=dict(opacity=0)
        ).update_layout(
            xaxis_title=f"{self.topic_names[topic_x]}",
            yaxis_title=f"{self.topic_names[topic_y]}",
            font=dict(family="Roboto Mono"),
        )
        fig = fig.update_layout(
            font=dict(family="Roboto Mono", color="black", size=21),
            margin=dict(l=5, r=5, t=5, b=5),
        )
        fig = fig.update_xaxes(range=[min(x), max(x)])
        fig = fig.update_yaxes(range=[min(y), max(y)])
        return fig

    def plot_topics_over_time(self, top_k: int = 6):
        try:
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        fig = go.Figure()
        vocab = self.get_vocab()
        n_topics = self.temporal_components_.shape[1]
        try:
            topic_names = self.topic_names
        except AttributeError:
            topic_names = [f"Topic {i}" for i in range(n_topics)]
        for i_topic, topic_imp_t in enumerate(self.temporal_importance_.T):
            component_over_time = self.temporal_components_[:, i_topic, :]
            name_over_time = []
            for component, importance in zip(component_over_time, topic_imp_t):
                if importance < 0:
                    component = -component
                top = np.argpartition(-component, top_k)[:top_k]
                values = component[top]
                if np.all(values == 0) or np.all(np.isnan(values)):
                    name_over_time.append("<not present>")
                    continue
                top = top[np.argsort(-values)]
                name_over_time.append(", ".join(vocab[top]))
            times = self.time_bin_edges[:-1]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=topic_imp_t,
                    mode="markers+lines",
                    text=name_over_time,
                    name=topic_names[i_topic],
                    hovertemplate="<b>%{text}</b>",
                    marker=dict(
                        line=dict(width=2, color="black"),
                        size=14,
                    ),
                    line=dict(width=3),
                )
            )
        fig.add_hline(y=0, line_dash="dash", opacity=0.5)
        fig.update_layout(
            template="plotly_white",
            hoverlabel=dict(font_size=16, bgcolor="white"),
            hovermode="x",
            font=dict(family="Roboto Mono"),
        )
        fig.update_xaxes(title="Time Slice Start")
        fig.update_yaxes(title="Topic Importance")
        return fig

    def _topics_over_time(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        date_format: str = "%Y %m %d",
    ) -> list[list[str]]:
        temporal_components = self.temporal_components_
        slices = self.get_time_slices()
        slice_names = []
        for start_dt, end_dt in slices:
            start_str = start_dt.strftime(date_format)
            end_str = end_dt.strftime(date_format)
            slice_names.append(f"{start_str} - {end_str}")
        n_topics = self.temporal_components_.shape[1]
        try:
            topic_names = self.topic_names
        except AttributeError:
            topic_names = [f"Topic {i}" for i in range(n_topics)]
        columns = []
        rows = []
        columns.append("Time Slice")
        for topic in topic_names:
            columns.append(topic)
        for slice_name, components, weights in zip(
            slice_names, temporal_components, self.temporal_importance_
        ):
            fields = []
            fields.append(slice_name)
            vocab = self.get_vocab()
            for component, weight in zip(components, weights):
                if np.all(component == 0) or np.all(np.isnan(component)):
                    fields.append("Topic not present.")
                    continue
                if weight < 0:
                    component = -component
                top = np.argpartition(-component, top_k)[:top_k]
                importance = component[top]
                top = top[np.argsort(-importance)]
                top = top[importance != 0]
                scores = component[top]
                words = vocab[top]
                if show_scores:
                    concat_words = ", ".join(
                        [
                            f"{word}({importance:.2f})"
                            for word, importance in zip(words, scores)
                        ]
                    )
                else:
                    concat_words = ", ".join([word for word in words])
                fields.append(concat_words)
            rows.append(fields)
        return [columns, *rows]
