from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.decomposition import FastICA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from turftopic.base import ContextualModel, Encoder
from turftopic.namers.base import TopicNamer
from turftopic.vectorizer import default_vectorizer

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)


class SemanticSignalSeparation(ContextualModel):
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
            Encoder, str
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
            status.update("Decomposing embeddings")
            doc_topic = self.decomposition.fit_transform(self.embeddings)
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
        word_vectors = self.axial_components_.T
        n_topics = self.axial_components_.shape[0]
        axis_vectors = np.eye(n_topics)
        cosine_components = cosine_similarity(axis_vectors, word_vectors)
        return cosine_components

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
        )
        fig = fig.update_layout(
            width=1000,
            height=1000,
            font=dict(family="Times New Roman", color="black", size=21),
            margin=dict(l=5, r=5, t=5, b=5),
        )
        fig = fig.add_hline(y=0, line_color="black", line_width=4)
        fig = fig.add_vline(x=0, line_color="black", line_width=4)
        return fig
