import tempfile
import time
import typing
import warnings
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import HDBSCAN
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, normalize, scale

from turftopic.base import ContextualModel, Encoder
from turftopic.dynamic import DynamicTopicModel
from turftopic.encoders.multimodal import MultimodalEncoder
from turftopic.feature_importance import (
    bayes_rule,
    cluster_centroid_distance,
    ctf_idf,
    fighting_words,
    linear_classifier,
    soft_ctf_idf,
)
from turftopic.models._hierarchical_clusters import (
    VALID_LINKAGE_METHODS,
    ClusterNode,
    LinkageMethod,
)
from turftopic.multimodal import (
    Image,
    ImageRepr,
    MultimodalEmbeddings,
    MultimodalModel,
)
from turftopic.types import VALID_DISTANCE_METRICS, DistanceMetric
from turftopic.utils import safe_binarize
from turftopic.vectorizers.default import default_vectorizer

integer_message = """
You tried to pass an integer to ClusteringTopicModel as its first argument.
We assume you tried to specify the number of topics.
Since in ClusteringTopicModel the clustering model determines the number of topics,
and this process may be automatic, you have to pass along a clustering model
where the number of clusters is predefined.

For instance: ClusteringTopicModel(clustering=KMeans(10))

Alternatively you can reduce the number of topics in the model by specifying
the desired reduced number on initialization.

ClusteringTopicModel(n_reduce_to=10)
"""

WordImportance = Literal[
    "soft-c-tf-idf",
    "c-tf-idf",
    "centroid",
    "bayes",
    "linear",
    "fighting-words",
]
VALID_WORD_IMPORTANCE = list(typing.get_args(WordImportance))

TopicRepresentation = Literal[
    "component",
    "centroid",
]
VALID_TOPIC_REPRESENTATIONS = list(typing.get_args(TopicRepresentation))

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)


def factorize_labels(labels: Iterable[Any]) -> np.ndarray:
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    for i, _class in enumerate(le.classes_):
        if (str(_class) == -1) or (not np.isfinite(_class)):
            labels[labels == i] = -1
    return labels


def calculate_topic_vectors(
    cluster_labels: np.ndarray,
    embeddings: np.ndarray,
    time_index: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculates topic centroids."""
    centroids = []
    unique_labels = np.unique(cluster_labels)
    unique_labels = np.sort(unique_labels)
    for label in unique_labels:
        label_index = cluster_labels == label
        if time_index is not None:
            label_index = label_index * time_index
        label_embeddings = embeddings[label_index]
        centroid = np.mean(label_embeddings, axis=0)
        centroids.append(centroid)
    centroids = np.stack(centroids)
    return centroids


def build_tsne(*args, **kwargs):
    try:
        from openTSNE import TSNE

        class OpenTSNEWrapper(TSNE):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def fit_transform(self, X: np.ndarray, y=None):
                return super().fit(X)

            def fit(self, X: np.ndarray, y=None):
                self.fit_transform(X, y)
                return self

        return OpenTSNEWrapper(*args, **kwargs)

    except ModuleNotFoundError:
        from sklearn.manifold import TSNE

        warnings.warn(
            """OpenTSNE is not installed, default scikit-learn implementation will be used.
        Your model could potentially run orders of magnitudes faster by installing openTSNE.
        """
        )
        return TSNE(*args, **kwargs)


class ClusteringTopicModel(
    ContextualModel, ClusterMixin, DynamicTopicModel, MultimodalModel
):
    """Topic models, which assume topics to be clusters of documents
    in semantic space.
    Models also include a dimensionality reduction step to aid clustering.

    ```python
    from turftopic import ClusteringTopicModel
    from sklearn.cluster import HDBSCAN
    import umap

    corpus: list[str] = ["some text", "more text", ...]

    # Construct a Top2Vec-like model
    model = ClusteringTopicModel(
        dimensionality_reduction=umap.UMAP(5),
        clustering=HDBSCAN(),
        feature_importance="centroid"
    ).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    dimensionality_reduction: TransformerMixin, default None
        Dimensionality reduction step to run before clustering.
        Defaults to TSNE with cosine distance.
        To imitate the behavior of BERTopic or Top2Vec you should use UMAP.
    clustering: ClusterMixin, default None
        Clustering method to use for finding topics.
        Defaults to OPTICS with 25 minimum cluster size.
        To imitate the behavior of BERTopic or Top2Vec you should use HDBSCAN.
    feature_importance: WordImportance, default 'soft-c-tf-idf'
        Method for estimating term importances.
        'centroid' uses distances from cluster centroid similarly
        to Top2Vec.
        'c-tf-idf' uses BERTopic's c-tf-idf.
        'soft-c-tf-idf' uses Soft c-TF-IDF from GMM, the results should
        be very similar to 'c-tf-idf'.
        'bayes' uses Bayes' rule.
        'linear' calculates most predictive directions in embedding space and projects
        words onto them.
        'fighting-words' calculates word importances based on the Fighting Words
        algorithm from Monroe et al.
    n_reduce_to: int, default None
        Number of topics to reduce topics to.
        The specified reduction method will be used to merge them.
        By default, topics are not merged.
    reduction_method: LinkageMethod, default 'average'
        Method used for hierarchically merging topics.
        Could be "smallest", which is Top2Vec's default merging strategy, or
        any of the linkage methods listed in [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
    reduction_distance_metric: DistanceMetric, default 'cosine'
        Distance metric to use for hierarchical topic reduction.
    reduction_topic_representation: {'component', 'centroid'}, default 'component'
        Topic representation used for hierarchical clustering.
        If 'component' the topic-word importance scores will be used as topic vectors, (this is how it's done in BERTopic)
        if 'centroid' the centroid vectors of clusters will be used as topic vectors (Top2Vec).
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    """

    def __init__(
        self,
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        dimensionality_reduction: Optional[TransformerMixin] = None,
        clustering: Optional[ClusterMixin] = None,
        feature_importance: WordImportance = "soft-c-tf-idf",
        n_reduce_to: Optional[int] = None,
        reduction_method: LinkageMethod = "average",
        reduction_distance_metric: DistanceMetric = "cosine",
        reduction_topic_representation: TopicRepresentation = "component",
        random_state: Optional[int] = None,
    ):
        self.encoder = encoder
        self.random_state = random_state
        if feature_importance not in VALID_WORD_IMPORTANCE:
            raise ValueError(
                f"feature_importance must be one of {VALID_WORD_IMPORTANCE} got {feature_importance} instead."
            )
        if reduction_method not in VALID_LINKAGE_METHODS:
            raise ValueError(
                f"Topic reduction method has to be one of: {VALID_LINKAGE_METHODS}, but got {reduction_method} instead."
            )
        if reduction_distance_metric not in VALID_DISTANCE_METRICS:
            raise ValueError(
                f"Distance metric should be one of: {VALID_DISTANCE_METRICS}, but got {reduction_distance_metric} instead."
            )
        if reduction_topic_representation not in VALID_TOPIC_REPRESENTATIONS:
            raise ValueError(
                f"Topic representation should be one of: {VALID_TOPIC_REPRESENTATIONS}, but got {reduction_topic_representation} instead."
            )
        if isinstance(encoder, int):
            raise TypeError(integer_message)
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        self.validate_encoder()
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        if clustering is None:
            self.clustering = HDBSCAN(
                min_samples=10,
                min_cluster_size=25,
            )
        else:
            self.clustering = clustering
        if dimensionality_reduction is None:
            self.dimensionality_reduction = build_tsne(
                n_components=2,
                metric="cosine",
                perplexity=15,
                random_state=random_state,
            )
        else:
            self.dimensionality_reduction = dimensionality_reduction
        self.feature_importance = feature_importance
        self.reduction_distance_metric = reduction_distance_metric
        self.reduction_topic_representation = reduction_topic_representation
        self.n_reduce_to = n_reduce_to
        self.reduction_method = reduction_method

    @property
    def topic_representations(self) -> np.ndarray:
        if self.reduction_topic_representation == "component":
            return self.components_
        else:
            return self._calculate_topic_vectors()

    def _calculate_topic_vectors(
        self,
        is_in_slice: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if classes is None:
            classes = self.classes_
        if embeddings is None:
            embeddings = self.embeddings
        if labels is None:
            labels = self.labels_
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        n_topics = len(classes)
        n_dims = embeddings.shape[1]
        topic_vectors = np.full((n_topics, n_dims), np.nan)
        for label in np.unique(labels):
            doc_idx = labels == label
            if is_in_slice is not None:
                doc_idx = doc_idx & is_in_slice
            topic_vectors[label_to_idx[label], :] = np.mean(
                embeddings[doc_idx], axis=0
            )
        return topic_vectors

    def estimate_components(
        self, feature_importance: Optional[WordImportance] = None
    ) -> np.ndarray:
        """Estimates feature importances based on a fitted clustering.

        Parameters
        ----------
        feature_importance: WordImportance, default None
            Method for estimating term importances.
            'centroid' uses distances from cluster centroid similarly
            to Top2Vec.
            'c-tf-idf' uses BERTopic's c-tf-idf.
            'soft-c-tf-idf' uses Soft c-TF-IDF from GMM, the results should
            be very similar to 'c-tf-idf'.
            'bayes' uses Bayes' rule.
            'linear' calculates most predictive directions in embedding space and projects
            words onto them.
            'fighting-words' calculates word importances based on the Fighting Words
            algorithm from Monroe et al.

        Returns
        -------
        ndarray of shape (n_components, n_vocab)
            Topic-term matrix.
        """
        if feature_importance is not None:
            if feature_importance not in VALID_WORD_IMPORTANCE:
                raise ValueError(
                    f"feature_importance must be one of {VALID_WORD_IMPORTANCE} got {feature_importance} instead."
                )
            self.feature_importance = feature_importance
        self.hierarchy.estimate_components()
        doc_topic_matrix = safe_binarize(self.labels_, classes=self.classes_)
        if feature_importance == "c-tf-idf":
            _, self._idf_diag = ctf_idf(
                doc_topic_matrix,
                self.doc_term_matrix,
                return_idf=True,
            )
        if feature_importance == "soft-c-tf-idf":
            _, self._idf_diag = soft_ctf_idf(
                doc_topic_matrix,
                self.doc_term_matrix,
                return_idf=True,
            )
        return self.components_

    def reduce_topics(
        self,
        n_reduce_to: int,
        reduction_method: Optional[LinkageMethod] = None,
        metric: Optional[DistanceMetric] = None,
    ) -> np.ndarray:
        """Reduces the clustering to the desired amount with the given method.

        Parameters
        ----------
        n_reduce_to: int, default None
            Number of topics to reduce topics to.
            The specified reduction method will be used to merge them.
            By default, topics are not merged.
        reduction_method: LinkageMethod, default None
            Method used for hierarchically merging topics.
            Could be "smallest", which is Top2Vec's default merging strategy, or
            any of the linkage methods listed in [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
        reduction_distance_metric: DistanceMetric, default None
            Distance metric to use for hierarchical topic reduction.

        Returns
        -------
        ndarray of shape (n_documents)
            New cluster labels for documents.
        """
        if not hasattr(self, "original_labels_"):
            self.original_labels_ = self.labels_
            self.original_names_ = self.topic_names
            self.original_classes_ = self.classes_
        if reduction_method is None:
            reduction_method = self.reduction_method
        if metric is None:
            metric = self.reduction_distance_metric
        self.hierarchy.reduce_topics(
            n_reduce_to, method=reduction_method, metric=metric
        )
        return self.labels_

    def reset_topics(self):
        """Resets topics to the original cllustering."""
        original_labels = getattr(self, "original_labels_", None)
        if original_labels is None:
            warnings.warn("Topics have never been reduced, nothing to reset.")
        else:
            self.hierarchy = ClusterNode.create_root(
                self, labels=self.original_labels_
            )
            self.topic_names_ = self.original_names_

    @property
    def classes_(self):
        try:
            return self.hierarchy.classes_
        except AttributeError as e:
            raise AttributeError(
                "Model has not been fitted yet, and doesn't have classes_"
            ) from e

    @property
    def components_(self):
        try:
            return self.hierarchy.components_
        except AttributeError as e:
            raise AttributeError(
                "Model has not been fitted yet, and doesn't have components_"
            ) from e

    @property
    def labels_(self):
        try:
            return self.hierarchy.labels_
        except AttributeError as e:
            raise AttributeError(
                "Model has not been fitted yet, and doesn't have labels_"
            ) from e

    @property
    def document_topic_matrix(self):
        return safe_binarize(self.labels_, classes=self.classes_)

    def join_topics(
        self, to_join: Sequence[int], joint_id: Optional[int] = None
    ):
        """Joins the given topics in the cluster hierarchy to a single topic.

        Parameters
        ----------
        to_join: Sequence of int
            Topics to join together by ID.
        joint_id: int, default None
            New ID for the joint cluster.
            Default is the smallest ID of the topics to join.
        """
        self.hierarchy.join_topics(to_join, joint_id=joint_id)

    def fit_predict(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fits model and predicts cluster labels for all given documents.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        y: None
            Ignored, when the dimensionality reduction is TSNE (the default),
            in case of a dimensionality reduction that can utilize labels,
            you can pass labels to the model to inform the clustering process.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.

        Returns
        -------
        ndarray of shape (n_documents)
            Cluster label for all documents (-1 for outliers)
        """
        console = Console()
        with console.status("Fitting model") as status:
            if embeddings is None:
                status.update("Encoding documents")
                embeddings = self.encode_documents(raw_documents)
                console.log("Encoding done.")
            self.embeddings = embeddings
            status.update("Extracting terms")
            self.doc_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            status.update("Reducing Dimensionality")
            # If y is specified, we pass it to the dimensionality
            # reduction method as supervisory signal
            if y is not None:
                y = factorize_labels(y)
            self.reduced_embeddings = (
                self.dimensionality_reduction.fit_transform(embeddings, y=y)
            )
            console.log("Dimensionality reduction done.")
            status.update("Clustering documents")
            labels = self.clustering.fit_predict(self.reduced_embeddings)
            console.log("Clustering done.")
            status.update("Estimating parameters.")
            # Initializing hierarchy
            self.hierarchy = ClusterNode.create_root(self, labels=labels)
            doc_topic_matrix = safe_binarize(
                self.labels_, classes=self.classes_
            )
            if self.feature_importance == "c-tf-idf":
                _, self._idf_diag = ctf_idf(
                    doc_topic_matrix,
                    self.doc_term_matrix,
                    return_idf=True,
                )
            if self.feature_importance == "soft-c-tf-idf":
                _, self._idf_diag = soft_ctf_idf(
                    doc_topic_matrix,
                    self.doc_term_matrix,
                    return_idf=True,
                )
            console.log("Parameter estimation done.")
            if self.n_reduce_to is not None:
                n_topics = self.classes_.shape[0]
                status.update(
                    f"Reducing topics from {n_topics} to {self.n_reduce_to}"
                )
                self.reduce_topics(
                    self.n_reduce_to,
                    self.reduction_method,
                    self.reduction_distance_metric,
                )
                console.log(
                    f"Topic reduction done from {n_topics} to {self.n_reduce_to}."
                )
        console.log("Model fitting done.")
        return self.labels_

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        self.fit_predict(raw_documents, y, embeddings)
        embeddings = (
            embeddings
            if embeddings is not None
            else getattr(self, "embeddings", None)
        )
        document_topic_matrix = self.transform(
            raw_documents, embeddings=embeddings
        )
        return document_topic_matrix

    def fit_transform_multimodal(
        self,
        raw_documents: list[str],
        images: list[ImageRepr],
        y=None,
        embeddings: Optional[MultimodalEmbeddings] = None,
    ) -> np.ndarray:
        self.validate_embeddings(embeddings)
        self.multimodal_embeddings = embeddings
        if self.multimodal_embeddings is None:
            self.multimodal_embeddings = self.encode_multimodal(
                raw_documents, images
            )
        doc_topic_matrix = self.fit_transform(
            raw_documents,
            embeddings=self.multimodal_embeddings["document_embeddings"],
            y=y,
        )
        self.image_topic_matrix = self.transform(
            raw_documents,
            embeddings=self.multimodal_embeddings["image_embeddings"],
        )
        self.top_images: list[list[Image.Image]] = self.collect_top_images(
            images, self.image_topic_matrix
        )
        return doc_topic_matrix

    def estimate_temporal_components(
        self,
        time_labels,
        time_bin_edges,
        feature_importance: Optional[WordImportance] = None,
    ) -> np.ndarray:
        """Estimates temporal components based on a fitted topic model.

        Parameters
        ----------
        feature_importance: WordImportance, default None
            Method for estimating term importances.
            'centroid' uses distances from cluster centroid similarly
            to Top2Vec.
            'c-tf-idf' uses BERTopic's c-tf-idf.
            'soft-c-tf-idf' uses Soft c-TF-IDF from GMM, the results should
            be very similar to 'c-tf-idf'.
            'bayes' uses Bayes' rule.
            'linear' calculates most predictive directions in embedding space and projects
            words onto them.

        Returns
        -------
        ndarray of shape (n_time_bins, n_components, n_vocab)
            Temporal topic-term matrix.
        """
        if getattr(self, "components_", None) is None:
            raise NotFittedError(
                "The model has not been fitted yet, please fit the model before estimating temporal components."
            )
        if feature_importance is None:
            feature_importance = self.feature_importance
        n_comp, n_vocab = self.components_.shape
        self.time_bin_edges = time_bin_edges
        n_bins = len(self.time_bin_edges) - 1
        self.temporal_components_ = np.full(
            (n_bins, n_comp, n_vocab),
            np.nan,
            dtype=self.components_.dtype,
        )
        self.temporal_importance_ = np.zeros((n_bins, n_comp))
        for i_timebin in np.unique(time_labels):
            topic_importances = self.document_topic_matrix[
                time_labels == i_timebin
            ].sum(axis=0)
            if not topic_importances.sum() == 0:
                topic_importances = topic_importances / topic_importances.sum()
            self.temporal_importance_[i_timebin, :] = topic_importances
            t_dtm = self.doc_term_matrix[time_labels == i_timebin]
            t_doc_topic = self.document_topic_matrix[time_labels == i_timebin]
            if feature_importance == "c-tf-idf":
                self.temporal_components_[i_timebin], _ = ctf_idf(
                    t_doc_topic, t_dtm, return_idf=True
                )
            elif feature_importance == "soft-c-tf-idf":
                self.temporal_components_[i_timebin], _ = soft_ctf_idf(
                    t_doc_topic, t_dtm, return_idf=True
                )
            elif feature_importance == "bayes":
                self.temporal_components_[i_timebin] = bayes_rule(
                    t_doc_topic, t_dtm
                )
            elif feature_importance == "fighting-words":
                self.temporal_components_[i_timebin] = fighting_words(
                    t_doc_topic, t_dtm
                )
            elif feature_importance in ["centroid", "linear"]:
                t_topic_vectors = self._calculate_topic_vectors(
                    is_in_slice=time_labels == i_timebin,
                )
                if feature_importance == "centroid":
                    components = cluster_centroid_distance(
                        t_topic_vectors,
                        self.vocab_embeddings,
                    )
                    mask_terms = t_dtm.sum(axis=0).astype(np.float64)
                    mask_terms = np.squeeze(np.asarray(mask_terms))
                    components[:, mask_terms == 0] = np.nan
                    self.temporal_components_[i_timebin] = components
                else:
                    self.temporal_components_[i_timebin] = linear_classifier(
                        t_doc_topic,
                        embeddings=self.embeddings,
                        vocab_embedding=self.vocab_embeddings,
                    )
        return self.temporal_components_

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
            doc_topic_matrix = safe_binarize(
                self.labels_, classes=self.classes_
            )
        else:
            doc_topic_matrix = self.fit_transform(
                raw_documents, embeddings=embeddings
            )
        n_comp, n_vocab = self.components_.shape
        n_bins = len(self.time_bin_edges) - 1
        self.temporal_components_ = np.zeros(
            (n_bins, n_comp, n_vocab), dtype=doc_topic_matrix.dtype
        )
        self.temporal_importance_ = np.zeros((n_bins, n_comp))
        if embeddings is None:
            embeddings = self.encode_documents(raw_documents)
        self.embeddings = embeddings
        self.estimate_temporal_components(
            time_labels, self.time_bin_edges, self.feature_importance
        )
        return doc_topic_matrix

    @staticmethod
    def _labels_to_indices(labels, classes):
        n_classes = len(classes)
        class_to_index = dict(zip(classes, np.arange(n_classes)))
        return np.array([class_to_index[label] for label in labels])

    def plot_clusters_datamapplot(
        self, dimensions: tuple[int, int] = (0, 1), *args, **kwargs
    ):
        try:
            import datamapplot
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "You need to install datamapplot to be able to use plot_clusters_datamapplot()."
            ) from e
        coordinates = self.reduced_embeddings[:, dimensions]
        coordinates = scale(coordinates) * 4
        indices = self._labels_to_indices(self.labels_, self.classes_)
        labels = np.array(self.topic_names)[indices]
        if -1 in self.classes_:
            i_outlier = np.where(self.classes_ == -1)[0][0]
            kwargs["noise_label"] = self.topic_names[i_outlier]
        plot = datamapplot.create_interactive_plot(
            coordinates, labels, *args, **kwargs
        )

        def show_fig():
            with tempfile.TemporaryDirectory() as temp_dir:
                file_name = Path(temp_dir).joinpath("fig.html")
                plot.save(file_name)
                webbrowser.open("file://" + str(file_name.absolute()), new=2)
                time.sleep(2)

        plot.show = show_fig
        plot.write_html = plot.save
        return plot

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if getattr(self, "components_", None) is None:
            raise NotFittedError(
                "You can only transform documents once the model has been fitted."
            )
        idf_diag = getattr(self, "_idf_diag", None)
        if idf_diag is not None:
            X = self.vectorizer.transform(raw_documents)
            X = normalize(X, axis=1, norm="l1", copy=False)
            X = X * idf_diag
            doc_topic_matrix = np.exp(cosine_similarity(X, self.components_))
        elif self.feature_importance == "centroid":
            if embeddings is None:
                embeddings = self.encode_documents(raw_documents)
            doc_topic_matrix = np.exp(
                cosine_similarity(embeddings, self._calculate_topic_vectors())
            )
        else:
            doc_topic_matrix = safe_binarize(
                self.labels_, classes=self.classes_
            )
        return doc_topic_matrix


class BERTopic(ClusteringTopicModel):
    """Convenience function to construct a BERTopic model in Turftopic.
    The model is essentially just a ClusteringTopicModel
    with BERTopic's defaults (UMAP -> HDBSCAN -> C-TF-IDF).

    ```bash
    pip install turftopic[umap-learn]
    ```

    ```python
    from turftopic import BERTopic

    corpus: list[str] = ["some text", "more text", ...]

    model = BERTopic().fit(corpus)
    model.print_topics()
    ```
    """

    def __init__(
        self,
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        dimensionality_reduction: Optional[TransformerMixin] = None,
        clustering: Optional[ClusterMixin] = None,
        feature_importance: WordImportance = "c-tf-idf",
        n_reduce_to: Optional[int] = None,
        reduction_method: LinkageMethod = "average",
        reduction_distance_metric: DistanceMetric = "cosine",
        reduction_topic_representation: TopicRepresentation = "component",
        random_state: Optional[int] = None,
    ):
        if dimensionality_reduction is None:
            try:
                from umap import UMAP
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "UMAP is not installed in your environment, but BERTopic requires it."
                ) from e
            dimensionality_reduction = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=random_state,
            )
        if clustering is None:
            clustering = HDBSCAN(
                min_cluster_size=10,
                metric="euclidean",
                cluster_selection_method="eom",
            )
        super().__init__(
            encoder=encoder,
            vectorizer=vectorizer,
            dimensionality_reduction=dimensionality_reduction,
            clustering=clustering,
            n_reduce_to=n_reduce_to,
            random_state=random_state,
            feature_importance=feature_importance,
            reduction_method=reduction_method,
            reduction_distance_metric=reduction_distance_metric,
            reduction_topic_representation=reduction_topic_representation,
        )


class Top2Vec(ClusteringTopicModel):
    """Convenience function to construct a Top2Vec model in Turftopic.
    The model is essentially the same as ClusteringTopicModel
    with defaults that resemble Top2Vec (UMAP -> HDBSCAN -> Centroid term importance).

    ```bash
    pip install turftopic[umap-learn]
    ```

    ```python
    from turftopic import Top2Vec

    corpus: list[str] = ["some text", "more text", ...]

    model = Top2Vec().fit(corpus)
    model.print_topics()
    ```
    """

    def __init__(
        self,
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        dimensionality_reduction: Optional[TransformerMixin] = None,
        clustering: Optional[ClusterMixin] = None,
        feature_importance: WordImportance = "centroid",
        n_reduce_to: Optional[int] = None,
        reduction_method: LinkageMethod = "smallest",
        reduction_distance_metric: DistanceMetric = "cosine",
        reduction_topic_representation: TopicRepresentation = "centroid",
        random_state: Optional[int] = None,
    ):
        if dimensionality_reduction is None:
            try:
                from umap import UMAP
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "UMAP is not installed in your environment, but Top2Vec requires it."
                ) from e
            dimensionality_reduction = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=random_state,
            )
        if clustering is None:
            clustering = HDBSCAN(
                min_cluster_size=15,
                metric="euclidean",
                cluster_selection_method="eom",
            )
        super().__init__(
            encoder=encoder,
            vectorizer=vectorizer,
            dimensionality_reduction=dimensionality_reduction,
            clustering=clustering,
            n_reduce_to=n_reduce_to,
            random_state=random_state,
            feature_importance=feature_importance,
            reduction_method=reduction_method,
            reduction_distance_metric=reduction_distance_metric,
            reduction_topic_representation=reduction_topic_representation,
        )
