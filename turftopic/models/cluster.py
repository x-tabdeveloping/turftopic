import warnings
from datetime import datetime
from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import OPTICS, AgglomerativeClustering
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import label_binarize

from turftopic.base import ContextualModel, Encoder
from turftopic.dynamic import DynamicTopicModel
from turftopic.feature_importance import (bayes_rule,
                                          cluster_centroid_distance, ctf_idf,
                                          soft_ctf_idf)
from turftopic.vectorizer import default_vectorizer

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

feature_message = """
feature_importance must be one of 'soft-c-tf-idf', 'c-tf-idf', 'centroid'
"""

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)


def smallest_hierarchical_join(
    topic_vectors: np.ndarray,
    topic_sizes: np.ndarray,
    classes_: np.ndarray,
    n_to: int,
) -> list[tuple]:
    """Iteratively joins smallest topics."""
    merge_inst = []
    topic_vectors = np.copy(topic_vectors)
    topic_sizes = np.copy(topic_sizes)
    classes = list(classes_)
    while len(classes) > n_to:
        smallest = np.argmin(topic_sizes)
        dist = cosine_distances(
            np.atleast_2d(topic_vectors[smallest]), topic_vectors
        )
        closest = np.argsort(dist[0])[1]
        merge_inst.append((classes[smallest], classes[closest]))
        classes.pop(smallest)
        topic_vectors[closest] = (
            (topic_vectors[smallest] * topic_sizes[smallest])
            + (topic_vectors[closest] * topic_sizes[closest])
        ) / (topic_sizes[smallest] + topic_sizes[closest])
        topic_vectors = np.delete(topic_vectors, smallest, axis=0)
        topic_sizes[closest] = topic_sizes[closest] + topic_sizes[smallest]
        topic_sizes = np.delete(topic_sizes, smallest, axis=0)
    return merge_inst


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


class ClusteringTopicModel(ContextualModel, ClusterMixin, DynamicTopicModel):
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
    feature_importance: {'soft-c-tf-idf', 'c-tf-idf', 'bayes', 'centroid'}, default 'soft-c-tf-idf'
        Method for estimating term importances.
        'centroid' uses distances from cluster centroid similarly
        to Top2Vec.
        'c-tf-idf' uses BERTopic's c-tf-idf.
        'soft-c-tf-idf' uses Soft c-TF-IDF from GMM, the results should
        be very similar to 'c-tf-idf'.
        'bayes' uses Bayes' rule.
    n_reduce_to: int, default None
        Number of topics to reduce topics to.
        The specified reduction method will be used to merge them.
        By default, topics are not merged.
    reduction_method: 'agglomerative', 'smallest'
        Method used to reduce the number of topics post-hoc.
        When 'agglomerative', BERTopic's topic reduction method is used,
        where topic vectors are hierarchically clustered.
        When 'smallest', the smallest topic gets merged into the closest
        non-outlier cluster until the desired number
        is achieved similarly to Top2Vec.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    """

    def __init__(
        self,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        dimensionality_reduction: Optional[TransformerMixin] = None,
        clustering: Optional[ClusterMixin] = None,
        feature_importance: Literal[
            "c-tf-idf",
            "soft-c-tf-idf",
            "centroid",
            "bayes",
        ] = "soft-c-tf-idf",
        n_reduce_to: Optional[int] = None,
        reduction_method: Literal[
            "agglomerative", "smallest"
        ] = "agglomerative",
        random_state: Optional[int] = None,
    ):
        self.encoder = encoder
        self.random_state = random_state
        if feature_importance not in [
            "c-tf-idf",
            "soft-c-tf-idf",
            "centroid",
            "bayes",
        ]:
            raise ValueError(feature_message)
        if isinstance(encoder, int):
            raise TypeError(integer_message)
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        if clustering is None:
            self.clustering = OPTICS(min_samples=25)
        else:
            self.clustering = clustering
        if dimensionality_reduction is None:
            self.dimensionality_reduction = TSNE(
                n_components=2, metric="cosine", random_state=random_state
            )
        else:
            self.dimensionality_reduction = dimensionality_reduction
        self.feature_importance = feature_importance
        self.n_reduce_to = n_reduce_to
        self.reduction_method = reduction_method

    def _calculate_topic_vectors(
        self, is_in_slice: Optional[np.ndarray] = None
    ) -> np.ndarray:
        label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        n_topics = len(self.classes_)
        n_dims = self.embeddings.shape[1]
        topic_vectors = np.full((n_topics, n_dims), np.nan)
        for label in np.unique(self.labels_):
            doc_idx = self.labels_ == label
            if is_in_slice is not None:
                doc_idx = doc_idx & is_in_slice
            topic_vectors[label_to_idx[label], :] = np.mean(
                self.embeddings[doc_idx], axis=0
            )
        return topic_vectors

    def _merge_agglomerative(self, n_reduce_to: int) -> np.ndarray:
        n_topics = self.components_.shape[0]
        res = {old_label: old_label for old_label in self.classes_}
        if n_topics <= n_reduce_to:
            return self.labels_
        interesting_topic_vectors = np.stack(
            [
                vec
                for label, vec in zip(self.classes_, self.topic_vectors_)
                if label != -1
            ]
        )
        old_labels = [label for label in self.classes_ if label != -1]
        new_labels = AgglomerativeClustering(
            n_clusters=n_reduce_to,
            metric="cosine",
            linkage="average",
        ).fit_predict(interesting_topic_vectors)
        res = {}
        if -1 in self.classes_:
            res[-1] = -1
        for i_old, i_new in zip(old_labels, new_labels):
            res[i_old] = i_new
        return np.array([res[label] for label in self.labels_])

    def _merge_smallest(self, n_reduce_to: int):
        merge_inst = smallest_hierarchical_join(
            self.topic_vectors_[self.classes_ != -1],
            self.topic_sizes_[self.classes_ != -1],
            self.classes_[self.classes_ != -1],
            n_reduce_to,
        )
        labels = np.copy(self.labels_)
        for from_topic, to_topic in merge_inst:
            labels[labels == from_topic] = to_topic
        return labels

    def reduce_topics(
        self,
        n_reduce_to: int,
        reduction_method: Literal["smallest", "agglomerative"],
    ) -> np.ndarray:
        """Reduces the clustering to the desired amount with the given method.

        Parameters
        ----------
        n_reduce_to: int, default None
            Number of topics to reduce topics to.
            The specified reduction method will be used to merge them.
            By default, topics are not merged.
        reduction_method: 'agglomerative', 'smallest'
            Method used to reduce the number of topics post-hoc.
            When 'agglomerative', BERTopic's topic reduction method is used,
            where topic vectors are hierarchically clustered.
            When 'smallest', the smallest topic gets merged into the closest
            non-outlier cluster until the desired number
            is achieved similarly to Top2Vec.

        Returns
        -------
        ndarray of shape (n_documents)
            New cluster labels for documents.
        """
        if not hasattr(self, "original_labels_"):
            self.original_labels_ = self.labels_
        if reduction_method == "smallest":
            self.labels_ = self._merge_smallest(n_reduce_to)
        elif reduction_method == "agglomerative":
            self.labels_ = self._merge_agglomerative(n_reduce_to)
        self.estimate_components(self.feature_importance)
        return self.labels_

    def join_topics(self, topic_ids: list[int]):
        """Joins given topic together into one topic and reestimates term importances.

        Example:
        ```python
        model.join_topics([0,3,2])
        ```

        Parameters
        ----------
        topic_ids: list[int]
            Topic IDs to join together.
            The new topic will get the lowest ID.
        """
        topic_ids = sorted(topic_ids)
        new_topic = topic_ids[0]
        new_labels = []
        self.original_labels_ = self.labels_
        for label in self.labels_:
            if label in topic_ids:
                new_labels.append(new_topic)
            else:
                new_labels.append(label)
        self.labels_ = np.array(new_labels)
        self.estimate_components(self.feature_importance)

    def reset_topics(self):
        """Resets topic reductions to the original clustering."""
        if not hasattr(self, "original_labels_"):
            warnings.warn("Topics have never been reduced, nothing to reset.")
        else:
            self.labels_ = self.original_labels_
            self.estimate_components(self.feature_importance)

    def estimate_components(
        self,
        feature_importance: Literal[
            "centroid", "soft-c-tf-idf", "bayes", "c-tf-idf"
        ],
    ) -> np.ndarray:
        """Estimates feature importances based on a fitted clustering.

        Parameters
        ----------
        feature_importance: {'soft-c-tf-idf', 'c-tf-idf', 'bayes', 'centroid'}, default 'soft-c-tf-idf'
            Method for estimating term importances.
            'centroid' uses distances from cluster centroid similarly
            to Top2Vec.
            'c-tf-idf' uses BERTopic's c-tf-idf.
            'soft-c-tf-idf' uses Soft c-TF-IDF from GMM, the results should
            be very similar to 'c-tf-idf'.
            'bayes' uses Bayes' rule.

        Returns
        -------
        ndarray of shape (n_components, n_vocab)
            Topic-term matrix.
        """
        self.topic_names_ = None
        if getattr(self, "labels_", None) is None:
            raise NotFittedError(
                "The model has not been fitted yet, please fit the model before estimating temporal components."
            )
        clusters = np.unique(self.labels_)
        self.classes_ = np.sort(clusters)
        self.topic_sizes_ = np.array(
            [np.sum(self.labels_ == label) for label in self.classes_]
        )
        self.topic_vectors_ = self._calculate_topic_vectors()
        document_topic_matrix = label_binarize(
            self.labels_, classes=self.classes_
        )
        if feature_importance == "soft-c-tf-idf":
            self.components_ = soft_ctf_idf(
                document_topic_matrix, self.doc_term_matrix
            )  # type: ignore
        elif feature_importance == "centroid":
            if not hasattr(self, "vocab_embeddings"):
                self.vocab_embeddings = self.encoder_.encode(
                    self.vectorizer.get_feature_names_out()
                )  # type: ignore
                if (
                    self.vocab_embeddings.shape[1]
                    != self.topic_vectors_.shape[1]
                ):
                    raise ValueError(
                        NOT_MATCHING_ERROR.format(
                            n_dims=self.topic_vectors_.shape[1],
                            n_word_dims=self.vocab_embeddings.shape[1],
                        )
                    )
            self.components_ = cluster_centroid_distance(
                self.topic_vectors_,
                self.vocab_embeddings,
            )
        elif feature_importance == "bayes":
            self.components_ = bayes_rule(
                document_topic_matrix, self.doc_term_matrix
            )
        else:
            self.components_ = ctf_idf(
                document_topic_matrix, self.doc_term_matrix
            )
        return self.components_

    def fit_predict(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fits model and predicts cluster labels for all given documents.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        y: None
            Ignored, exists for sklearn compatibility.
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
                embeddings = self.encoder_.encode(raw_documents)
                console.log("Encoding done.")
            self.embeddings = embeddings
            status.update("Extracting terms")
            self.doc_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            status.update("Reducing Dimensionality")
            reduced_embeddings = self.dimensionality_reduction.fit_transform(
                embeddings
            )
            console.log("Dimensionality reduction done.")
            status.update("Clustering documents")
            self.labels_ = self.clustering.fit_predict(reduced_embeddings)
            console.log("Clustering done.")
            status.update("Estimating parameters.")
            self.estimate_components(self.feature_importance)
            console.log("Parameter estimation done.")
            if self.n_reduce_to is not None:
                n_topics = self.classes_.shape[0]
                status.update(
                    f"Reducing topics from {n_topics} to {self.n_reduce_to}"
                )
                self.reduce_topics(self.n_reduce_to, self.reduction_method)
                console.log(
                    f"Topic reduction done from {n_topics} to {self.n_reduce_to}."
                )
                status.update("Reestimating parameters.")
                self.estimate_components(self.feature_importance)
                console.log("Reestimation done.")
        console.log("Model fitting done.")
        self.doc_topic_matrix = label_binarize(
            self.labels_, classes=self.classes_
        )
        return self.labels_

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        labels = self.fit_predict(raw_documents, y, embeddings)
        return label_binarize(labels, classes=self.classes_)

    def estimate_temporal_components(
        self,
        time_labels,
        time_bin_edges,
        feature_importance: Literal[
            "c-tf-idf", "soft-c-tf-idf", "centroid", "bayes"
        ],
    ) -> np.ndarray:
        """Estimates temporal components based on a fitted topic model.

        Parameters
        ----------
        feature_importance: {'soft-c-tf-idf', 'c-tf-idf', 'bayes', 'centroid'}, default 'soft-c-tf-idf'
            Method for estimating term importances.
            'centroid' uses distances from cluster centroid similarly
            to Top2Vec.
            'c-tf-idf' uses BERTopic's c-tf-idf.
            'soft-c-tf-idf' uses Soft c-TF-IDF from GMM, the results should
            be very similar to 'c-tf-idf'.
            'bayes' uses Bayes' rule.

        Returns
        -------
        ndarray of shape (n_time_bins, n_components, n_vocab)
            Temporal topic-term matrix.
        """
        if getattr(self, "components_", None) is None:
            raise NotFittedError(
                "The model has not been fitted yet, please fit the model before estimating temporal components."
            )
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
            topic_importances = self.doc_topic_matrix[
                time_labels == i_timebin
            ].sum(axis=0)
            if not topic_importances.sum() == 0:
                topic_importances = topic_importances / topic_importances.sum()
            self.temporal_importance_[i_timebin, :] = topic_importances
            t_dtm = self.doc_term_matrix[time_labels == i_timebin]
            t_doc_topic = self.doc_topic_matrix[time_labels == i_timebin]
            if feature_importance == "c-tf-idf":
                self.temporal_components_[i_timebin] = ctf_idf(
                    t_doc_topic, t_dtm
                )
            elif feature_importance == "soft-c-tf-idf":
                self.temporal_components_[i_timebin] = soft_ctf_idf(
                    t_doc_topic, t_dtm
                )
            elif feature_importance == "bayes":
                self.temporal_components_[i_timebin] = bayes_rule(
                    t_doc_topic, t_dtm
                )
            elif feature_importance == "centroid":
                t_topic_vectors = self._calculate_topic_vectors(
                    time_labels == i_timebin,
                )
                components = cluster_centroid_distance(
                    t_topic_vectors,
                    self.vocab_embeddings,
                )
                mask_terms = t_dtm.sum(axis=0).astype(np.float64)
                mask_terms = np.squeeze(np.asarray(mask_terms))
                components[:, mask_terms == 0] = np.nan
                self.temporal_components_[i_timebin] = components
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
            doc_topic_matrix = label_binarize(
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
            embeddings = self.encoder_.encode(raw_documents)
        self.embeddings = embeddings
        self.estimate_temporal_components(
            time_labels, self.time_bin_edges, self.feature_importance
        )
        return doc_topic_matrix
