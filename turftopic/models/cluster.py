from typing import Literal, Optional, Union

import numpy as np
import scipy.sparse as spr
from sentence_transformers import SentenceTransformer
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import OPTICS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import Isomap
from sklearn.preprocessing import label_binarize

from turftopic.base import ContextualModel
from turftopic.centroid_distance import cluster_centroid_distance
from turftopic.soft_ctf_idf import soft_ctf_idf

integer_message = """
You tried to pass an integer to ClusteringTopicModel as its first argument.
We assume you tried to specify the number of topics.
Since in ClusteringTopicModel the clustering model determines the number of topics,
and this process may be automatic, you have to pass along a clustering model
where the number of clusters is predefined.

For instance: ClusteringTopicModel(clustering=KMeans(10))
"""


class ClusteringTopicModel(ContextualModel, ClusterMixin):
    def __init__(
        self,
        encoder: Union[
            SentenceTransformer, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        dimensionality_reduction: Optional[TransformerMixin] = None,
        clustering: Optional[ClusterMixin] = None,
        vectorizer: Optional[CountVectorizer] = None,
        feature_importance: Literal["ctfidf", "centroid"] = "centroid",
    ):
        self.encoder = encoder
        if isinstance(encoder, int):
            raise TypeError(integer_message)
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = CountVectorizer(min_df=10)
        else:
            self.vectorizer = vectorizer
        if clustering is None:
            self.clustering = OPTICS()
        else:
            self.clustering = clustering
        if dimensionality_reduction is None:
            self.dimensionality_reduction = Isomap(
                n_neighbors=25, n_components=5
            )
        else:
            self.dimensionality_reduction = dimensionality_reduction
        self.feature_importance = feature_importance

    def fit_predict(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        doc_term_matrix = self.vectorizer.fit_transform(raw_documents)
        vocab = self.vectorizer.get_feature_names_out()
        cluster_labels = self.clustering.fit_predict(embeddings)
        clusters = np.unique(cluster_labels)
        self.classes_ = np.sort(clusters)
        if self.feature_importance == "ctfidf":
            document_topic_matrix = label_binarize(
                cluster_labels, classes=self.classes_
            )
            self.components_ = soft_ctf_idf(document_topic_matrix, doc_term_matrix)  # type: ignore
        else:
            vocab_embeddings = self.encoder_.encode(vocab)  # type: ignore
            self.components_ = cluster_centroid_distance(
                cluster_labels,
                embeddings,
                vocab_embeddings,
                metric="euclidean",
            )
        self.labels_ = cluster_labels
        return cluster_labels

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        labels = self.fit_predict(raw_documents, y, embeddings)
        return label_binarize(labels, classes=self.classes_)
