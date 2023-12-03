from typing import Literal, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import ClusterMixin, TransformerMixin
from sklearn.cluster import OPTICS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import Isomap
from sklearn.preprocessing import label_binarize

from turftopic.base import ContextualModel
from turftopic.centroid_distance import cluster_centroid_distance
from turftopic.soft_ctf_idf import soft_ctf_idf


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

    def fit_predict(self, raw_documents, y=None):
        embeddings = self.encoder_.encode(raw_documents)
        dtm = self.vectorizer.fit_transform(raw_documents)
        self.vocab_ = self.vectorizer.get_feature_names_out()
        cluster_labels = self.clustering.fit_predict(embeddings)
        clusters = np.unique(cluster_labels)
        self.classes_ = np.sort(clusters)
        if self.feature_importance == "ctfidf":
            document_topic_matrix = label_binarize(
                cluster_labels, classes=self.classes_
            )
            self.components_ = soft_ctf_idf(document_topic_matrix, dtm)
        else:
            vocab_embeddings = self.encoder_.encode(self.vocab_)
            self.components_ = cluster_centroid_distance(
                cluster_labels,
                embeddings,
                vocab_embeddings,
                metric="euclidean",
            )
        self.labels_ = cluster_labels
        return cluster_labels

    def fit(self, raw_documents, y=None):
        self.fit_predict(raw_documents, y)
        return self

    def fit_transform(self, raw_documents, y=None):
        labels = self.fit_predict(raw_documents, y)
        return label_binarize(labels, classes=self.classes_)
