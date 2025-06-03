import typing
import warnings
from collections import Counter
from typing import Optional, Sequence

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances

from turftopic.base import ContextualModel
from turftopic.feature_importance import (
    bayes_rule,
    cluster_centroid_distance,
    ctf_idf,
    fighting_words,
    linear_classifier,
    soft_ctf_idf,
)
from turftopic.hierarchical import TopicNode
from turftopic.utils import safe_binarize

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)

LinkageMethod = typing.Literal[
    # Top2Vec's merging method
    "smallest",
    # Linkage methods from SciPy
    "single",
    "complete",
    "average",
    "centroid",
    "median",
    "ward",
    "weighted",
]
VALID_LINKAGE_METHODS = list(typing.get_args(LinkageMethod))


def smallest_linkage(
    n_reduce_to: int,
    topic_vectors: np.ndarray,
    topic_sizes: np.ndarray,
    classes: np.ndarray,
    metric: str = "cosine",
) -> np.ndarray:
    """Iteratively joins smallest topics based on Top2Vec's algorithm.
    Returns linkage matrix in scipy format
    (see [Scipy Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage)).
    """
    linkage_entries = []
    topic_vectors = np.copy(topic_vectors)
    topic_sizes = np.copy(topic_sizes)
    if -1 in classes:
        topic_sizes = topic_sizes[classes != -1]
        topic_vectors = topic_vectors[classes != -1]
        classes = classes[classes != -1]
    n_classes = len(classes)
    n_iterations = n_classes - n_reduce_to
    for i_iteration in range(n_iterations):
        smallest = np.argmin(topic_sizes)
        dist = pairwise_distances(
            np.atleast_2d(topic_vectors[smallest]),
            topic_vectors,
            metric=metric,
        )[0]
        # Obviously the closest is the cluster itself, so we get the second closest
        closest = np.argsort(dist)[1]
        n_obs = topic_sizes[smallest] + topic_sizes[closest]
        linkage_entries.append(
            [classes[smallest], classes[closest], dist[closest], n_obs]
        )
        classes = np.append(classes, [i_iteration + n_classes], axis=0)
        new_topic_vector = (
            (topic_vectors[smallest] * topic_sizes[smallest])
            + (topic_vectors[closest] * topic_sizes[closest])
        ) / (topic_sizes[smallest] + topic_sizes[closest])
        topic_vectors = np.append(
            topic_vectors, np.atleast_2d(new_topic_vector), axis=0
        )
        topic_sizes = np.append(topic_sizes, [n_obs], axis=0)
        classes = np.delete(classes, [smallest, closest], axis=0)
        topic_vectors = np.delete(topic_vectors, [smallest, closest], axis=0)
        topic_sizes = np.delete(topic_sizes, [smallest, closest], axis=0)
    return np.array(linkage_entries)


class ClusterNode(TopicNode):
    """Hierarchical Topic Node for clustering models.
    Supports merging topics based on a hierarchical merging strategy."""

    @classmethod
    def create_root(cls, model: ContextualModel, labels: np.ndarray):
        """Creates root node from a topic models' components and topic importances in documents."""
        classes = np.sort(np.unique(labels))
        document_topic_matrix = safe_binarize(labels, classes=classes)
        children = []
        for topic_id, doc_top in zip(classes, document_topic_matrix.T):
            children.append(
                cls(
                    model,
                    path=(topic_id,),
                    document_topic_vector=doc_top,
                    children=None,
                )
            )
        res = cls(
            model,
            path=(),
            word_importance=None,
            document_topic_vector=None,
            children=children,
        )
        res.estimate_components()
        return res

    def join_topics(
        self, to_join: Sequence[int], joint_id: Optional[int] = None
    ):
        """Joins a number of topics into a new topic with a given ID.

        Parameters
        ----------
        to_join: Sequence of int
            Children in the hierarchy to join (IDs indicate the last element of the path).
        joint_id: int, default None
            ID to give to the joint topic. By default, this will be the topic with the smallest ID.
        """
        if self.children is None:
            raise TypeError("Node doesn't have children, can't merge.")
        if len(set(to_join)) < len(to_join):
            raise ValueError(
                f"You can't join a cluster with itself: {to_join}"
            )
        if joint_id is None:
            joint_id = min(to_join)
        children = [self[i] for i in to_join]
        joint_membership = np.stack(
            [child.document_topic_vector for child in children]
        )
        joint_membership = np.sum(joint_membership, axis=0)
        child_ids = [child.path[-1] for child in children]
        joint_node = TopicNode(
            model=self.model,
            children=children,
            document_topic_vector=joint_membership,
            path=(*self.path, joint_id),
        )
        for child in joint_node:
            child._append_path(joint_id)
        self.children = [
            child for child in self.children if child.path[-1] not in child_ids
        ] + [joint_node]
        component_map = self._estimate_children_components()
        for child in self.children:
            child.word_importance = component_map[child.path[-1]]

    def estimate_components(self) -> np.ndarray:
        component_map = self._estimate_children_components()
        for child in self.children:
            child.word_importance = component_map[child.path[-1]]
        return self.components_

    @property
    def labels_(self) -> np.ndarray:
        topic_document_membership = np.stack(
            [child.document_topic_vector for child in self.children]
        )
        labels = np.argmax(topic_document_membership, axis=0)
        strength = np.max(topic_document_membership, axis=0)
        # documents that are not in this part of the hierarchy are treated as outliers
        labels[strength == 0] = -1
        return np.array(
            [self.children[label].path[-1] for label in labels if label != -1]
        )

    def _estimate_children_components(self) -> dict[int, np.ndarray]:
        """Estimates feature importances based on a fitted clustering."""
        clusters = np.unique(self.labels_)
        classes = np.sort(clusters)
        labels = self.labels_
        topic_vectors = self.model._calculate_topic_vectors(
            classes=classes, labels=labels
        )
        document_topic_matrix = safe_binarize(labels, classes=classes)
        if self.model.feature_importance == "soft-c-tf-idf":
            components = soft_ctf_idf(
                document_topic_matrix, self.model.doc_term_matrix
            )  # type: ignore
        if self.model.feature_importance == "fighting-words":
            components = fighting_words(
                document_topic_matrix, self.model.doc_term_matrix
            )  # type: ignore
        elif self.model.feature_importance in ["centroid", "linear"]:
            if not hasattr(self.model, "vocab_embeddings"):
                self.model.vocab_embeddings = self.model.encode_documents(
                    self.model.vectorizer.get_feature_names_out()
                )  # type: ignore
                if (
                    self.model.vocab_embeddings.shape[1]
                    != topic_vectors.shape[1]
                ):
                    raise ValueError(
                        NOT_MATCHING_ERROR.format(
                            n_dims=topic_vectors.shape[1],
                            n_word_dims=self.model.vocab_embeddings.shape[1],
                        )
                    )
            if self.model.feature_importance == "centroid":
                components = cluster_centroid_distance(
                    topic_vectors,
                    self.model.vocab_embeddings,
                )
            else:
                components = linear_classifier(
                    document_topic_matrix,
                    self.model.embeddings,
                    self.model.vocab_embeddings,
                )
        elif self.model.feature_importance == "bayes":
            components = bayes_rule(
                document_topic_matrix, self.model.doc_term_matrix
            )
        else:
            components = ctf_idf(
                document_topic_matrix, self.model.doc_term_matrix
            )
        return dict(zip(classes, components))

    def _merge_clusters(self, linkage_matrix: np.ndarray):
        classes = self.classes_
        max_class = len(classes[classes != -1])
        for i_cluster, (left, right, *_) in enumerate(linkage_matrix):
            self.join_topics(
                [int(left), int(right)], int(max_class + i_cluster)
            )

    def _calculate_linkage(
        self, n_reduce_to: int, method: str = "average", metric: str = "cosine"
    ) -> np.ndarray:
        if method not in VALID_LINKAGE_METHODS:
            raise ValueError(
                f"Linkage method has to be one of: {VALID_LINKAGE_METHODS}, but got {method} instead."
            )
        classes = self.classes_
        labels = self.labels_
        topic_sizes = np.array([np.sum(labels == label) for label in classes])
        topic_representations = self.model.topic_representations
        if method == "smallest":
            return smallest_linkage(
                n_reduce_to=n_reduce_to,
                topic_vectors=topic_representations,
                topic_sizes=topic_sizes,
                classes=classes,
                metric=metric,
            )
        else:
            n_classes = len(classes[classes != -1])
            topic_vectors = topic_representations[classes != -1]
            n_reductions = n_classes - n_reduce_to
            cond_dist = pdist(topic_vectors, metric=metric)
            # Making the algorithm more numerically stable
            if metric == "cosine":
                cond_dist[~np.isfinite(cond_dist)] = -1
            return linkage(cond_dist, method=method)[:n_reductions]

    def reduce_topics(
        self, n_reduce_to: int, method: str = "average", metric: str = "cosine"
    ):
        n_topics = np.sum(self.classes_ != -1)
        if n_topics <= n_reduce_to:
            warnings.warn(
                f"Number of clusters is already {n_topics} <= {n_reduce_to}, nothing to do."
            )
            return
        linkage_matrix = self._calculate_linkage(
            n_reduce_to, method=method, metric=metric
        )
        self.linkage_matrix_ = linkage_matrix
        self._merge_clusters(linkage_matrix)
