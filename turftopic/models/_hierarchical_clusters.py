from typing import Sequence

import numpy as np
from sklearn.preprocessing import label_binarize

from turftopic.feature_importance import (bayes_rule,
                                          cluster_centroid_distance, ctf_idf,
                                          soft_ctf_idf)
from turftopic.hierarchical import TopicNode

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)


class ClusterNode(TopicNode):
    def merge_children(self, to_join: Sequence[int | tuple[int, ...]]):
        if self.children is None:
            raise TypeError("Node doesn't have children, can't merge.")
        children = [self[i] for i in to_join]
        joint_membership = np.stack(
            [child.document_topic_vector for child in children]
        )
        joint_membership = np.sum(joint_membership, axis=0)
        child_ids = [child.path[-1] for child in children]
        min_id = min(child_ids)
        for child in children:
            child.path = (*self.path, min_id, child.path[-1])
        joint_node = TopicNode(
            model=self.model,
            children=children,
            document_topic_vector=joint_membership,
            path=(*self.path, min_id),
        )
        self.children = [
            child for child in self.children if child.path[-1] not in child_ids
        ] + [joint_node]
        component_map = self.estimate_children_components()
        for child in self.children:
            child.word_importance = component_map[child.path[-1]]

    @property
    def labels_(self) -> np.ndarray:
        topic_document_membership = np.stack(
            [child.document_topic_vector for child in self.children]
        )
        labels = np.argmax(topic_document_membership, axis=0)
        strength = np.max(topic_document_membership, axis=0)
        # documents that are not in this part of the hierarchy are treated as outliers
        labels[strength == 0] = -1
        return [
            self.children[label].path[-1] for label in labels if label != -1
        ]

    def estimate_children_components(self) -> dict[int, np.ndarray]:
        """Estimates feature importances based on a fitted clustering."""
        clusters = np.unique(self.labels_)
        classes = np.sort(clusters)
        labels = self.labels_
        topic_vectors = self.model._calculate_topic_vectors(
            classes=classes, labels=labels
        )
        document_topic_matrix = label_binarize(labels, classes=classes)
        if self.model.feature_importance == "soft-c-tf-idf":
            components = soft_ctf_idf(
                document_topic_matrix, self.model.doc_term_matrix
            )  # type: ignore
        elif self.model.feature_importance == "centroid":
            if not hasattr(self.model, "vocab_embeddings"):
                self.model.vocab_embeddings = self.model.encoder_.encode(
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
            components = cluster_centroid_distance(
                topic_vectors,
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
