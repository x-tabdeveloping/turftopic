import datetime
import itertools
import warnings
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Callable

import numpy as np
import scipy.sparse as spr
from sklearn.metrics.pairwise import cosine_similarity

from turftopic.base import ContextualModel
from turftopic.hierarchical import TopicNode
from turftopic.vectorizers.merging import UnionVectorizer


def nanaverage(a, weights=None, axis=0):
    if weights is None:
        return np.nanmean(a, axis=axis)
    scaled_sum = np.nansum(a * weights[:, None], axis=axis)
    denominator = weights.sum()
    return scaled_sum / denominator


def safe_agg(a, agg, weights, axis=0):
    try:
        return agg(a, weights=weights, axis=axis)
    except Exception:
        return agg(a, axis=axis)


def safe_cosine(a, b):
    a = np.where(np.isfinite(a), a, np.zeros_like(a))
    b = np.where(np.isfinite(b), b, np.zeros_like(b))
    return cosine_similarity(a, b)


MergeHistory = list[list[int]]


def stack_components(component_matrices) -> tuple[np.ndarray, np.ndarray]:
    stacked_components = np.concatenate(component_matrices, axis=0)
    n_comps = [comp.shape[0] for comp in component_matrices]
    labels = np.repeat(np.arange(len(component_matrices)), n_comps)
    return stacked_components, labels


def merge_from_history(
    component_matrices: list[np.ndarray],
    merge_history: MergeHistory,
    weights=None,
    agg=nanaverage,
) -> np.ndarray:
    stacked_components, old_labels = stack_components(component_matrices)
    flat_merge_hist = list(itertools.chain.from_iterable(merge_history))
    unique_components = np.unique(flat_merge_hist)
    assert unique_components[0] == 0
    new_to_old = defaultdict(list)
    current = 0
    for merge_h, comp in zip(merge_history, component_matrices):
        n_model_components = comp.shape[0]
        for i_old, to_new in enumerate(merge_h):
            new_to_old[to_new].append(i_old + current)
        current += n_model_components
    n_dims = stacked_components.shape[1]
    new_components = np.zeros(
        (len(unique_components), n_dims), dtype=stacked_components.dtype
    )
    for i_new, old_ind in new_to_old.items():
        new_components[i_new] = safe_agg(
            stacked_components[old_ind],
            agg=agg,
            weights=weights,
            axis=0,
        )
    return new_components


def merge_models(
    models: ContextualModel,
    match_threshold=0.7,
    merge_method: str | Callable = "asymmetric_mean",
    weighted=True,
    return_merge_graph: bool = False,
):
    import igraph as ig

    if len(models) == 0:
        raise ValueError("List is empty, cannot merge zero topic models.")
    if not all([hasattr(model, "merge_with") for model in models]):
        raise ValueError(
            "Models do not all have a merge_with method, cannot merge"
        )
    model = deepcopy(models[0])
    node_labels = list(model.topic_names)
    vertex_components = model.components_.tolist()
    vocabs = [model.get_vocab()] * len(model.components_)
    stage = [0] * len(node_labels)
    pointer = 0
    edges = []
    for i_model, new_model in enumerate(models[1:]):
        model.merge_with(
            new_model,
            match_threshold=match_threshold,
            merge_method=merge_method,
            weighted=weighted,
        )
        orig_to, new_to = model.merge_history_
        for orig_index, joint_index in enumerate(orig_to):
            edges.append(
                (
                    orig_index + pointer,
                    joint_index + pointer + len(orig_to) + len(new_to),
                )
            )
        pointer += len(orig_to)
        for new_index, joint_index in enumerate(new_to):
            edges.append(
                (
                    new_index + pointer,
                    joint_index + pointer + len(new_to),
                )
            )
        pointer += len(new_to)
        node_labels += new_model.topic_names
        vertex_components.extend(new_model.components_.tolist())
        vocabs.extend([new_model.get_vocab()] * len(new_model.components_))
        node_labels += model.topic_names
        vocabs.extend([model.get_vocab()] * len(model.components_))
        vertex_components.extend(model.components_.tolist())
        stage.extend([(i_model + 1) * 2 - 1] * len(new_to))
        stage.extend([(i_model + 1) * 2] * (len(model.topic_names)))
    n_vertices = len(node_labels)
    g = ig.Graph(n_vertices, edges, directed=True)
    g.vs["name"] = node_labels
    g.vs["stage"] = stage
    g.vs["vocab"] = vocabs
    g.vs["components"] = vertex_components
    if return_merge_graph:
        return model, g
    else:
        return model


def symmetric_merge(
    component_matrices: list[np.ndarray],
    weights=None,
    match_threshold: float = 0.7,
    agg=nanaverage,
    sim_fn=safe_cosine,
    allow_within_model_match=False,
    topic_embeddings=None,
) -> tuple[np.ndarray, MergeHistory]:
    """Performs a symmetric merge on a number of topic models.

    Parameters
    ----------
    component_matrices: list[np.ndarray]
        List of topic representations from all topic models.
    weights: Sequence, None
        Weight for each of the models in case of a weighted merge.
    match_threshold: float, default 0.7
        Similarity threshold above which two topics will be considered a match.
    agg: Callable, default np.average
        Aggregation method to use for matching topics.
    sim_fn: Callable, default cosine_similarity
        Function to produce the similarity matrix.
    allow_within_model_match: bool, default False
        Determines whether matches can happen within models.
    topic_embeddings: list[np.ndarray]
        Topic representations for calculating similarity.
        This can be useful when you want to merge components in vocab space,
        but want to calculate similarities based on embedding space.

    Returns
    -------
    ndarray of shape (n_new_topics, n_dims)
        New topic representations merged from the old ones.
    MergeHistory (list[list[int]])
        Indicates which joint topic the original topics were merged into.
        The data structure is a list of lists, where each list contains the indices of the joint topics
        each of the original topics were merged into.
        e.g. [[0,2,1], [3,2]] would indicate that the topics of the first model were merged into
        topics 0, 2 and 1 in the joint model, while the second model's topics were merged into 3 and 2.
    OPTIONAL: ndarray of shape (n_new_topics, n_dims)
        New topic embeddings.
    """
    stacked_components, old_labels = stack_components(component_matrices)
    if topic_embeddings is None:
        similarity = sim_fn(stacked_components, stacked_components)
    else:
        stacked_embeddings, old_embedding_labels = stack_components(
            topic_embeddings
        )
        similarity = sim_fn(stacked_components, stacked_components)
    if not allow_within_model_match:
        i_processed = 0
        for comp in component_matrices:
            n_comp = comp.shape[0]
            similarity[
                i_processed : i_processed + n_comp,
                i_processed : i_processed + n_comp,
            ] = np.eye(n_comp).T
            i_processed += n_comp
    matches = spr.csr_array(similarity > match_threshold)
    n_graph_components, labels = spr.csgraph.connected_components(
        matches, directed=False
    )
    merge_history = []
    current_ind = 0
    for i_model, comp in enumerate(component_matrices):
        n_model_components = comp.shape[0]
        merge_history.append(
            labels[current_ind : current_ind + n_model_components]
        )
        current_ind += n_model_components
    new_components = merge_from_history(
        component_matrices, merge_history, weights=weights, agg=agg
    )
    if topic_embeddings is None:
        return new_components, merge_history
    new_embeddings = merge_from_history(
        topic_embeddings, merge_history, weights=weights, agg=agg
    )
    return new_components, merge_history, new_embeddings


def keep_first(a, axis=0):
    return np.take(a, 0, axis=axis)


def asymmetric_merge(
    component_matrices: list[np.ndarray],
    weights=None,
    match_threshold: float = 0.7,
    agg=keep_first,
    sim_fn=safe_cosine,
    topic_embeddings=None,
) -> tuple[np.ndarray, MergeHistory]:
    """Performs an asymmetric merge on a number of topic models.

    Parameters
    ----------
    component_matrices: list[np.ndarray]
        List of topic representations from all topic models.
    weights: Sequence, None
        Weight for each of the models in case of a weighted merge.
    match_threshold: float, default 0.7
        Similarity threshold above which two topics will be considered a match.
    agg: Callable, default np.average
        Aggregation method to use for matching topics.
    sim_fn: Callable, default cosine_similarity
        Function to produce the similarity matrix.
    allow_within_model_match: bool, default False
        Determines whether matches can happen within models.
    topic_embeddings: list[np.ndarray]
        Topic representations for calculating similarity.
        This can be useful when you want to merge components in vocab space,
        but want to calculate similarities based on embedding space.

    Returns
    -------
    ndarray of shape (n_new_topics, n_dims)
        New topic representations merged from the old ones.
    MergeHistory (list[list[int]])
        Indicates which joint topic the original topics were merged into.
        The data structure is a list of lists, where each list contains the indices of the joint topics
        each of the original topics were merged into.
        e.g. [[0,2,1], [3,2]] would indicate that the topics of the first model were merged into
        topics 0, 2 and 1 in the joint model, while the second model's topics were merged into 3 and 2.
    OPTIONAL: ndarray of shape (n_new_topics, n_dims)
        New topic embeddings.
    """
    merge_history = []
    components = np.copy(component_matrices[0])
    if topic_embeddings is not None:
        embeddings = np.copy(topic_embeddings[0])
    # First components will just be kept
    merge_history.append(list(range(components.shape[0])))
    for i_model, incoming_components in enumerate(component_matrices[1:]):
        n_current = components.shape[0]
        _merge_inst = []
        if topic_embeddings is None:
            similarity = sim_fn(components, incoming_components)
        else:
            incoming_embeddings = topic_embeddings[i_model]
            similarity = sim_fn(embeddings, incoming_embeddings)
        maxsim_ind = np.argmax(similarity.T, axis=1)
        to_add = []
        for i_new_comp, ind_most_similar_old in enumerate(maxsim_ind):
            if similarity[ind_most_similar_old, i_new_comp] > match_threshold:
                components[ind_most_similar_old] = safe_agg(
                    np.concatenate(
                        [
                            components[[ind_most_similar_old], :],
                            incoming_components[[i_new_comp], :],
                        ],
                        axis=0,
                    ),
                    agg=agg,
                    weights=weights,
                    axis=0,
                )
                _merge_inst.append(ind_most_similar_old)
                if topic_embeddings is not None:
                    embeddings[ind_most_similar_old] = safe_agg(
                        np.concatenate(
                            [
                                embeddings[[ind_most_similar_old], :],
                                incoming_embeddings[[i_new_comp], :],
                            ],
                            axis=0,
                        ),
                        agg=agg,
                        weights=weights,
                        axis=0,
                    )
            else:
                _merge_inst.append(n_current + len(to_add))
                to_add.append(i_new_comp)
        components = np.concatenate(
            [components, incoming_components[to_add]], axis=0
        )
        if topic_embeddings is not None:
            embeddings = np.concatenate(
                [embeddings, incoming_embeddings[to_add]], axis=0
            )
        merge_history.append(_merge_inst)
    if topic_embeddings is None:
        return components, merge_history
    else:
        return components, merge_history, embeddings


NAMED_METHODS = {
    "keep_first": partial(
        asymmetric_merge, sim_fn=safe_cosine, agg=keep_first
    ),
    "asymmetric_mean": partial(
        asymmetric_merge, sim_fn=safe_cosine, agg=nanaverage
    ),
    "symmetric_mean": partial(
        symmetric_merge, sim_fn=safe_cosine, agg=nanaverage
    ),
}


def get_merge_fn(merge_method: str | Callable) -> Callable:
    """Finds merge method by name, or returns the input argument if it's a function."""
    if isinstance(merge_method, str):
        if merge_method in NAMED_METHODS:
            return NAMED_METHODS[merge_method]
        else:
            available_methods = ", ".join(NAMED_METHODS.keys())
            raise ValueError(
                f"Named merge method {merge_method} not found."
                f" Available methods are {available_methods}"
            )
    else:
        return merge_method


class ExponentialDecayMerge:
    def __init__(self, merge_method: Callable, decay_constant=10):
        self.i_merge = 0
        self.decay_constant = 1
        self.merge_method = merge_method
        self.merge_fn = get_merge_fn(self.merge_method)

    def __call__(self, component_matrices, weights=None, match_threshold=0.7):
        if weights is None:
            weights = np.ones(len(component_matrices))
        # Leftmost is assumed to be the current model
        lr = np.exp(
            -self.decay_constant
            * (np.arange(len(component_matrices) - 1) + self.i_merge)
        )
        lr = np.insert(lr, 0, 1)
        decayed_weights = weights * lr
        self.i_merge += len(component_matrices) - 1
        return self.merge_fn(
            component_matrices,
            weights=decayed_weights,
            match_threshold=match_threshold,
        )


def merge_hierarchies(
    left: TopicNode,
    right: TopicNode,
    new_components: np.ndarray,
    merge_history: MergeHistory,
    joint_vectorizer: UnionVectorizer,
    joint_model,
) -> TopicNode:
    """Merges all hierarchies from the merged models into one."""
    assert len(merge_history) == 2, "The merge has to be over two models."
    left_to_joint, right_to_joint = merge_history
    flat_merge_hist = list(itertools.chain.from_iterable(merge_history))
    unique_components = np.unique(flat_merge_hist)
    assert unique_components[0] == 0

    def _change_model(node):
        node.model = joint_model
        # Since the number of documents is very probably different in the two models,
        # we remove the document topic vector from each node
        node.document_topic_vector = None
        return node

    def _pad_left(node):
        if node.word_importance is not None:
            padded = np.ravel(
                joint_vectorizer.pad_left(
                    np.atleast_2d(node.word_importance), pad_value=np.nan
                )
            )
            node.word_importance = padded
        return node

    def _pad_right(node):
        if node.word_importance is not None:
            padded = np.ravel(
                joint_vectorizer.pad_right(
                    np.atleast_2d(node.word_importance), pad_value=np.nan
                )
            )
            node.word_importance = padded
        return node

    left = left.map(_change_model).map(_pad_left)
    right = right.map(_change_model).map(_pad_right)
    if right.children is None:
        return left.copy(deep=True)
    if left.children is None:
        return right.copy(deep=True)
    flat_children = [*left.children, *right.children]
    flat_to_joint = [*left_to_joint, *right_to_joint]
    joint_to_flat = defaultdict(list)
    for flat_idx, joint_idx in enumerate(flat_to_joint):
        joint_to_flat[joint_idx].append(flat_idx)
    joint_children = []
    for (i_topic, old_children), comp in zip(
        joint_to_flat.items(), new_components
    ):

        def _indent(node):
            node.path = (i_topic, *node.path)
            return node

        children = None
        if len(old_children) > 1:
            children = [
                flat_children[i_child].map(_indent) for i_child in old_children
            ]
        child = TopicNode(
            joint_model,
            path=(i_topic,),
            word_importance=comp,
            document_topic_vector=None,
            children=children,
        )
        joint_children.append(child)
    return TopicNode(joint_model, path=(), children=joint_children)
