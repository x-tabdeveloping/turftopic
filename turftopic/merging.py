import warnings
from functools import partial
from typing import Callable

import numpy as np
import scipy.sparse as spr
from sklearn.metrics.pairwise import cosine_similarity


def safe_agg(a, agg, weights, axis=0):
    try:
        return agg(a, weights=weights, axis=axis)
    except Exception:
        return agg(a, axis=axis)


def symmetric_merge(
    component_matrices: list[np.ndarray],
    weights=None,
    match_threshold: float = 0.7,
    agg=np.average,
    sim_fn=cosine_similarity,
    allow_within_model_match=False,
) -> np.ndarray:
    stacked_components = np.concatenate(component_matrices, axis=0)
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
    n_dims = stacked_components.shape[1]
    new_components = np.zeros((n_graph_components, n_dims))
    for i_graph_component in np.arange(n_graph_components):
        new_components[i_graph_component, :] = safe_agg(
            stacked_components[labels == i_graph_component],
            agg=agg,
            weights=weights,
            axis=0,
        )
    return new_components


def keep_first(a, axis=0):
    return np.take(a, 0, axis=axis)


def assymmetric_merge(
    component_matrices: list[np.ndarray],
    weights=None,
    match_threshold: float = 0.7,
    agg=keep_first,
    sim_fn=cosine_similarity,
):
    components = np.copy(component_matrices[0])
    for incoming_components in component_matrices[1:]:
        similarity = sim_fn(components, incoming_components)
        match = similarity > match_threshold
        for i_comp, old_matches_new in enumerate(match):
            if np.any(old_matches_new):
                components[i_comp] = safe_agg(
                    np.concatenate(
                        [
                            components[[i_comp], :],
                            incoming_components[old_matches_new],
                        ],
                        axis=0,
                    ),
                    agg=agg,
                    weights=weights,
                    axis=0,
                )
        to_add, *_ = np.nonzero(match.sum(axis=0) <= 0)
        components = np.concatenate(
            [components, incoming_components[to_add]], axis=0
        )
    return components


NAMED_METHODS = {
    "keep_first": partial(
        assymmetric_merge, sim_fn=cosine_similarity, agg=keep_first
    ),
    "asymmetric_mean": partial(
        assymmetric_merge, sim_fn=cosine_similarity, agg=np.nanmean
    ),
    "symmetric_mean": partial(
        symmetric_merge, sim_fn=cosine_similarity, agg=np.nanmean
    ),
}


def get_merge_fn(merge_method: str | Callable) -> Callable:
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
