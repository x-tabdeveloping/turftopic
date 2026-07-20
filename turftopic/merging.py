import datetime
import itertools
import warnings
from collections import defaultdict
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


MergeHistory = list[list[int]]


def merge_from_history(
    component_matrices: list[np.ndarray],
    merge_history: MergeHistory,
    weights=None,
    agg=np.average,
) -> np.ndarray:
    stacked_components = np.concatenate(component_matrices, axis=0)
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


def symmetric_merge(
    component_matrices: list[np.ndarray],
    weights=None,
    match_threshold: float = 0.7,
    agg=np.average,
    sim_fn=cosine_similarity,
    allow_within_model_match=False,
) -> tuple[np.ndarray, MergeHistory]:
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
    return new_components, merge_history


def keep_first(a, axis=0):
    return np.take(a, 0, axis=axis)


def asymmetric_merge(
    component_matrices: list[np.ndarray],
    weights=None,
    match_threshold: float = 0.7,
    agg=keep_first,
    sim_fn=cosine_similarity,
) -> tuple[np.ndarray, MergeHistory]:
    merge_history = []
    components = np.copy(component_matrices[0])
    # First components will just be kept
    merge_history.append(list(range(components.shape[0])))
    for incoming_components in component_matrices[1:]:
        n_current = components.shape[0]
        _merge_inst = []
        similarity = sim_fn(components, incoming_components)
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
            else:
                to_add.append(i_new_comp)
                _merge_inst.append(n_current + len(to_add))
        components = np.concatenate(
            [components, incoming_components[to_add]], axis=0
        )
        merge_history.append(_merge_inst)
    return components, merge_history


NAMED_METHODS = {
    "keep_first": partial(
        asymmetric_merge, sim_fn=cosine_similarity, agg=keep_first
    ),
    "asymmetric_mean": partial(
        asymmetric_merge, sim_fn=cosine_similarity, agg=np.nanmean
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
