from functools import cache
from typing import Callable

import numpy as np
from scipy import optimize
from scipy.sparse import issparse


def decomposition_gaussian_bic(
    n_components: int, decomp_class, X, random_state: int = 42
) -> float:
    """Computes Bayesian information criterion for
    a decomposition model with assumed Gaussian noise.

    Parameters
    ----------
    n_components: int
        Number of components in the model.
    decomp_class
        Class of the decomposition model, e.g. `decomp_class=FastICA`.
    X: array
        Training data to decompose.
    random_state: int, default 42
        Seed to pass to the decomposition algorithm.
    """
    decomp = decomp_class(n_components=n_components, random_state=42)
    doc_topic = decomp.fit_transform(X)
    if not hasattr(decomp, "reconstruction_err_"):
        X_reconstruction = decomp.inverse_transform(doc_topic)
        # Computing residual sum of squares
        if issparse(X):
            X = np.asarray(X.todense())
        rss = np.sum(np.square(X - X_reconstruction))
    else:
        rss = np.square(decomp.reconstruction_err_)
    n_params = decomp.components_.size + np.prod(doc_topic.shape)
    n_datapoints = np.prod(X.shape)
    # Computing Bayesian information criterion assuming IID
    bic = n_params * np.log(n_datapoints) + n_datapoints * np.log(
        rss / n_datapoints
    )
    return bic


def optimize_n_components(
    f_ic: Callable[int, float], min_n: int = 2, max_n: int = 250, verbose=False
) -> int:
    """Optimizes the nuber of components using the Brent minimum finding
    algorithm given an information criterion.
    The bracket is found by exponentially incrementing n_components.

    Parameters
    ----------
    f_ic: Callable[int, float]
        Information criterion given N components.
    min_n: int
        Minimum value of N.
    max_n: int
        Maximum value of N, in case the algorithm doesn't converge.
    """
    if verbose:
        print("Optimizing N based on an information criterion...")

    # Caching and adding debugging statements
    @cache
    def _f_ic(n_components) -> float:
        # Making sure n_components is an integer
        # The optimization algorithm definitely give it a float
        n_components = int(n_components)
        val = f_ic(n_components)
        if verbose:
            print(f" - IC(N={n_components})={val:.2f}")
        return val

    # Finding bracket
    if verbose:
        print(" - Finding bracket...")
    low = min_n
    n_comp = 2
    while not _f_ic(n_comp) < _f_ic(min_n):
        n_comp += 1
        if n_comp >= 10:
            if verbose:
                print(
                    " - Couldn't find lower value than n=1 up to n=10, stopping."
                )
            return 1
    middle = n_comp
    current = _f_ic(middle)
    inc = 5
    while not current > _f_ic(middle):
        n_comp += inc
        if n_comp >= max_n:
            if verbose:
                print(f" - Bracket didn't converge, returning max N: {max_n}")
            return max_n
        current = _f_ic(n_comp)
        if current < _f_ic(middle):
            low = n_comp - inc
            middle = n_comp
        inc *= 2
    bracket = low, middle, n_comp
    if verbose:
        print(f" - Running optimization with bracket: {bracket}")
    # Optimizing
    res = optimize.minimize_scalar(
        _f_ic,
        method="brent",
        bracket=(low, middle, n_comp),
        options=dict(xtol=0.2),
    )
    if verbose:
        print(f" - Converged after {res.nit} iterations.")
    return int(res.x)
