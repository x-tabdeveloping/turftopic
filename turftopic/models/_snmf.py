"""This file implements semi-NMF, where doc_topic proportions are not allowed to be negative, but components are unbounded."""

import warnings
from functools import partial
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from tqdm import trange

EPSILON = np.finfo(np.float32).eps

try:
    import jax.numpy as jnp
    from jax import jit
except ModuleNotFoundError:
    warnings.warn("JAX not found, continuing with NumPy implementation.")
    jnp = np

    # Dummy JIT as the identity function
    def jit(f):
        return f


def init_G(
    X, n_components: int, constant=0.2, random_state=None
) -> np.ndarray:
    """Returns W"""
    kmeans = KMeans(n_components, random_state=random_state).fit(X.T)
    # n_components, n_columns
    G = label_binarize(kmeans.labels_, classes=np.arange(n_components))
    return G + constant


def separate(A):
    abs_A = jnp.abs(A)
    pos = (abs_A + A) / 2
    neg = (abs_A - A) / 2
    return pos, neg


def update_F(X, G):
    return X @ G @ jnp.linalg.inv(G.T @ G)


def update_G(X, G, F, sparsity=0):
    pos_xtf, neg_xtf = separate(X.T @ F)
    pos_gftf, neg_gftf = separate(G @ (F.T @ F))
    numerator = pos_xtf + neg_gftf
    denominator = neg_xtf + pos_gftf
    denominator += sparsity
    denominator = jnp.maximum(denominator, EPSILON)
    delta_G = jnp.sqrt(numerator / denominator)
    G *= delta_G
    return G


def rec_err(X, F, G):
    err = X - (F @ G.T)
    return jnp.linalg.norm(err)


@jit
def step(G, F, X, sparsity=0):
    G = update_G(X.T, G, F, sparsity)
    F = update_F(X.T, G)
    error = rec_err(X.T, F, G)
    return G, F, error


class SNMF(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_components: int,
        tol: float = 1e-5,
        max_iter: int = 200,
        progress_bar: bool = True,
        random_state: Optional[int] = None,
        sparsity: float = 0.0,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.progress_bar = progress_bar
        self.random_state = random_state
        self.sparsity = sparsity
        self.verbose = verbose

    def fit_transform(self, X: np.ndarray, y=None):
        G = init_G(X.T, self.n_components, random_state=self.random_state)
        F = update_F(X.T, G)
        error_at_init = rec_err(X.T, F, G)
        prev_error = error_at_init
        _step = partial(step, sparsity=self.sparsity, X=X)
        for i in trange(
            self.max_iter,
            desc="Iterative updates.",
            disable=not self.progress_bar,
        ):
            G, F, error = _step(G, F)
            difference = prev_error - error
            if (error < error_at_init) and (
                (prev_error - error) / error_at_init
            ) < self.tol:
                if self.verbose:
                    print(f"Converged after {i} iterations")
                self.n_iter_ = i
                break
            prev_error = error
            if self.verbose:
                print(
                    f"Iteration: {i}, Error: {error}, init_error: {error_at_init}, difference from previous: {difference}"
                )
        else:
            warnings.warn(
                "SNMF did not converge, try specifying a higher max_iter."
            )
        self.components_ = np.array(F.T)
        self.reconstruction_err_ = error
        self.n_iter_ = i
        return np.array(G)

    def fit_timeslice(self, X_t: np.ndarray, G_t: np.ndarray):
        F = update_F(X_t.T, G_t)
        return F.T

    def transform(self, X: np.ndarray):
        G = jnp.maximum(X @ jnp.linalg.pinv(self.components_), 0)
        return np.array(G)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_components)
            Transformed data matrix.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Returns a data matrix of the original shape.
        """
        return X @ self.components_
