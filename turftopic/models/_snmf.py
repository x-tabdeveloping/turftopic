"""This file implements semi-NMF, where doc_topic proportions are not allowed to be negative, but components are unbounded."""

import warnings
from functools import partial
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, copy
from sklearn.cluster import KMeans
from tqdm import trange

from turftopic.utils import safe_binarize

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
    G = safe_binarize(kmeans.labels_, classes=np.arange(n_components))
    return G + constant


def add_G(G, n_add: int, constant=0.2):
    new_components = jnp.broadcast_to(
        jnp.mean(G, axis=1), (n_add, G.shape[0])
    ).T
    new_components = jnp.where(new_components < 0, constant, new_components)
    return jnp.concatenate([G, new_components], axis=1)


def separate(A):
    abs_A = jnp.abs(A)
    pos = (abs_A + A) / 2
    neg = (abs_A - A) / 2
    return pos, neg


def update_F(X, G, F, n_freeze=None):
    F_new = X @ G @ jnp.linalg.pinv(G.T @ G)
    if n_freeze is None:
        return F_new
    else:
        return jnp.concatenate([F[:, :n_freeze], F_new[:, n_freeze:]], axis=1)


def update_G(X, G, F, sparsity=0, n_freeze=None):
    G_new = G
    pos_xtf, neg_xtf = separate(X.T @ F)
    pos_gftf, neg_gftf = separate(G_new @ (F.T @ F))
    numerator = pos_xtf + neg_gftf
    denominator = neg_xtf + pos_gftf
    denominator += sparsity
    denominator = jnp.maximum(denominator, EPSILON)
    delta_G_new = jnp.sqrt(numerator / denominator)
    G_new *= delta_G_new
    G_new = G_new / jnp.maximum(jnp.linalg.norm(G_new), EPSILON)
    if n_freeze is None:
        return G_new
    else:
        return jnp.concatenate([G[:n_freeze], G_new[n_freeze:]], axis=0)


def rec_err(X, F, G):
    err = X - (F @ G.T)
    return jnp.linalg.norm(err)


def step(G, F, X, sparsity=0, n_freeze=None):
    G = update_G(X.T, G, F, sparsity, n_freeze=n_freeze)
    F = update_F(X.T, G, F, n_freeze=n_freeze)
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
        F = update_F(X.T, G, F=None)
        self.error_at_init = rec_err(X.T, F, G)
        prev_error = self.error_at_init
        _step = jit(partial(step, sparsity=self.sparsity, X=X, n_freeze=0))
        for i in trange(
            self.max_iter,
            desc="Iterative updates.",
            disable=not self.progress_bar,
        ):
            G, F, error = _step(G, F)
            difference = prev_error - error
            if (error < self.error_at_init) and (
                (prev_error - error) / self.error_at_init
            ) < self.tol:
                if self.verbose:
                    print(f"Converged after {i} iterations")
                self.n_iter_ = i
                break
            prev_error = error
            if self.verbose:
                print(
                    f"Iteration: {i}, Error: {error}, init_error: {self.error_at_init}, difference from previous: {difference}"
                )
        else:
            warnings.warn(
                "SNMF did not converge, try specifying a higher max_iter."
            )
        self.components_ = np.array(F.T)
        self.reconstruction_err_ = error
        self.n_datapoints_ = X.shape[0]
        self.n_iter_ = i
        return np.array(G)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def bic(self, X):
        rss = np.square(self.rec_err(X))
        n_docs, n_dims = X.shape
        # BIC1 from https://pmc.ncbi.nlm.nih.gov/articles/PMC9181460/
        bic1 = np.log(rss) + self.n_components * (
            (n_docs + n_dims) / (n_docs * n_dims)
        ) * np.log((n_docs * n_dims) / (n_docs + n_dims))
        return bic1

    def fit_new_components(self, X: np.ndarray, n_new_components: int):
        G_old = self.transform(X)
        old_n_components = self.n_components
        G = add_G(G_old, n_add=n_new_components)
        F = update_F(X.T, G, self.components_.T, n_freeze=old_n_components)
        prev_error = rec_err(X.T, F, G)
        _step = jit(
            partial(
                step, sparsity=self.sparsity, X=X, n_freeze=self.n_components
            )
        )
        for i in trange(
            self.max_iter,
            desc="Iterative updates.",
            disable=not self.progress_bar,
        ):
            G, F, error = _step(G, F)
            difference = prev_error - error
            if (error < self.error_at_init) and (
                (prev_error - error) / self.error_at_init
            ) < self.tol:
                if self.verbose:
                    print(f"Converged after {i} iterations")
                self.n_iter_ = i
                break
            prev_error = error
            if self.verbose:
                print(
                    f"Iteration: {i}, Error: {error}, init_error: {self.error_at_init}, difference from previous: {difference}"
                )
        self.components_ = np.array(F.T)
        self.n_iter_ = i
        self.n_components = old_n_components + n_new_components
        self.reconstruction_err_ = error
        return self

    def rec_err(self, X):
        G = self.transform(X)
        F = self.components_.T
        return rec_err(X.T, F, G)

    def fit_timeslice(self, X_t: np.ndarray, G_t: np.ndarray):
        F = update_F(X_t.T, G_t, F=None)
        return F.T

    def transform(self, X: np.ndarray, F=None):
        G = init_G(
            X.T,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        if F is None:
            F = self.components_.T
        update = jit(lambda G: update_G(X.T, G, F, sparsity=self.sparsity))
        error_at_init = rec_err(X.T, F, G)
        prev_error = error_at_init
        for i in range(self.max_iter):
            G = update(G)
            err = rec_err(X.T, F, G)
            if (err < error_at_init) and (
                (prev_error - err) / error_at_init
            ) < self.tol:
                if self.verbose:
                    print(f"Converged after {i} iterations")
                break
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
