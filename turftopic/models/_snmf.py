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
    from jax.lax import while_loop
except ModuleNotFoundError:
    warnings.warn("JAX not found, continuing with NumPy implementation.")
    jnp = np

    # Dummy JIT as the identity function
    def jit(f):
        return f

    # Naive Python implementation of JAX's while_loop
    def while_loop(cond_fun, body_fun, init_val):
        val = init_val
        while cond_fun(val):
            val = body_fun(val)
        return val


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


def inference_loop(
    X,
    G,
    F,
    sparsity=0,
    n_freeze=None,
    tol=1e-5,
    max_iter=200,
    track_convergence=True,
    freeze_F=False,
):
    init_error = rec_err(X.T, F, G)
    init_state = {
        "G": G,
        "F": F,
        "error": init_error,
        "error_diff": np.inf,
        "n_iter": 0,
    }

    def _cond_fn(state):
        keep_running = state["n_iter"] < max_iter
        if track_convergence:
            converged = jnp.logical_and(
                state["error"] < init_error,
                (state["error_diff"] / init_error) < tol,
            )
            keep_running = jnp.logical_and(
                keep_running, (jnp.logical_not(converged))
            )
        return keep_running

    def _body_fn(state):
        if not freeze_F:
            G, F, new_error = step(
                state["G"],
                state["F"],
                X,
                sparsity=sparsity,
                n_freeze=n_freeze,
            )
        else:
            G = update_G(
                X.T,
                state["G"],
                state["F"],
                sparsity=sparsity,
                n_freeze=n_freeze,
            )
            F = state["F"]
            new_error = rec_err(X.T, F, G)
        return {
            "G": G,
            "F": F,
            "error": new_error,
            "error_diff": state["error"] - new_error,
            "n_iter": state["n_iter"] + 1,
        }

    return while_loop(_cond_fn, _body_fn, init_state)


def infer_bic(G_init, F, X, sparsity, tol, max_iter):
    last_state = inference_loop(
        X=X,
        G=G_init,
        F=F,
        sparsity=sparsity,
        n_freeze=None,
        tol=tol,
        max_iter=max_iter,
        track_convergence=True,
        freeze_F=True,
    )
    rss = rec_err(X.T, F, last_state["G"])
    n_components = G_init.shape[1]
    n_docs, n_dims = X.shape
    # BIC1 from https://pmc.ncbi.nlm.nih.gov/articles/PMC9181460/
    bic1 = jnp.log(rss) + n_components * (
        (n_docs + n_dims) / (n_docs * n_dims)
    ) * jnp.log((n_docs * n_dims) / (n_docs + n_dims))
    return bic1


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
        last_state = inference_loop(
            X=X,
            G=G,
            F=F,
            sparsity=self.sparsity,
            n_freeze=None,
            tol=self.tol,
            max_iter=self.max_iter,
            track_convergence=True,
            freeze_F=False,
        )
        self.components_ = np.array(last_state["F"].T)
        self.reconstruction_err_ = float(last_state["error"])
        self.n_datapoints_ = X.shape[0]
        self.n_iter_ = int(last_state["n_iter"])
        return np.array(last_state["G"])

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def bic(self, X, F=None, G=None):
        if F is None:
            F = self.components_.T
        n_components = F.shape[1]
        if G is None:
            G = self.transform(X, F)
        rss = rec_err(X.T, F, G)
        n_components = G.shape[1]
        n_docs, n_dims = X.shape
        # BIC1 from https://pmc.ncbi.nlm.nih.gov/articles/PMC9181460/
        bic1 = jnp.log(rss) + n_components * (
            (n_docs + n_dims) / (n_docs * n_dims)
        ) * jnp.log((n_docs * n_dims) / (n_docs + n_dims))
        return float(bic1)

    def _fit_new(self, X, n_new: int):
        G_old = self.transform(X)
        old_n_components = self.n_components
        G = add_G(G_old, n_add=n_new)
        F = update_F(X.T, G, self.components_.T, n_freeze=old_n_components)
        last_state = inference_loop(
            X=X,
            G=G,
            F=F,
            sparsity=self.sparsity,
            n_freeze=old_n_components,
            tol=self.tol,
            max_iter=self.max_iter,
            track_convergence=True,
            freeze_F=False,
        )
        return last_state

    def fit_new_components(self, X: np.ndarray, n_new_components: int):
        old_n_components = self.n_components
        last_state = self._fit_new(X, n_new_components)
        self.components_ = np.array(last_state["F"].T)
        self.n_iter_ = int(last_state["n_iter"])
        self.n_components = old_n_components + n_new_components
        self.reconstruction_err_ = float(last_state["error"])
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
        last_state = inference_loop(
            X=X,
            G=G,
            F=F,
            sparsity=self.sparsity,
            n_freeze=None,
            tol=self.tol,
            max_iter=self.max_iter,
            track_convergence=True,
            freeze_F=True,
        )
        return np.array(last_state["G"])

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
