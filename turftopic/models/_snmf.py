"""This file implements semi-NMF, where doc_topic proportions are not allowed to be negative, but components are unbounded."""

import warnings
from typing import Optional

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
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


@jit
def separate(A):
    abs_A = jnp.abs(A)
    pos = (abs_A + A) / 2
    neg = (abs_A - A) / 2
    return pos, neg


@jit
def update_F(X, G):
    return X @ G @ jnp.linalg.inv(G.T @ G)


@jit
def update_G(X, G, F, l1_reg=0):
    pos_xtf, neg_xtf = separate(X.T @ F)
    pos_gftf, neg_gftf = separate(G @ (F.T @ F))
    numerator = pos_xtf + neg_gftf
    denominator = neg_xtf + pos_gftf
    denominator += l1_reg
    denominator = jnp.maximum(denominator, EPSILON)
    delta_G = jnp.sqrt(numerator / denominator)
    G *= delta_G
    return G


@jit
def rec_err(X, F, G):
    err = X - (F @ G.T)
    return jnp.linalg.norm(err)


class SNMF(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_components: int,
        tol: float = 1e-5,
        max_iter: int = 200,
        progress_bar: bool = True,
        random_state: Optional[int] = None,
        l1_reg: float = 0.5,
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.progress_bar = progress_bar
        self.random_state = random_state
        self.l1_reg = l1_reg

    def fit_transform(self, X: np.ndarray, y=None):
        G = init_G(X.T, self.n_components)
        F = update_F(X.T, G)
        error_at_init = rec_err(X.T, F, G)
        prev_error = error_at_init
        for i in trange(
            self.max_iter,
            desc="Iterative updates.",
            disable=not self.progress_bar,
        ):
            G = update_G(X.T, G, F, self.l1_reg)
            F = update_F(X.T, G)
            error = rec_err(X.T, F, G)
            difference = prev_error - error
            if (error < error_at_init) and (
                (prev_error - error) / error_at_init
            ) < self.tol:
                print(f"Converged after {i} iterations")
                self.n_iter_ = i
                break
            prev_error = error
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

    def transform(self, X: np.ndarray):
        G = jnp.maximum(X @ jnp.linalg.pinv(self.components_), 0)
        return np.array(G)
