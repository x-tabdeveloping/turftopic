"""This is an encode-only implementation of the TopK autoencoder.
The training code lives in the x-tabdeveloping/latent_terms GitHub repo"""

import warnings
from functools import partial
from typing import Optional

import numpy as np
import scipy.sparse as spr
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import trange

try:
    import jax.numpy as jnp
    from jax import jit
    from jax.lax import top_k
except ModuleNotFoundError:
    warnings.warn("JAX not found, continuing with NumPy implementation.")
    jnp = np

    # Dummy JIT as the identity function
    def jit(f):
        return f

    # NumPy implementation of the TopK activation function.
    def top_k(a, k, *, axis=-1):
        if axis is None:
            axis_size = a.size
        else:
            axis_size = a.shape[axis]
        index_array = np.argpartition(a, axis_size - k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k) - 1, axis=axis)
        topk_values = np.take_along_axis(a, topk_indices, axis=axis)
        return topk_values, topk_indices


def top_k_activation(z, k: int):
    values, indices = top_k(z, k=k, axis=-1)
    threshold = jnp.min(values, axis=-1)
    condition = threshold[:, None] <= z
    return jnp.where(condition, z, 0)


def encode(params, x, k: int):
    z = x @ params["W_e"] + params["b_e"]
    return top_k_activation(z, k)


class TopKAutoEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_latent: int = 32768,
        top_k: int = 16,
        lr: float = 1e-3,
        batch_size: int = 4096,
        n_epochs: int = 10,
        alpha: float = 0.03,
        show_progress_bar: bool = True,
        random_state: Optional[int] = None,
    ):
        self.random_state = random_state
        self.n_latent = n_latent
        self.lr = lr
        self.alpha = alpha
        self.top_k = top_k
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.show_progress_bar = show_progress_bar

    def fit(self, X, y=None):
        # Training is implemented here: https://github.com/x-tabdeveloping/latent_terms
        return self

    def to_dict(self) -> dict:
        return dict(
            attr=self.get_params(),
            params=self._params,
            loss_curve=self.loss_curve_,
        )

    @classmethod
    def from_dict(cls, data):
        obj = cls(**data["attr"])
        params = data["params"]
        obj.coef_ = np.array(params["W_e"])
        obj.coef_d_ = np.array(params["W_d"])
        obj.intercept_ = np.array(params["b_e"])
        obj.intercept_d_ = np.array(params["b_d"])
        obj.loss_curve_ = data["loss_curve"]
        return obj

    @property
    def _params(self):
        return {
            "W_e": self.coef_,
            "b_e": self.intercept_,
            "W_d": self.coef_d_,
            "b_d": self.intercept_d_,
        }

    def transform(self, X):
        if spr.issparse(X):
            X = X.todense()
        Z = []
        _encode = jit(partial(encode, params=self._params, k=self.top_k))
        for batch_start in trange(
            0,
            X.shape[0],
            self.batch_size,
            leave=False,
            desc="Going through all batches",
            disable=not self.show_progress_bar,
        ):
            batch_end = batch_start + self.batch_size
            batch_x = X[batch_start:batch_end]
            batch_z = _encode(x=batch_x)
            Z.append(spr.csr_array(batch_z))
        return spr.vstack(Z, format="csr")

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
