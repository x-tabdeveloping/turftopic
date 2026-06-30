import numpy as np
import scipy.sparse as spr
from sklearn.base import BaseEstimator, TransformerMixin


class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, b: float = 0.7, k1: float = 8):
        self.b = b
        self.k1 = k1

    def fit(self, X, y=None):
        self.N_ = X.shape[0]
        self.avgdl_ = X.sum(axis=1).mean()
        self.term_freq_ = np.ravel(np.asarray((X > 0).sum(axis=0)))
        self.idf_ = np.log(
            (self.N_ - self.term_freq_ + 0.5) / (self.term_freq_ + 0.5)
        )
        return self

    def transform(self, X):
        if spr.issparse(X):
            X = spr.csr_array(X)
        d_len = np.ravel(np.asarray(X.sum(axis=1)))
        K_D = 1 - self.b + self.b * d_len / self.avgdl_
        return (
            self.idf_[None, :]
            * (X * (self.k1 + 1))
            / (X + self.k1 * K_D[:, None])
        )

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
