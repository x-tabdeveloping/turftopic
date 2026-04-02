from typing import Callable, Optional

import numpy as np
from sklearn.base import TransformerMixin

Lengths = list[int]


def flatten_repr(
    repr: list[np.ndarray],
) -> tuple[np.ndarray, Lengths]:
    """Flattens ragged array to normal array.

    Parameters
    ----------
    repr: list[ndarray]
        Ragged representation array.

    Returns
    -------
    flat_repr: ndarray
        Flattened representation array.
    lengths: list[int]
        Length of each document in the corpus.
    """
    lengths = [r.shape[0] for r in repr]
    return np.concatenate(repr, axis=0), lengths


def unflatten_repr(
    flat_repr: np.ndarray, lengths: Lengths
) -> list[np.ndarray]:
    """Unflattens flat array to ragged array.

    Parameters
    ----------
    flat_repr: ndarray
        Flattened representation array.
    lengths: list[int]
        Length of each document in the corpus.

    Returns
    -------
    repr: list[ndarray]
        Ragged representation array.

    """
    repr = []
    start_index = 0
    for length in lengths:
        repr.append(flat_repr[start_index:length])
        start_index += length
    return repr


def pool_flat(flat_repr: np.ndarray, lengths: Lengths, agg=np.mean):
    pooled = []
    start_index = 0
    for length in lengths:
        pooled.append(agg(flat_repr[start_index:length], axis=0))
        start_index += length
    return np.stack(pooled)


class TokenLevel(TransformerMixin):
    def __init__(
        self,
        model: TransformerMixin,
        batch_size: int = 32,
        pooling: Optional[Callable] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.pooling = pooling

    def transform(
        self, raw_documents: list[str], embeddings: list[np.ndarray] = None
    ):
        if embeddings is None:
            embeddings = self.model.encoder.encode_tokens(
                raw_documents, batch_size=self.batch_size
            )
        flat_embeddings, lengths = flatten_repr(embeddings)
        out_array = self.model.transform(
            raw_documents, embeddings=flat_embeddings
        )
        if self.pooling is None:
            return unflatten_repr(out_array, lengths)
        else:
            return pool_flat(out_array, lengths)

    def fit_transform(
        self,
        raw_documents: list[str],
        y=None,
        embeddings: list[np.ndarray] = None,
    ):
        if embeddings is None:
            embeddings = self.model.encoder.encode_tokens(
                raw_documents, batch_size=self.batch_size
            )
        flat_embeddings, lengths = flatten_repr(embeddings)
        out_array = self.model.fit_transform(
            raw_documents, y, embeddings=flat_embeddings
        )
        if self.pooling is None:
            return unflatten_repr(out_array, lengths)
        else:
            return pool_flat(out_array, lengths)


class Windowed(TransformerMixin):
    def __init__(
        self,
        model: TransformerMixin,
        batch_size: int = 32,
        pooling: Optional[Callable] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.pooling = pooling

    def transform(
        self, raw_documents: list[str], embeddings: list[np.ndarray] = None
    ):
        if embeddings is None:
            embeddings = self.model.encoder.encode_tokens(
                raw_documents, batch_size=self.batch_size
            )
        flat_embeddings, lengths = flatten_repr(embeddings)
        out_array = self.model.transform(
            raw_documents, embeddings=flat_embeddings
        )
        if self.pooling is None:
            return unflatten_repr(out_array, lengths)
        else:
            return pool_flat(out_array, lengths)

    def fit_transform(
        self,
        raw_documents: list[str],
        y=None,
        embeddings: list[np.ndarray] = None,
    ):
        if embeddings is None:
            embeddings = self.model.encoder.encode_tokens(
                raw_documents, batch_size=self.batch_size
            )
        flat_embeddings, lengths = flatten_repr(embeddings)
        out_array = self.model.fit_transform(
            raw_documents, y, embeddings=flat_embeddings
        )
        if self.pooling is None:
            return unflatten_repr(out_array, lengths)
        else:
            return pool_flat(out_array, lengths)
