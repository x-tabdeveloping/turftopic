from typing import Callable, Optional

import numpy as np
from sklearn.base import TransformerMixin

from turftopic.encoders.contextual import Offsets

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


def get_document_chunks(
    raw_documents: list[str], offsets: list[Offsets]
) -> list[str]:
    chunks = []
    for doc, _offs in zip(raw_documents, offsets):
        for start_char, end_char in _offs:
            chunks.append(raw_documents[start_char, end_char])
    return chunks


class LateModel(TransformerMixin):
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
        self,
        raw_documents: list[str],
        embeddings: list[np.ndarray] = None,
        offsets: list[Offsets] = None,
    ):
        if (embeddings is None) or (offsets is None):
            embeddings, offsets = self.model.encoder.encode_tokens(
                raw_documents, batch_size=self.batch_size
            )
        flat_embeddings, lengths = flatten_repr(embeddings)
        chunks = get_document_chunks(raw_documents, offsets)
        out_array = self.model.transform(chunks, embeddings=flat_embeddings)
        if self.pooling is None:
            return unflatten_repr(out_array, lengths)
        else:
            return pool_flat(out_array, lengths)

    def fit_transform(
        self,
        raw_documents: list[str],
        y=None,
        embeddings: list[np.ndarray] = None,
        offsets: list[Offsets] = None,
    ):
        if (embeddings is None) or (offsets is None):
            embeddings, offsets = self.model.encoder.encode_tokens(
                raw_documents, batch_size=self.batch_size
            )
        flat_embeddings, lengths = flatten_repr(embeddings)
        chunks = get_document_chunks(raw_documents, offsets)
        out_array = self.model.fit_transform(
            chunks, embeddings=flat_embeddings
        )
        if self.pooling is None:
            return unflatten_repr(out_array, lengths)
        else:
            return pool_flat(out_array, lengths)
