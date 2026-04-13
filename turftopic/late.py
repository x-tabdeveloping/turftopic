import itertools
import warnings
from typing import Callable, Iterable, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize
from tqdm import trange

from turftopic.base import ContextualModel

Offsets = list[tuple[int, int]]
Lengths = list[int]


class LateSentenceTransformer(SentenceTransformer):
    """SentenceTransformer model that can produce token and window-level embeddings.
    Its output can be used by topic models that can use multi-vector document representations.

    !!! warning
        This is not checked yet in the library,
        but we recommend that you use SentenceTransformers that are
        a) **Mean pooled**
        b) **L2 Normalized**
        This will guarrantee that the token/window embeddings are in the same embedding space as the documents.
    """

    has_used_token_level = False

    def encode(
        self, sentences: Union[str, list[str], np.ndarray], *args, **kwargs
    ):
        if not self.has_used_token_level:
            warnings.warn(
                "Encoder is contextual but topic model is not using contextual embeddings. Perhaps you wanted to use another topic model."
            )
        return super().encode(sentences, *args, **kwargs)

    def _encode_tokens(
        self,
        texts,
        batch_size=32,
        show_progress_bar=True,
    ) -> tuple[list[np.ndarray], list[Offsets]]:
        """
        Returns
        -------
        token_embeddings: list[np.ndarray]
            Embedding matrix of tokens for each document.
        offsets: list[list[tuple[int, int]]]
            Start and end character of each token in each document.
        """
        self.has_used_token_level = True
        token_embeddings = self.encode(
            texts, output_value="token_embeddings", batch_size=batch_size
        )
        offsets = self.tokenizer(
            texts, return_offsets_mapping=True, verbose=False
        )["offset_mapping"]
        offsets = [
            offs[: len(embs)] for offs, embs in zip(offsets, token_embeddings)
        ]
        token_embeddings = [
            embs.numpy(force=True)
            for embs in token_embeddings
            if torch.is_tensor(embs)
        ]
        return token_embeddings, offsets

    def encode_tokens(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ):
        """Produces contextual token embeddings over all documents.

        Parameters
        ----------
        sentences: list[str]
            Documents to encode contextually.
        batch_size: int, default 32
            Size of the batch of document to encode at once.
        show_progress_bar: bool, default True
            Indicates whether a progress bar should be displayed when encoding.

        Returns
        -------
        token_embeddings: list[np.ndarray]
            Embedding matrix of tokens for each document.
        offsets: list[list[tuple[int, int]]]
            Start and end character of each token in each document.
        """
        # This is needed because the above implementation does not normalize embeddings,
        # which normally happens to document embeddings.
        token_embeddings, offsets = self._encode_tokens(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        token_embeddings = [normalize(emb) for emb in token_embeddings]
        return token_embeddings, offsets

    def encode_windows(
        self,
        sentences: list[str],
        batch_size: int = 32,
        window_size: int = 50,
        step_size: int = 40,
        show_progress_bar: bool = True,
    ):
        """Produces contextual embeddings for a sliding window of tokens similar to C-Top2Vec.

        Parameters
        ----------
        sentences: list[str]
            Documents to encode contextually.
        batch_size: int, default 32
            Size of the batch of document to encode at once.
        window_size: int, default 50
            Size of the sliding window.
        step_size: int, default 40
            Step size of the window.
            If step_size < window_size, windows will overlap.
            If step_size == window_size, then windows are separate.
            If step_size > window_size, there will be gaps between the windows.
            In this case, we throw a warning, as this is probably unintended behaviour.
        show_progress_bar: bool, default True
            Indicates whether a progress bar should be displayed when encoding.

        Returns
        -------
        window_embeddings: list[np.ndarray]
            Embedding matrix of windows for each document.
        offsets: list[list[tuple[int, int]]]
            Start and end character of each token in each document.
        """
        token_embeddings, token_offsets = self._encode_tokens(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        window_embeddings = []
        window_offsets = []
        for emb, offs in zip(token_embeddings, token_offsets):
            _offsets = []
            _embeddings = []
            for start_index in range(0, len(emb), step_size):
                end_index = start_index + window_size
                window_emb = np.mean(emb[start_index:end_index], axis=0)
                off = offs[start_index:end_index]
                _embeddings.append(window_emb)
                _offsets.append((off[0][0], off[-1][1]))
            window_embeddings.append(normalize(np.stack(_embeddings)))
            window_offsets.append(_offsets)
        return window_embeddings, window_offsets


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
        repr.append(flat_repr[start_index : start_index + length])
        start_index += length
    return repr


def pool_flat(flat_repr: np.ndarray, lengths: Lengths, agg=np.nanmean):
    """Pools vectors within documents using the agg function.

    Parameters
    ----------
    flat_repr: ndarray of shape (n_total_tokens, n_dims)
        Flattened document representations.
    lengths: Lengths
        Number of tokens in each document.

    Returns
    -------
    ndarray of shape (n_documents, n_dims)
        Pooled representation for each document.
    """
    pooled = []
    start_index = 0
    for length in lengths:
        pooled.append(
            agg(flat_repr[start_index : start_index + length], axis=0)
        )
        start_index += length
    return np.stack(pooled)


def get_document_chunks(
    raw_documents: list[str], offsets: list[Offsets]
) -> list[str]:
    """Extracts text chunks from documents based on token/window offsets.

    Parameters
    ----------
    raw_documents: list[str]
        Text documents.
    offsets: list[Offsets]
        Offsets returned when encoding.

    Returns
    -------
    list[str]
        Text chunks of tokens/windows in the documents.
    """
    chunks = []
    for doc, _offs in zip(raw_documents, offsets):
        for start_char, end_char in _offs:
            chunks.append(doc[start_char:end_char])
    return chunks


class LateWrapper(ContextualModel, TransformerMixin):
    """Wraps existing Turftopic model so that they can accept and create
    multi-vector document representations.

    !!! warning
        The model HAS TO HAVE a late interaction encoder model
        (e.g. `LateSentenceTransformer`)

    Parameters
    ----------
    model
        Turftopic model to turn into late-interaction model.
    batch_size: int, default 32
        Batch size of the transformer.
    window_size: int, default None
        Size of the sliding window to average tokens over.
        If None, documents will be represented at a token level.
    step_size: int, default None
        Step size of the window.
        If (step_size == None) or (step_size == window_size), then windows are separate.
        If step_size < window_size, windows will overlap.
        If step_size > window_size, there will be gaps between the windows.
        In this case, we throw a warning, as this is probably unintended behaviour.
    pooling: Callable, default None
        Indicates whether and how to pool document-topic matrices.
        If None, multi-vector topic proportions are returned in a ragged array.
        If Callable, multiple vectors are averaged with the callable in each document.
        You could for example take the mean by specifying `pooling=np.nanmean`.
    """

    def __init__(
        self,
        model: TransformerMixin,
        batch_size: Optional[int] = 32,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        pooling: Optional[Callable] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.pooling = pooling
        self.window_size = window_size
        self.step_size = step_size

    def encode_late(
        self, raw_documents: list[str]
    ) -> tuple[np.ndarray, list[Offsets]]:
        if self.window_size is None:
            embeddings, offsets = self.model.encoder.encode_tokens(
                raw_documents, batch_size=self.batch_size
            )
            return embeddings, offsets
        # If the window_size is specified, but not step_size, we set the step size to the window size
        # Thereby getting non-overlapping windows
        step_size = (
            self.window_size if self.step_size is None else self.step_size
        )
        embeddings, offsets = self.model.encoder.encode_windows(
            raw_documents,
            batch_size=self.batch_size,
            window_size=self.window_size,
            step_size=step_size,
        )
        return embeddings, offsets

    def transform(
        self,
        raw_documents: list[str],
        embeddings: list[np.ndarray] = None,
        offsets: list[Offsets] = None,
    ):
        if (embeddings is None) or (offsets is None):
            embeddings, offsets = self.encode_late(raw_documents)
        flat_embeddings, lengths = flatten_repr(embeddings)
        chunks = get_document_chunks(raw_documents, offsets)
        out_array = self.model.transform(chunks, embeddings=flat_embeddings)
        if self.pooling is None:
            return unflatten_repr(out_array, lengths), offsets
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
            embeddings, offsets = self.encode_late(raw_documents)
        flat_embeddings, lengths = flatten_repr(embeddings)
        chunks = get_document_chunks(raw_documents, offsets)
        out_array = self.model.fit_transform(
            chunks, embeddings=flat_embeddings
        )
        if self.pooling is None:
            return unflatten_repr(out_array, lengths), offsets
        else:
            return pool_flat(out_array, lengths)

    @property
    def components_(self):
        return self.model.components_

    @property
    def hierarchy(self):
        return self.model.hierarchy

    @property
    def topic_names(self):
        return self.model.topic_names

    @property
    def classes_(self):
        return self.model.classes_

    @property
    def vectorizer(self):
        return self.model.vectorizer
