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
    def encode(
        self, sentences: Union[str, list[str], np.ndarray], *args, **kwargs
    ):
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
        token_embeddings = []
        offsets = []
        for start_index in trange(
            0,
            len(texts),
            batch_size,
            desc="Encoding batches...",
        ):
            batch = texts[start_index : start_index + batch_size]
            features = self.tokenize(batch)
            with torch.no_grad():
                output_features = self.forward(features)
            n_tokens = output_features["attention_mask"].sum(axis=1)
            # Find first nonzero elements in each document
            # The document could be padded from the left, so we have to watch out for this.
            start_token = torch.argmax(
                (output_features["attention_mask"] > 0).to(torch.long), axis=1
            )
            end_token = start_token + n_tokens
            for i_doc in range(len(batch)):
                _token_embeddings = (
                    output_features["token_embeddings"][
                        i_doc, start_token[i_doc] : end_token[i_doc], :
                    ]
                    .float()
                    .numpy(force=True)
                )
                _n = _token_embeddings.shape[0]
                # We extract the character offsets and prune it at the maximum context length
                _offsets = self.tokenizer(
                    batch[i_doc], return_offsets_mapping=True, verbose=False
                )["offset_mapping"][:_n]
                token_embeddings.append(_token_embeddings)
                offsets.append(_offsets)
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
            chunks.append(doc[start_char:end_char])
    return chunks


class LateWrapper(ContextualModel, TransformerMixin):
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
            embeddings, offsets = self.encode_late(raw_documents)
        flat_embeddings, lengths = flatten_repr(embeddings)
        chunks = get_document_chunks(raw_documents, offsets)
        out_array = self.model.fit_transform(
            chunks, embeddings=flat_embeddings
        )
        if self.pooling is None:
            return unflatten_repr(out_array, lengths)
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
