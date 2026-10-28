import itertools
import warnings
from abc import abstractmethod
from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as spr
import torch
from sentence_transformers import MultiVectorEncoder, SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize

from turftopic.base import ContextualModel

Offsets = list[tuple[int, int]]
Lengths = list[int]


def sliding_window(
    token_embeddings: list[np.ndarray],
    token_offsets: Offsets,
    step_size: int,
    window_size: int,
) -> tuple[list[np.ndarray], Offsets]:
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


def to_numpy(arr: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    if torch.is_tensor(arr):
        return arr.float().numpy(force=True)
    warnings.warn(
        f"Array is neither torch or numpy but {type(arr)}. Returning without conversion. This might cause issues."
    )
    return arr


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
        **kwargs,
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
            texts,
            output_value="token_embeddings",
            **kwargs,
        )
        offsets = self.tokenizer(
            texts, return_offsets_mapping=True, verbose=False
        )["offset_mapping"]
        offsets = [
            offs[: len(embs)] for offs, embs in zip(offsets, token_embeddings)
        ]
        token_embeddings = [
            to_numpy(embs)
            for embs in token_embeddings
            if torch.is_tensor(embs)
        ]
        return token_embeddings, offsets

    def encode_tokens(
        self,
        sentences: list[str],
        **kwargs,
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
            **kwargs,
        )
        token_embeddings = [normalize(emb) for emb in token_embeddings]
        return token_embeddings, offsets

    def encode_windows(
        self,
        sentences: list[str],
        window_size: int = 50,
        step_size: int = 40,
        **kwargs,
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
            **kwargs,
        )
        return sliding_window(
            token_embeddings,
            token_offsets,
            window_size=window_size,
            step_size=step_size,
        )


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
    if len(repr) == 0:
        return np.array([])
    if isinstance(repr[0], np.ndarray):
        lengths = [r.shape[0] for r in repr]
        return np.concatenate(repr, axis=0), lengths
    if spr.issparse(repr[0]):
        lengths = [r.shape[0] for r in repr]
        return spr.vstack(repr), lengths
    else:
        lengths = [len(r) for r in repr]
        return list(itertools.chain.from_iterable(repr)), lengths


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


def pool_flat(
    flat_repr: np.ndarray | spr.sparray, lengths: Lengths, agg=np.nanmean
):
    """Pools vectors within documents using the agg function.

    Parameters
    ----------
    flat_repr: ndarray or sparse array of shape (n_total_tokens, n_dims)
        Flattened document representations.
    lengths: Lengths
        Number of tokens in each document.

    Returns
    -------
    ndarray or sparse array of shape (n_documents, n_dims)
        Pooled representation for each document.
    """
    if spr.issparse(flat_repr):
        stack = partial(spr.vstack, format="csr")
        array = spr.csr_matrix
    else:
        stack = np.stack
        array = np.asarray
    pooled = []
    start_index = 0
    for length in lengths:
        pooled.append(
            array(agg(flat_repr[start_index : start_index + length], axis=0))
        )
        start_index += length
    return stack(pooled)


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


class LateModel(ContextualModel, TransformerMixin):
    """Should have the following attributes:

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

    def load_encoder(self):
        if isinstance(self.encoder, str):
            if self.trf_kwargs is None:
                trf_kwargs = dict()
            else:
                trf_kwargs = self.trf_kwargs
            self.encoder_ = MultiVectorEncoder(self.encoder, **trf_kwargs)
            self._encoder_preloaded = False
        elif isinstance(self.encoder, SentenceTransformer):
            self.encoder_ = self.encoder
            # This is a nasty trick, and I should probably find something better in the future,
            # but this casts a sentence transformer directly into a LateSentenceTransformer
            self.encoder_.__class__ = LateSentenceTransformer
        else:
            self.encoder_ = self.encoder
            self._encoder_preloaded = True

    @staticmethod
    def flatten_repr(repr):
        return flatten_repr(repr)

    @staticmethod
    def unflatten_repr(flat_repr, lengths):
        return unflatten_repr(flat_repr, lengths)

    @staticmethod
    def pool_flat(flat_repr, lengths, agg=np.nanmean):
        return pool_flat(flat_repr, lengths, agg)

    def pool(self, doc_topic_matrix: list[np.ndarray], lengths):
        if self.pooling is None:
            return doc_topic_matrix, lengths
        else:
            return pool_flat(doc_topic_matrix, lengths, self.pooling)

    def ragged_vectorize(
        self,
        raw_documents: list[str],
        offsets: Offsets,
        train_vectorizer=False,
    ):
        if train_vectorizer:
            self.vectorizer.fit(raw_documents)
        dtm = []
        for doc, offs in zip(raw_documents, offsets):
            _dtm = self.vectorizer.transform(
                [doc[start_ind:end_ind] for start_ind, end_ind in offs]
            )
            dtm.append(_dtm)
        return dtm

    def encode_tokens(self, raw_documents: list[str]):
        encode_kwargs = dict(convert_to_numpy=True)
        if self.encode_kwargs is not None:
            encode_kwargs = self.encode_kwargs
        if isinstance(self.encoder_, MultiVectorEncoder):
            embeddings = self.encoder_.encode(raw_documents, **encode_kwargs)
            offsets = self.encoder_.tokenizer(
                raw_documents, return_offsets_mapping=True, verbose=False
            )["offset_mapping"]
            offsets = [
                offs[: len(embs)] for offs, embs in zip(offsets, embeddings)
            ]
            return embeddings, offsets
        else:
            embeddings, offsets = self.encoder_.encode_tokens(
                raw_documents, **encode_kwargs
            )
            return embeddings, offsets

    def encode_documents(self, raw_documents: list[str]):
        encode_kwargs = dict(convert_to_numpy=True)
        if self.encode_kwargs is not None:
            encode_kwargs = self.encode_kwargs
        if isinstance(self.encoder_, MultiVectorEncoder):
            embeddings, offsets = self.encode_tokens(raw_documents)
            flat_emb, lengths = flatten_repr(embeddings)
            return pool_flat(flat_emb, lengths)
        else:
            return self.encoder_.encode(raw_documents, **encode_kwargs)

    def encode_late(
        self, raw_documents: list[str]
    ) -> tuple[np.ndarray, list[Offsets]]:
        token_embeddings, token_offsets = self.encode_tokens(raw_documents)
        if self.window_size is None:
            return token_embeddings, token_offsets
        # If the window_size is specified, but not step_size, we set the step size to the window size
        # Thereby getting non-overlapping windows
        step_size = (
            self.window_size if self.step_size is None else self.step_size
        )
        return sliding_window(
            token_embeddings,
            token_offsets,
            window_size=self.window_size,
            step_size=step_size,
        )

    @abstractmethod
    def fit_transform(
        self,
        raw_documents,
        y=None,
        embeddings: Optional[list[np.ndarray]] = None,
        offsets: Optional[Offsets] = None,
    ) -> tuple[list[np.ndarray], Offsets] | np.ndarray:
        """Fits model and infers topic importances for each document.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        y: None
            Ignored, exists for sklearn compatibility.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Ragged array of document encodings.
        offsets: Offsets, optional
            Character offset for each unit in the document.

        Returns
        -------
        (list[np.ndarray], Offsets) or ndarray
            Document-topic matrix.
            If pooling = None, this should be a ragged array with offsets,
            otherwise a regular matrix.
        """
        pass

    def fit(
        self,
        raw_documents,
        y=None,
        embeddings: Optional[list[np.ndarray]] = None,
        offsets: Optional[Offsets] = None,
    ):
        """Fits model on the given corpus.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        y: None
            Ignored, exists for sklearn compatibility.
        embeddings: list[ndarray], optional
            Ragged array of document encodings.
        offsets: Offsets, optional
            Character offset for each unit in the document.
        """
        self.fit_transform(
            raw_documents, y, embeddings=embeddings, offsets=offsets
        )
        return self


class LateWrapper(LateModel):
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
        self.encode_kwargs = self.model.encode_kwargs

    def encode_late(
        self, raw_documents: list[str]
    ) -> tuple[np.ndarray, list[Offsets]]:
        if self.window_size is None:
            embeddings, offsets = self.model.encoder.encode_tokens(
                raw_documents, batch_size=self.batch_size, **self.encode_kwargs
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
            **self.encode_kwargs,
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
