import itertools
import warnings
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import scipy.sparse as spr
from sklearn.base import clone
from sklearn.decomposition._nmf import (NMF, MiniBatchNMF, _initialize_nmf,
                                        _update_coordinate_descent)
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import check_array
from sklearn.utils.validation import check_non_negative

from turftopic.base import Encoder

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)


def batched(iterable, n: int) -> Iterable[list[str]]:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def fit_timeslice(
    X,
    W,
    H,
    tol=1e-4,
    max_iter=200,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    verbose=0,
    shuffle=False,
    random_state=None,
):
    """Fits topic_term_matrix based on a precomputed document_topic_matrix.
    This is used to get temporal components in dynamic KeyNMF.
    """
    Ht = check_array(H.T, order="C")
    if random_state is None:
        rng = np.random.mtrand._rand
    else:
        rng = np.random.RandomState(random_state)
    for n_iter in range(1, max_iter + 1):
        violation = 0.0
        violation += _update_coordinate_descent(
            X.T, Ht, W, l1_reg_H, l2_reg_H, shuffle, rng
        )
        if n_iter == 1:
            violation_init = violation
        if violation_init == 0:
            break
        if verbose:
            print("violation:", violation / violation_init)
        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break
    return W, Ht.T, n_iter


class SBertKeywordExtractor:
    def __init__(
        self, top_n: int, encoder: Encoder, vectorizer: CountVectorizer
    ):
        self.top_n = top_n
        self.encoder = encoder
        self.vectorizer = vectorizer
        self.key_to_index: dict[str, int] = {}
        self.term_embeddings: Optional[np.ndarray] = None

    @property
    def is_encoder_promptable(self) -> bool:
        prompts = getattr(self.encoder, "prompts", None)
        if prompts is None:
            return False
        if ("query" in prompts) and ("passage" in prompts):
            return True

    @property
    def n_vocab(self) -> int:
        return len(self.key_to_index)

    def _add_terms(self, new_terms: list[str]):
        for term in new_terms:
            self.key_to_index[term] = self.n_vocab
        if not self.is_encoder_promptable:
            term_encodings = self.encoder.encode(new_terms)
        else:
            term_encodings = self.encoder.encode(
                new_terms, prompt_name="passage"
            )
        if self.term_embeddings is not None:
            self.term_embeddings = np.concatenate(
                (self.term_embeddings, term_encodings), axis=0
            )
        else:
            self.term_embeddings = term_encodings

    def batch_extract_keywords(
        self,
        documents: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[dict[str, float]]:
        if not len(documents):
            return []
        if embeddings is None:
            if not self.is_encoder_promptable:
                embeddings = self.encoder.encode(documents)
            else:
                embeddings = self.encoder.encode(
                    documents, prompt_name="query"
                )
        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of documents doesn't match number of embeddings."
            )
        keywords = []
        vectorizer = clone(self.vectorizer)
        document_term_matrix = vectorizer.fit_transform(documents)
        batch_vocab = vectorizer.get_feature_names_out()
        new_terms = list(set(batch_vocab) - set(self.key_to_index.keys()))
        if len(new_terms):
            self._add_terms(new_terms)
        total = embeddings.shape[0]
        for i in range(total):
            terms = document_term_matrix[i, :].todense()
            embedding = embeddings[i].reshape(1, -1)
            mask = terms > 0
            if not np.any(mask):
                keywords.append(dict())
                continue
            important_terms = np.ravel(np.asarray(mask))
            word_embeddings = [
                self.term_embeddings[self.key_to_index[term]]
                for term in batch_vocab[important_terms]
            ]
            if self.term_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    NOT_MATCHING_ERROR.format(
                        n_dims=embeddings.shape[1],
                        n_word_dims=self.term_embeddings.shape[1],
                    )
                )
            sim = cosine_similarity(embedding, word_embeddings).astype(
                np.float64
            )
            sim = np.ravel(sim)
            kth = min(self.top_n, len(sim) - 1)
            top = np.argpartition(-sim, kth)[:kth]
            top_words = batch_vocab[important_terms][top]
            top_sims = [sim for sim in sim[top] if sim > 0]
            keywords.append(dict(zip(top_words, top_sims)))
        return keywords


class KeywordNMF:
    def __init__(
        self,
        n_components: int,
        seed: Optional[int] = None,
        top_n: Optional[int] = None,
    ):
        self.n_components = n_components
        self.key_to_index: dict[str, int] = {}
        self.index_to_key: list[str] = []
        self.top_n = top_n
        # n_components * n_vocab
        self.components: Optional[np.ndarray] = None
        self.seed = seed
        self.temporal_components: Optional[np.ndarray] = None
        self.temporal_importance_: Optional[np.ndarray] = None

    def prune_keywords(self, keywords: dict[str, float]) -> dict[str, float]:
        """If there are more keywords than allowed, this prunes them."""
        if (self.top_n is None) or (self.top_n >= len(keywords)):
            return keywords
        words, similarities = zip(*keywords.items())
        similarities = np.array(similarities)
        selected = np.argsort(-similarities)[: self.top_n]
        items = [(words[i], similarities[i]) for i in selected]
        return dict(items)

    @property
    def n_vocab(self) -> int:
        return len(self.index_to_key)

    def _add_word_components(self, X: spr.csr_matrix):
        """Initializes components for novel vocabulary."""
        _, H = _initialize_nmf(X, self.n_components, random_state=self.seed)
        if self.components is None:
            self.components = H
        else:
            n_new = X.shape[1] - self.components.shape[1]
            if n_new:
                self.components = np.concatenate(
                    (self.components, H[:, -n_new:]), axis=1
                )
        if self.temporal_components is not None:
            n_new = X.shape[1] - self.temporal_components.shape[-1]
            if n_new:
                new_comps = H[:, -n_new:]
                new_comps = np.broadcast_to(
                    new_comps,
                    (self.temporal_components.shape[0], *new_comps.shape),
                )
                self.temporal_components = np.concatenate(
                    (self.temporal_components, new_comps), axis=-1
                )

    def vectorize(
        self, keywords: list[dict[str, float]], fitting: bool = False
    ) -> spr.csr_array:
        indices = []
        indptr = [0]
        values = []
        for k in keywords:
            k = self.prune_keywords(k)
            for w, v in k.items():
                # Adding vocab item if missing
                if (w not in self.key_to_index) and fitting:
                    self.key_to_index[w] = self.n_vocab
                    self.index_to_key.append(w)
                if w in self.key_to_index:
                    indices.append(self.key_to_index[w])
                    values.append(v)
            indptr.append(len(indices))
        shape = (len(indptr) - 1, self.n_vocab)
        document_term_matrix = spr.csr_matrix(
            (values, indices, indptr), shape=shape
        )
        return document_term_matrix

    def fit_transform(self, keywords: list[dict[str, float]]) -> np.ndarray:
        X = self.vectorize(keywords, fitting=True)
        check_non_negative(X, "NMF (input X)")
        W, H = _initialize_nmf(X, self.n_components, random_state=self.seed)
        W, H, self.n_iter = NMF(
            self.n_components, init="custom", random_state=self.seed
        )._fit_transform(X, W=W, H=H, update_H=True)
        self.components = H.astype(X.dtype)
        return W

    def transform(self, keywords: list[dict[str, float]]):
        if self.components is None:
            raise NotFittedError(
                "Can't transform() if the model has not been fitted."
            )
        X = self.vectorize(keywords, fitting=False)
        check_non_negative(X, "NMF (input X)")
        W, _, _ = NMF(
            self.n_components, init="custom", random_state=self.seed
        )._fit_transform(X, W=None, H=self.components, update_H=False)
        return W.astype(X.dtype)

    def partial_fit(self, keyword_batch: list[dict[str, float]]):
        X = self.vectorize(keyword_batch, fitting=True)
        try:
            check_non_negative(X, "NMF (input X)")
            self._add_word_components(X)
            W, _ = _initialize_nmf(
                X, self.n_components, random_state=self.seed
            )
            _minibatchnmf = MiniBatchNMF(
                self.n_components, init="custom", random_state=self.seed
            ).partial_fit(X, W=W, H=self.components)
            self.components = _minibatchnmf.components_.astype(X.dtype)
        except ValueError as e:
            warnings.warn(f"Batch failed with error: {e}, skipping.")
            return self
        return self

    def fit_transform_dynamic(
        self,
        keywords: list[dict[str, float]],
        time_labels: np.ndarray,
        time_bin_edges: list[datetime],
    ) -> np.ndarray:
        self.time_bin_edges = time_bin_edges
        n_bins = len(time_bin_edges) - 1
        document_term_matrix = self.vectorize(keywords, fitting=True)
        check_non_negative(document_term_matrix, "NMF (input X)")
        document_topic_matrix, H = _initialize_nmf(
            document_term_matrix,
            self.n_components,
            random_state=self.seed,
        )
        document_topic_matrix, H, self.n_iter = NMF(
            self.n_components, init="custom", random_state=self.seed
        )._fit_transform(
            document_term_matrix, W=document_topic_matrix, H=H, update_H=True
        )
        self.components = H.astype(document_term_matrix.dtype)
        n_comp, n_vocab = self.components.shape
        self.temporal_components = np.zeros(
            (n_bins, n_comp, n_vocab), dtype=document_term_matrix.dtype
        )
        self.temporal_importance_ = np.zeros((n_bins, n_comp))
        for label in np.unique(time_labels):
            idx = np.nonzero(time_labels == label)
            X = document_term_matrix[idx]
            W = document_topic_matrix[idx]
            _, H = _initialize_nmf(
                X, self.components.shape[0], random_state=self.seed
            )
            _, H, _ = fit_timeslice(X, W, H, random_state=self.seed)
            self.temporal_components[label] = H
            topic_importances = np.squeeze(np.asarray(W.sum(axis=0)))
            self.temporal_importance_[label] = topic_importances
        return document_topic_matrix

    def partial_fit_dynamic(
        self,
        keyword_batch: list[dict[str, float]],
        time_labels: np.ndarray,
        time_bin_edges: list[datetime],
    ) -> np.ndarray:
        if self.temporal_components is None:
            self.fit_transform_dynamic(
                keyword_batch, time_labels, time_bin_edges
            )
        else:
            document_term_matrix = self.vectorize(keyword_batch, fitting=True)
            check_non_negative(document_term_matrix, "NMF (input X)")
            self._add_word_components(document_term_matrix)
            document_topic_matrix = self.transform(keyword_batch)
            _minibatchnmf = MiniBatchNMF(
                self.n_components, init="custom", random_state=self.seed
            ).partial_fit(
                document_term_matrix,
                W=document_topic_matrix,
                H=self.components,
            )
            self.components = _minibatchnmf.components_.astype(
                document_term_matrix.dtype
            )
            document_topic_matrix = self.transform(keyword_batch)
            for label in np.unique(time_labels):
                idx = np.nonzero(time_labels == label)
                X = document_term_matrix[idx]
                W = document_topic_matrix[idx]
                _minibatchnmf = MiniBatchNMF(
                    self.n_components, init="custom", random_state=self.seed
                ).partial_fit(
                    X,
                    W=W,
                    H=self.temporal_components[label],
                )
                self.temporal_components[label] = _minibatchnmf.components_
                topic_importances = np.squeeze(np.asarray(W.sum(axis=0)))
                self.temporal_importance_[label] += topic_importances
