from functools import partial
from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import Encoder
from turftopic.feature_importance import (
    combined_lexical_semantic,
    ctf_idf,
    fighting_words,
    npmi,
    soft_ctf_idf,
)
from turftopic.late import (
    LateModel,
    Offsets,
)
from turftopic.models.vslda._trainer import VMFLDATrainer
from turftopic.vectorizers.default import default_vectorizer

FEATURE_IMPORTANCE_METHODS = {
    "soft-c-tf-idf": soft_ctf_idf,
    "c-tf-idf": ctf_idf,
    "fighting-words": fighting_words,
    "npmi": partial(npmi, smoothing=2),
    "lexical-semantic": partial(combined_lexical_semantic, smoothing=2),
}
FeatureImportance = Literal[
    "soft-c-tf-idf",
    "c-tf-idf",
    "npmi",
    "fighting-words",
    "lexical-semantic",
]


class vSLDA(LateModel):
    """von-Mises distributed sentence LDA.

    ```python
    from turftopic import vSLDA
    corpus: list[str] = ["some text", "more text", ...]

    model = vSLDA(10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int
        Number of topics.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    n_iter: int, default 25
        Number of iterations to run with the EM algorithm.
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
    feature_importance: FeatureImportance, default 'npmi'
        Feature importance method to use.
    trf_kwargs: dict, default None
        Keyword arguments to apply when loading the Encoder model.
    encode_kwargs: dict, default None
        Keyword arguments to apply encoding documents with the encoder.

    """

    def __init__(
        self,
        n_components: int,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
        n_iter: int = 25,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        pooling=None,
        feature_importance: FeatureImportance = "npmi",
        trf_kwargs=None,
        encode_kwargs=None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.encode_kwargs = encode_kwargs
        self.trf_kwargs = trf_kwargs
        self.random_state = random_state
        self.window_size = window_size
        self.n_iter = n_iter
        self.step_size = step_size
        self.pooling = pooling
        self.load_encoder()
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        if feature_importance not in FEATURE_IMPORTANCE_METHODS:
            valid = list(FEATURE_IMPORTANCE_METHODS.keys())
            raise ValueError(
                f"{feature_importance} not in list of valid feature importance methods: {valid}"
            )
        self.feature_importance = feature_importance

    @property
    def topic_embeddings(self):
        return self.trainer_.topic_means

    def estimate_components(
        self,
        feature_importance: Optional[FeatureImportance] = None,
        doc_topic_matrix=None,
        doc_term_matrix=None,
    ) -> np.ndarray:
        feature_importance = feature_importance or self.feature_importance
        imp_fn = FEATURE_IMPORTANCE_METHODS[feature_importance]
        doc_topic_matrix = (
            doc_topic_matrix
            if doc_topic_matrix is not None
            else self.doc_topic_matrix
        )
        doc_term_matrix = (
            doc_term_matrix
            if doc_term_matrix is not None
            else self.doc_term_matrix
        )
        if feature_importance != "lexical-semantic":
            self.components_ = imp_fn(doc_topic_matrix, doc_term_matrix)
            return self.components_
        vocab = self.get_vocab()
        if getattr(self, "vocab_embeddings", None) is None or (
            self.vocab_embeddings.shape[0] != vocab.shape[0]
        ):
            self.vocab_embeddings = self.encode_vocabulary_items(vocab)
        self.components_ = imp_fn(
            doc_topic_matrix,
            doc_term_matrix,
            self.topic_embeddings,
            self.vocab_embeddings,
        )
        return self.components_

    def fit_transform(
        self,
        raw_documents,
        y=None,
        embeddings: Optional[list[np.ndarray]] = None,
        offsets: Optional[Offsets] = None,
    ) -> np.ndarray:
        console = Console()
        with console.status("Fitting model") as status:
            if (embeddings is None) or (offsets is None):
                status.update("Encoding documents")
                embeddings, offsets = self.encode_late(raw_documents)
                console.log("Documents encoded.")
            self.embeddings = embeddings
            status.update("Extracting terms.")
            ragged_dtm = self.ragged_vectorize(
                raw_documents, offsets=offsets, train_vectorizer=True
            )
            console.log("Term extraction done.")
            status.update("Fitting model using EM.")
            self.trainer_ = VMFLDATrainer(embeddings, self.n_components)
            self.trainer_.sample(embeddings, num_iterations=self.n_iter)
            console.log("Model fitted.")
            status.update("Estimating term importances.")
            flat_dtm, _ = self.flatten_repr(ragged_dtm)
            flat_topics, _ = self.flatten_repr(self.doc_topic_matrix)
            self.estimate_components(
                doc_topic_matrix=flat_topics, doc_term_matrix=flat_dtm
            )
            console.log("Model fitting done.")
        return self.transform(
            raw_documents, embeddings=embeddings, offsets=offsets
        )

    def transform(
        self,
        raw_documents,
        embeddings: Optional[np.ndarray] = None,
        offsets: Optional[Offsets] = None,
    ) -> np.ndarray:
        if embeddings is None:
            embeddings, offsets = self.encode_late(raw_documents)
        doc_topic_matrix = self.trainer_.infer_topic_proportions(embeddings)
        return self.pool(doc_topic_matrix, offsets)
