import warnings
from datetime import datetime
from functools import partial
from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel, Encoder
from turftopic.feature_importance import (
    npmi,
)
from turftopic.late import (
    LateSentenceTransformer,
    Offsets,
    flatten_repr,
    unflatten_repr,
)
from turftopic.models.vslda._trainer import VMFLDATrainer
from turftopic.vectorizers.default import default_vectorizer


class vSLDA(ContextualModel):
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
    trf_kwargs: dict, default None
        Keyword arguments to apply when loading the Encoder model.
    encode_kwargs: dict, default None
        Keyword arguments to apply encoding documents with the encoder.

    Attributes
    ----------
    """

    def __init__(
        self,
        n_components: Union[int, Literal["auto"]],
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
        n_iter: int = 25,
        window_size: Optional[int] = None,
        step_size: Optional[int] = None,
        batch_size: int = 32,
        trf_kwargs=None,
        encode_kwargs=None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.encode_kwargs = encode_kwargs
        self.trf_kwargs = trf_kwargs
        self.random_state = random_state
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.step_size = step_size
        self.load_encoder()
        self.encoder_.__class__ = LateSentenceTransformer
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer

    def encode_late(
        self, raw_documents: list[str]
    ) -> tuple[np.ndarray, list[Offsets]]:
        if self.window_size is None:
            embeddings, offsets = self.encoder_.encode_tokens(
                raw_documents, batch_size=self.batch_size
            )
            return embeddings, offsets
        # If the window_size is specified, but not step_size, we set the step size to the window size
        # Thereby getting non-overlapping windows
        step_size = (
            self.window_size if self.step_size is None else self.step_size
        )
        embeddings, offsets = self.encoder_.encode_windows(
            raw_documents,
            batch_size=self.batch_size,
            window_size=self.window_size,
            step_size=step_size,
        )
        return embeddings, offsets

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
            self.doc_topic_matrix = self.trainer_.infer_topic_proportions(
                embeddings
            )
            console.log("Model fitted.")
            status.update("Estimating term importances.")
            flat_dtm, _ = flatten_repr(ragged_dtm)
            flat_topics, _ = flatten_repr(self.doc_topic_matrix)
            self.components_ = npmi(flat_topics, flat_dtm)
            console.log("Model fitting done.")
        return self.doc_topic_matrix, offsets

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Infers topic importances for new documents based on a fitted model.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.

        Returns
        -------
        ndarray of shape (n_dimensions, n_topics)
            Document-topic matrix.
        """
        if embeddings is None:
            embeddings, offsets = self.encode_late(raw_documents)
        doc_topic_matrix = self.trainer_.infer_topic_proportions(embeddings)
        return doc_topic_matrix, offsets
