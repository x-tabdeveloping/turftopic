from typing import Optional

import numpy as np
from rich.console import Console
from sklearn.decomposition import NMF as _NMF
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel
from turftopic.vectorizers.default import default_vectorizer


class NMF(ContextualModel):
    """Turftopic wrapper around sklearns's implementation of Non-negative Matrix Factorization.
    Note that NMF is a Bag-of-words model and NOT a contextual model and uses static word representations!
    The encoder and passed embeddings will be ignored.

    ```python
    from turftopic import NMF
    corpus: list[str] = ["some text", "more text", ...]

    model = NMF(10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int
        Number of topics.
    encoder: None
        Ignored.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    trf_kwargs: dict, default None
        Ignored.
    encode_kwargs: dict, default None
        Ignored.

    """

    def __init__(
        self,
        n_components: int,
        encoder=None,
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
        trf_kwargs=None,
        encode_kwargs=None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.encode_kwargs = encode_kwargs
        self.trf_kwargs = trf_kwargs
        self.random_state = random_state
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer

    @property
    def components_(self):
        return self.model_.components_

    def fit_transform(
        self,
        raw_documents,
        y=None,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        console = Console()
        with console.status("Fitting model") as status:
            status.update("Extracting terms.")
            self.doc_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            status.update("Fitting the model")
            self.model_ = _NMF(
                self.n_components, random_state=self.random_state
            )
            self.doc_topic_matrix = self.model_.fit_transform(
                self.doc_term_matrix
            )
            console.log("Model fitting done.")
            self.top_documents = self.get_top_documents(
                raw_documents, document_topic_matrix=self.doc_topic_matrix
            )
        return self.doc_topic_matrix

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Infers topic importances for new documents based on a fitted model.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: None
            Ignored.

        Returns
        -------
        ndarray of shape (n_dimensions, n_topics)
            Document-topic matrix.
        """
        doc_term_matrix = self.vectorizer.transform(raw_documents)
        return self.model_.transform(doc_term_matrix)
