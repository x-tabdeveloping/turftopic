from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.decomposition import FastICA
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel, Encoder
from turftopic.vectorizer import default_vectorizer


class SemanticSignalSeparation(ContextualModel):
    """Separates the embedding matrix into 'semantic signals' with
    component analysis methods.
    Topics are assumed to be dimensions of semantics.

    ```python
    from turftopic import SemanticSignalSeparation

    corpus: list[str] = ["some text", "more text", ...]

    model = SemanticSignalSeparation(10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int, default 10
        Number of topics.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    decomposition: TransformerMixin, default None
        Custom decomposition method to use.
        Can be an instance of FastICA or PCA, or basically any dimensionality
        reduction method. Has to have `fit_transform` and `fit` methods.
        If not specified, FastICA is used.
    max_iter: int, default 200
        Maximum number of iterations for ICA.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    """

    def __init__(
        self,
        n_components: int = 10,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        decomposition: Optional[TransformerMixin] = None,
        max_iter: int = 200,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.max_iter = max_iter
        self.random_state = random_state
        if decomposition is None:
            self.decomposition = FastICA(
                n_components, max_iter=max_iter, random_state=random_state
            )
        else:
            self.decomposition = decomposition

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        console = Console()
        with console.status("Fitting model") as status:
            if embeddings is None:
                status.update("Encoding documents")
                embeddings = self.encoder_.encode(raw_documents)
                console.log("Documents encoded.")
            status.update("Decomposing embeddings")
            doc_topic = self.decomposition.fit_transform(embeddings)
            console.log("Decomposition done.")
            status.update("Extracting terms.")
            vocab = self.vectorizer.fit(raw_documents).get_feature_names_out()
            console.log("Term extraction done.")
            status.update("Encoding vocabulary")
            vocab_embeddings = self.encoder_.encode(vocab)
            console.log("Vocabulary encoded.")
            status.update("Estimating term importances")
            vocab_topic = self.decomposition.transform(vocab_embeddings)
            self.components_ = vocab_topic.T
            console.log("Model fitting done.")
        return doc_topic

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
            embeddings = self.encoder_.encode(raw_documents)
        return self.decomposition.transform(embeddings)

    def print_topics(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        show_negative: bool = True,
    ):
        super().print_topics(top_k, show_scores, show_negative)

    def export_topics(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        show_negative: bool = True,
        format: str = "csv",
    ) -> str:
        return super().export_topics(top_k, show_scores, show_negative, format)

    def print_representative_documents(
        self,
        topic_id,
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: bool = True,
    ):
        super().print_representative_documents(
            topic_id,
            raw_documents,
            document_topic_matrix,
            top_k,
            show_negative,
        )

    def export_representative_documents(
        self,
        topic_id,
        raw_documents,
        document_topic_matrix=None,
        top_k=5,
        show_negative: bool = True,
        format: str = "csv",
    ):
        return super().export_representative_documents(
            topic_id,
            raw_documents,
            document_topic_matrix,
            top_k,
            show_negative,
            format,
        )
