from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, FastICA
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

    model = SemanticSignalSeparation(10, objective="independence").fit(corpus)
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
    objective: 'orthogonality' or 'independence', default 'independence'
        Indicates what the components should be optimized for.
        When 'orthogonality', PCA is used to discover components,
        when 'independence', ICA is used to discover components.
    """

    def __init__(
        self,
        n_components: int,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        objective: Literal["orthogonality", "independence"] = "independence",
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
        self.objective = objective
        if objective == "independence":
            self.decomposition = FastICA(n_components)
        else:
            self.decomposition = PCA(n_components)

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
