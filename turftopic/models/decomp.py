from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel


class ComponentTopicModel(ContextualModel):
    def __init__(
        self,
        n_components: int,
        encoder: Union[
            SentenceTransformer, str
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
            self.vectorizer = CountVectorizer(min_df=10)
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
        embeddings = self.encoder_.encode(raw_documents)
        return self.decomposition.transform(embeddings)
