from typing import Literal, Optional, Union

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

    def fit_transform(self, raw_documents, y=None):
        embeddings = self.encoder_.encode(raw_documents)
        doc_topic = self.decomposition.fit_transform(embeddings)
        self.vectorizer.fit(raw_documents)
        self.vocab_ = self.vectorizer.get_feature_names_out()
        vocab_topic = self.decomposition.transform(self.vocab_)
        self.components_ = vocab_topic.T
        return doc_topic

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def transform(self, raw_documents):
        embeddings = self.encoder_.encode(raw_documents)
        return self.decomposition.transform(embeddings)
