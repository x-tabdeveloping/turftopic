from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ContextualModel(ABC, TransformerMixin, BaseEstimator):
    def get_topics(self, top_k: int = 10) -> Dict:
        n_topics = self.components_.shape[0]
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(n_topics))
        highest = np.argpartition(-self.components_, top_k)[:, :top_k]
        top = self.vocab_[highest]
        topics = dict()
        for topic, words in zip(classes, top):
            topics[topic] = list(words)
        return topics

    def encode_documents(self, raw_documents: Iterable[str]) -> np.ndarray:
        return self.encoder_.encode(raw_documents)

    @abstractmethod
    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pass

    def fit(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        self.fit_transform(raw_documents, y, embeddings)
        return self

    def get_vocab(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()
