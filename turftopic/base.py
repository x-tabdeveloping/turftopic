from abc import ABC
from typing import Dict

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
