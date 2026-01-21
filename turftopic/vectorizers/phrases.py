import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class PhraseVectorizer(BaseEstimator, TransformerMixin):
    """NPMI-score-based phrase extraction."""

    def __init__(
        self,
        max_ngram=3,
        min_df=10,
        max_df=1.0,
        threshold=0.5,
        stop_words="english",
        smoothing=5,
    ):
        self.stop_words = stop_words
        self.threshold = threshold
        self.max_ngram = max_ngram
        self.min_df = min_df
        self.max_df = max_df
        self.smoothing = smoothing
        self.ngram_range = (1, max_ngram)

    def fit_transform(self, raw_documents, y=None):
        self.vectorizer_ = CountVectorizer(
            stop_words=self.stop_words,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
        )
        dtm = self.vectorizer_.fit_transform(raw_documents)
        all_vocab = self.vectorizer_.get_feature_names_out()
        token_count = dict(
            zip(
                self.vectorizer_.get_feature_names_out(),
                np.ravel(dtm.sum(axis=1)),
            )
        )
        counts = np.ravel(dtm.sum(axis=1))
        word_indices = [
            i
            for word, i in self.vectorizer_.vocabulary_.items()
            if len(word.split()) == 1
        ]
        n_ws = dtm[:, word_indices].sum() + len(word_indices) * self.smoothing
        ngram_indices = []
        for i, (token, n_w1w2) in enumerate(zip(all_vocab, counts)):
            _words = token.split()
            if len(_words) == 1:
                continue
            w1, w2 = _words[0], _words[-1]
            n_w1 = token_count.get(w1, None)
            n_w2 = token_count.get(w2, None)
            if (n_w1 is None) or (n_w2 is None):
                continue
            p_w1w2 = (n_w1w2 + self.smoothing) / n_ws
            p_w1 = (n_w1 + self.smoothing) / n_ws
            p_w2 = (n_w2 + self.smoothing) / n_ws
            pmi = np.log2(p_w1w2 / (p_w1 * p_w2))
            npmi = pmi / (-np.log2(p_w1w2))
            if npmi > self.threshold:
                ngram_indices.append(i)
        self.indices_ = np.array(word_indices + ngram_indices)
        self.feature_names_out_ = all_vocab[self.indices_]
        self.vocabulary_ = dict(
            zip(self.feature_names_out_, range(len(self.feature_names_out_)))
        )
        dtm = dtm[:, self.indices_]
        return dtm

    def transform(self, raw_documents):
        dtm = self.vectorizer_.transform(raw_documents)
        return dtm[:, self.indices_]

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def get_feature_names_out(self):
        return self.feature_names_out_
