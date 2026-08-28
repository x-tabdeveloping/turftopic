import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


def _merge_countvectorizers(left_vectorizer, right_vectorizer):
    params = left_vectorizer.get_params()
    left_terms = left_vectorizer.get_feature_names_out()
    right_terms = right_vectorizer.get_feature_names_out()
    # Extracting terms present in right but not in left
    diff_terms = set(right_terms) - set(left_terms)
    joint_terms = list(left_terms) + list(diff_terms)
    # Mapping from vocabulary items to indices
    joint_vocab = dict(zip(joint_terms, range(len(joint_terms))))
    params["vocabulary"] = joint_vocab
    return type(left_vectorizer)(**params)


class VocabStore(BaseEstimator, TransformerMixin):
    """Dummy, that can't actually vectorize, but stores the terms."""

    def __init__(self, terms):
        self.terms = terms

    def get_feature_names_out(self):
        return np.array(self.terms)

    @property
    def vocabulary_(self):
        return dict(zip(self.terms, range(len(self.terms))))

    def fit_transform(self, raw_documents, y=None):
        raise NotImplementedError("Vocab store can't actually vectorize text.")

    def transform(self, raw_documents):
        raise NotImplementedError("Vocab store can't actually vectorize text.")

    @classmethod
    def from_merge(cls, left_vectorizer, right_vectorizer):
        left_terms = left_vectorizer.get_feature_names_out()
        right_terms = right_vectorizer.get_feature_names_out()
        # Extracting terms present in right but not in left
        diff_terms = set(right_terms) - set(left_terms)
        joint_terms = list(left_terms) + list(diff_terms)
        return cls(joint_terms)


def merge_vectorizers(
    left_vectorizer, right_vectorizer
) -> CountVectorizer | VocabStore:
    """Merges two vectorizers into one new vectorizer.

    Parameters
    ----------
    left_vectorizer
        Left vectorizer object to merge the other into.
    right_vectorizer
        Right vectorizer object to merge into the left vectorizer.

    Returns
    -------
    CountVectorizer or VocabStore
        If both vectorizers are CountVectorizer, a new CountVectorizer is returned,
        otherwise a dummy VocabStore object is returned.
    """
    if isinstance(left_vectorizer, CountVectorizer) and isinstance(
        right_vectorizer, CountVectorizer
    ):
        return _merge_countvectorizers(left_vectorizer, right_vectorizer)
    else:
        warnings.warn(
            "At least one vectorizer is not a CountVectorizer, returning a VocabStore object."
        )
        return VocabStore.from_merge(left_vectorizer, right_vectorizer)
