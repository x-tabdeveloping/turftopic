import warnings

import numpy as np
import scipy.sparse as spr
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


class UnionVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, left: TransformerMixin, right: TransformerMixin):
        self.left = left
        self.right = right
        if (getattr(self.left, "vocabulary_", None) is not None) and (
            (getattr(self.right, "vocabulary_", None) is not None)
        ):
            self.merge_vocab(self.left, self.right)

    def merge_vocab(self, left: TransformerMixin, right: TransformerMixin):
        left_terms = left.get_feature_names_out()
        right_terms = right.get_feature_names_out()
        # Extracting terms present in right but not in left
        self.diff_terms_ = list(set(right_terms) - set(left_terms))
        self.joint_terms_ = list(left_terms) + self.diff_terms_
        left_terms_to_idx = {term: i for i, term in enumerate(left_terms)}
        right_terms_to_idx = {term: i for i, term in enumerate(right_terms)}
        self.joint_to_left_ = [
            left_terms_to_idx.get(term, None) for term in self.joint_terms_
        ]
        self.joint_to_right_ = [
            right_terms_to_idx.get(term, None) for term in self.joint_terms_
        ]
        joint_to_idx = self.vocabulary_
        self.left_to_joint_ = {
            left_idx: joint_to_idx[left_term]
            for left_term, left_idx in left_terms_to_idx.items()
        }
        self.right_to_joint_ = {
            right_idx: joint_to_idx[right_term]
            for right_term, right_idx in right_terms_to_idx.items()
        }
        self.diff_indices_ = np.array(
            [right_terms_to_idx[term] for term in self.diff_terms_]
        )

    def fit_transform(self, raw_documents, y=None):
        left_X = self.left.fit_transform(raw_documents)
        right_X = self.right.fit_transform(raw_documents)
        self.merge_vocab(self.left, self.right)
        hstack = spr.hstack if spr.issparse(left_X) else np.hstack
        return hstack([left_X, right_X[:, self.diff_indices_]])

    def transform(self, raw_documents, y=None):
        left_X = self.left.transform(raw_documents)
        right_X = self.right.transform(raw_documents)
        hstack = spr.hstack if spr.issparse(left_X) else np.hstack
        return hstack([left_X, right_X[:, self.diff_indices_]])

    def pad_left(self, left_X, pad_value=0):
        if spr.issparse(left_X):
            joint_X = spr.lil_array(
                (left_X.shape[0], self.n_vocab), dtype=left_X.dtype
            )
        else:
            joint_X = np.full((left_X.shape[0], self.n_vocab), pad_value)
        joint_X[:, : left_X.shape[1]] = left_X
        if spr.issparse(joint_X):
            joint_X = spr.csr_array(joint_X)
        return joint_X

    def pad_right(self, right_X, pad_value=0):
        if spr.issparse(right_X):
            joint_X = spr.lil_array(
                (right_X.shape[0], self.n_vocab), dtype=right_X.dtype
            )
        else:
            joint_X = np.full(
                (right_X.shape[0], self.n_vocab),
                pad_value,
                dtype=right_X.dtype,
            )
        for joint_idx, right_idx in enumerate(self.joint_to_right_):
            if right_idx is not None:
                joint_X[:, joint_idx] = right_X[:, right_idx]
        if spr.issparse(joint_X):
            joint_X = spr.csr_array(joint_X)
        return joint_X

    def get_feature_names_out(self):
        return np.array(self.joint_terms_)

    @property
    def n_vocab(self):
        return len(self.joint_terms_)

    @property
    def vocabulary_(self):
        return dict(zip(self.joint_terms_, range(len(self.joint_terms_))))
