from typing import Dict, List, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from turftopic.base import ContextualModel


def extract_keywords(
    corpus, top_n: int, trf: SentenceTransformer
) -> List[Dict[str, float]]:
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out()
    vocab_embeddings = trf.encode(list(vocab))
    keywords = []
    corpus_embeddings = trf.encode(corpus)
    for i, embedding in enumerate(corpus_embeddings):
        terms = dtm[i, :].todense()
        embedding = embedding.reshape(1, -1)
        nonzero = terms > 0
        if not np.any(nonzero):
            keywords.append(dict())
            continue
        important_terms = np.squeeze(np.asarray(nonzero))
        word_embeddings = vocab_embeddings[important_terms]
        sim = cosine_similarity(embedding, word_embeddings)
        sim = np.ravel(sim)
        kth = min(top_n, len(sim) - 1)
        top = np.argpartition(-sim, kth)[:kth]
        top_words = vocab[important_terms][top]
        top_sims = sim[top]
        keywords.append(dict(zip(top_words, top_sims)))
    return keywords


class KeyNMF(ContextualModel):
    def __init__(
        self,
        n_components: int,
        top_n: int = 25,
        encoder: Union[
            SentenceTransformer, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.n_components = n_components
        self.top_n = top_n
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        self.vectorizer_ = DictVectorizer()
        self.nmf_ = NMF(n_components)

    def fit_transform(self, raw_documents, y=None):
        keywords = extract_keywords(raw_documents, self.top_n, self.encoder_)
        dtm = self.vectorizer_.fit_transform(keywords)
        dtm[dtm < 0] = 0  # type: ignore
        doc_topic_matrix = self.nmf_.fit_transform(dtm)
        self.components_ = self.nmf_.components_
        self.vocab_ = self.vectorizer_.get_feature_names_out()
        return doc_topic_matrix

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def transform(self, raw_documents):
        keywords = extract_keywords(raw_documents, self.top_n, self.encoder_)
        dtm = self.vectorizer_.fit_transform(keywords)
        dtm[dtm < 0] = 0  # type: ignore
        return self.vectorizer_.transform(dtm)
