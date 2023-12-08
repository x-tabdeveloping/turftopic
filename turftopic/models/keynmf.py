from typing import Dict, List, Optional, Union

import numpy as np
import scipy.sparse as spr
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import NMF
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from turftopic.base import ContextualModel


class KeyNMF(ContextualModel):
    def __init__(
        self,
        n_components: int,
        top_n: int = 25,
        encoder: Union[
            SentenceTransformer, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
    ):
        self.n_components = n_components
        self.top_n = top_n
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = CountVectorizer(min_df=10)
        else:
            self.vectorizer = vectorizer
        self.dict_vectorizer_ = DictVectorizer()
        self.nmf_ = NMF(n_components)

    def extract_keywords(
        self,
        embeddings: np.ndarray,
        document_term_matrix: spr.csr_matrix,
        vocab: np.ndarray,
    ) -> List[Dict[str, float]]:
        vocab_embeddings = self.encoder_.encode(list(vocab))
        keywords = []
        for i, embedding in enumerate(embeddings):
            terms = document_term_matrix[i, :].todense()
            embedding = embedding.reshape(1, -1)
            nonzero = terms > 0
            if not np.any(nonzero):
                keywords.append(dict())
                continue
            important_terms = np.squeeze(np.asarray(nonzero))
            word_embeddings = vocab_embeddings[important_terms]
            sim = cosine_similarity(embedding, word_embeddings)
            sim = np.ravel(sim)
            kth = min(self.top_n, len(sim) - 1)
            top = np.argpartition(-sim, kth)[:kth]
            top_words = vocab[important_terms][top]
            top_sims = sim[top]
            keywords.append(dict(zip(top_words, top_sims)))
        return keywords

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        document_term_matrix = self.vectorizer.fit_transform(raw_documents)
        keywords = self.extract_keywords(
            embeddings,
            document_term_matrix,
            self.vectorizer.get_feature_names_out(),
        )
        dtm = self.dict_vectorizer_.fit_transform(keywords)
        dtm[dtm < 0] = 0  # type: ignore
        doc_topic_matrix = self.nmf_.fit_transform(dtm)
        self.components_ = self.nmf_.components_
        return doc_topic_matrix

    def get_vocab(self) -> np.ndarray:
        return self.dict_vectorizer_.get_feature_names_out()

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        document_term_matrix = self.vectorizer.transform(raw_documents)
        vocab = self.vectorizer.get_feature_names_out()
        keywords = self.extract_keywords(
            embeddings, document_term_matrix, vocab
        )
        representations = self.dict_vectorizer_.transform(keywords)
        return self.nmf_.transform(representations)
