import itertools
from typing import Iterable

from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import Encoder


def batched(iterable, n: int) -> Iterable[list[str]]:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


class AttentionKeywordExtractor:
    def __init__(
        self, top_n: int, encoder: Encoder, vectorizer: CountVectorizer
    ):
        self.top_n = top_n
        self.encoder = encoder
        self.vectorizer = vectorizer
        self.key_to_index: dict[str, int] = {}
        self.term_embeddings: Optional[np.ndarray] = None

    @property
    def is_encoder_promptable(self) -> bool:
        prompts = getattr(self.encoder, "prompts", None)
        if prompts is None:
            return False
        if ("query" in prompts) and ("passage" in prompts):
            return True

    @property
    def n_vocab(self) -> int:
        return len(self.key_to_index)

    def _add_terms(self, new_terms: list[str]):
        for term in new_terms:
            self.key_to_index[term] = self.n_vocab
        if not self.is_encoder_promptable:
            term_encodings = self.encoder.encode(new_terms)
        else:
            term_encodings = self.encoder.encode(
                new_terms, prompt_name="passage"
            )
        if self.term_embeddings is not None:
            self.term_embeddings = np.concatenate(
                (self.term_embeddings, term_encodings), axis=0
            )
        else:
            self.term_embeddings = term_encodings

    def batch_extract_keywords(
        self,
        documents: list[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> list[dict[str, float]]:
        if not len(documents):
            return []
        if embeddings is None:
            if not self.is_encoder_promptable:
                embeddings = self.encoder.encode(documents)
            else:
                embeddings = self.encoder.encode(
                    documents, prompt_name="query"
                )
        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of documents doesn't match number of embeddings."
            )
        keywords = []
        vectorizer = clone(self.vectorizer)
        document_term_matrix = vectorizer.fit_transform(documents)
        batch_vocab = vectorizer.get_feature_names_out()
        new_terms = list(set(batch_vocab) - set(self.key_to_index.keys()))
        if len(new_terms):
            self._add_terms(new_terms)
        total = embeddings.shape[0]
        for i in range(total):
            terms = document_term_matrix[i, :].todense()
            embedding = embeddings[i].reshape(1, -1)
            mask = terms > 0
            if not np.any(mask):
                keywords.append(dict())
                continue
            important_terms = np.ravel(np.asarray(mask))
            word_embeddings = [
                self.term_embeddings[self.key_to_index[term]]
                for term in batch_vocab[important_terms]
            ]
            if self.term_embeddings.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    NOT_MATCHING_ERROR.format(
                        n_dims=embeddings.shape[1],
                        n_word_dims=self.term_embeddings.shape[1],
                    )
                )
            sim = cosine_similarity(embedding, word_embeddings).astype(
                np.float64
            )
            sim = np.ravel(sim)
            kth = min(self.top_n, len(sim) - 1)
            top = np.argpartition(-sim, kth)[:kth]
            top_words = batch_vocab[important_terms][top]
            top_sims = [sim for sim in sim[top] if sim > 0]
            keywords.append(dict(zip(top_words, top_sims)))
        return keywords
