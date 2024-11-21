from typing import Iterable, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from turftopic.namers.base import TopicNamer


class NgramTopicNamer(TopicNamer):
    """Retrieves the most similar n-grams from a corpus using an encoder model
    to the topic descriptions, these will be assigned as topic names.

    Parameters
    ----------
    corpus: Iterable[str]
        Corpus to take n-grams from.
    encoder: str or Encoder, default 'all-MiniLM-L6-v2'
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    ngram_range: tuple[int, int], default (2,5)
        The lower and upper boundary of the range of n-values for different word n-grams to be extracted.
    max_features: Optional[int], default 8000
        Top n-grams to keep, if None, all are kept.
    vectorizer: CountVectorizer, default None
        Vectorizer used for n-gram extraction.
        Can be used to prune or filter the vocabulary.
    """

    def __init__(
        self,
        corpus: Iterable[str],
        encoder: Union[
            SentenceTransformer, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        ngram_range: tuple[int, int] = (3, 4),
        max_features: Optional[int] = 8000,
        vectorizer: Optional[CountVectorizer] = None,
    ):
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
            )
        else:
            self.vectorizer = vectorizer
        console = Console()
        with console.status("Fitting namer") as status:
            status.update("Collecting n-grams")
            self.vectorizer.fit(corpus)
            self.ngrams = self.vectorizer.get_feature_names_out()
            console.log("N-grams learned")
            status.update("Encoding n-grams")
            if self.is_encoder_promptable:
                self.ngram_embeddings = self.encoder_.encode(
                    self.ngrams, prompt_name="passage"
                )
            else:
                self.ngram_embeddings = self.encoder_.encode(self.ngrams)
            console.log("N-grams encoded")

    @property
    def is_encoder_promptable(self) -> bool:
        prompts = getattr(self.encoder_, "prompts", None)
        if prompts is None:
            return False
        if ("query" in prompts) and ("passage" in prompts):
            return True

    def name_topic(
        self,
        keywords: list[list[str]],
    ) -> str:
        query = ", ".join(keywords)
        if self.is_encoder_promptable:
            query_embedding = self.encoder_.encode(
                [query], prompt_name="query"
            )
        else:
            query_embedding = self.encoder_.encode([query])
        similarities = cosine_similarity(
            query_embedding, self.ngram_embeddings
        )
        similarities = np.ravel(similarities)
        name = self.ngrams[np.argmax(similarities)]
        return name
