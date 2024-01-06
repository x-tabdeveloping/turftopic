import itertools
import os
from typing import Iterable, List

import numpy as np

from turftopic.encoders.base import ExternalEncoder


def batched(iterable, n: int) -> Iterable[List[str]]:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


class CohereEmbeddings(ExternalEncoder):
    """Encoder model using embeddings from Cohere."""

    def __init__(self, model: str = "large", batch_size: int = 25):
        import cohere

        try:
            self.client = cohere.Client(os.environ["COHERE_KEY"])
        except KeyError as e:
            raise KeyError(
                "You have to set the COHERE_KEY environment"
                " variable to use Cohere embeddings."
            ) from e
        self.model = model
        self.batch_size = batch_size

    def encode(self, sentences: Iterable[str]):
        result = []
        for b in batched(sentences, self.batch_size):
            response = self.client.embed(b)
            result.extend(response.embeddings)
        return np.array(result)
