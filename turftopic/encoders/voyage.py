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


class VoyageEmbeddings(ExternalEncoder):
    """Encoder model using embeddings from VoyageAI."""

    def __init__(self, model: str = "voyage-01", batch_size: int = 25):
        import voyageai

        try:
            voyageai.api_key = os.environ["VOYAGE_KEY"]
        except KeyError as e:
            raise KeyError(
                "You have to set the VOYAGE_KEY environment"
                " variable to use Voyage embeddings."
            ) from e
        self.model = model
        self.batch_size = batch_size

    def encode(self, sentences: Iterable[str]):
        from voyageai import get_embeddings

        result = []
        for b in batched(sentences, self.batch_size):
            response = get_embeddings(b, self.model, input_type="document")
            result.extend(response)
        return np.array(result)
