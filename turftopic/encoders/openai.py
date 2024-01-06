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


# The code here is heavily inspired by embetter.
class OpenAIEmbeddings(ExternalEncoder):
    """Encoder model using embeddings from OpenAI."""

    def __init__(
        self, model: str = "text-embedding-ada-002", batch_size: int = 25
    ):
        import openai

        try:
            openai.api_key = os.environ["OPENAI_KEY"]
        except KeyError as e:
            raise KeyError(
                "You have to set the OPENAI_KEY environment"
                " variable to use OpenAI embeddings."
            ) from e
        openai.organization = os.getenv("OPENAI_ORG")
        self.model = model
        self.batch_size = batch_size

    def encode(self, sentences: Iterable[str]):
        import openai

        result = []
        for b in batched(sentences, self.batch_size):
            resp = openai.Embedding.create(
                input=b, model=self.model
            )  # fmt: off
            result.extend([_["embedding"] for _ in resp["data"]])
        return np.array(result)
