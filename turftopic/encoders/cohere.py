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
    """Encoder model using embeddings from Cohere.

    The available models are:

     - `embed-english-v3.0`
     - `embed-multilingual-v3.0`
     - `embed-english-light-v3.0`
     - `embed-multilingual-light-v3.0`
     - `embed-english-v2.0`
     - `embed-english-light-v2.0`
     - `embed-multilingual-v2.0`

    ```python
    from turftopic.encoders import CohereEmbeddings
    from turftopic import GMM

    model = GMM(10, encoder=CohereEmbeddings())
    ```

    Parameters
    ----------
    model: str, default "embed-english-v3.0"
        Embedding model to use from Cohere.

    input_type: str, default "clustering"
        Input type passed to the embedding model.

    batch_size: int, default 25
        Sizes of the batches that will be sent to Cohere's API.
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        input_type: str = "clustering",
        batch_size: int = 25,
    ):
        import cohere

        try:
            self.client = cohere.Client(os.environ["COHERE_KEY"])
        except KeyError as e:
            raise KeyError(
                "You have to set the COHERE_KEY environment"
                " variable to use Cohere embeddings."
            ) from e
        self.model = model
        self.input_type = input_type
        self.batch_size = batch_size

    def encode(self, sentences: Iterable[str]):
        result = []
        for b in batched(sentences, self.batch_size):
            response = self.client.embed(b, input_type=self.input_type)
            result.extend(response.embeddings)
        return np.array(result)
