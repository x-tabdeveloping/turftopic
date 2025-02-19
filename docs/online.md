# Online Topic Modeling

Some models in Turftopic can be fitted in an online manner (currently this only includes [KeyNMF](KeyNMF.md)).
These models can be fitted in minibatches instead of the entire corpus at the same time.

#### Use Cases:

1. You can use online fitting when you have **very large corpora** at hand, and it would be impractical to fit a model on it at once.
2. You have **new data flowing in constantly**, and need a model that can morph the topics based on the incoming data. You can also do this in a dynamic fashion.
3. You need to **finetune** an already fitted topic model to novel data.


## Batch Fitting

We will use the batching function from the itertools recipes to produce batches.

> In newer versions of Python (>=3.12) you can just `from itertools import batched`

```python
def batched(iterable, n: int):
    "Batch data into lists of length n. The last batch may be shorter."
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
```

You can fit a model to a very large corpus in batches like so:

```python
from turftopic import KeyNMF

model = KeyNMF(10, top_n=5)

corpus = ["some string", "etc", ...]
for batch in batched(corpus, 200):
    batch = list(batch)
    model.partial_fit(batch)
```

You might want to train in epochs, so that the model sees the same documents multiple times, this might be useful in numerous settings:

```python
for epoch in range(5):
  for batch in batched(corpus, 200):
      batch = list(batch)
      model.partial_fit(batch)
```


## Finetuning a Model

You can pretrain a topic model on a large corpus and then finetune it on a novel corpus the model has not seen before.
This will morph the model's topics to the corpus at hand.

In this example I will load a pretrained KeyNMF model from disk. (see [Model Loading and Saving](persistence.md))

```python
from turftopic import load_model

model = load_model("pretrained_keynmf_model")

new_corpus: list[str] = [...]
# Finetune the model to the new corpus
model.partial_fit(new_corpus)

model.to_disk("finetuned_model/")
```

## Precomputed Embeddings

In the case of very large corpora it is common to precompute embeddings before fitting the model.
You can still do this with `partial_fit()`, you just have to be careful to correctly match the embedding indices with the corpus indices.

We provide an example of correct usage here.

You might have a `utils.py` file with a function to load your corpus:
```python
def load_corpus() -> list[str]:
    """Function that loads the corpus from some source."""
    ...
```

Then you have a file which computes the embeddings and saves them to disk:
```python
import numpy as np
from sentence_transformers import SentenceTransformers

from utils import load_corpus

corpus = load_corpus()

trf = SentenceTransformers("all-MiniLM-L6-v2")
embeddings = trf.encode(corpus)

np.save("embeddings.npy")
```

This file then trains the model on the precomputed embeddings:
```python
import numpy as np
from turftopic import KeyNMF

from utils import load_corpus

corpus = load_corpus()
embeddings = np.load("embeddings.npy")

model = KeyNMF(10, encoder="all-MiniLM-L6-v2")
for batch in batched(zip(corpus, embeddings), 200):
    text_batch, embedding_batch = zip(*batch)
    text_batch = list(text_batch)
    embedding_batch = np.stack(embedding_batch)
    model.partial_fit(text_batch, embeddings=embedding_batch)
```
