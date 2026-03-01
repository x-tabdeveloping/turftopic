# Implement a Wrapper/Custom Model

If you would like to use the convenience of Turftopic, including pretty printing and visualization utilities,
but the model you would like to use is either implemented in another library, or not yet implemented,
you might want to consider writing a Turftopic wrapper/custom model.

The primary interface of Turftopic models is implemented in the `ContextualModel` class, your topic model will have to inherit from this class:

```python
from turftopic.base import ContextualModel
```

## Minimal interface

For a ContextualModel to work you will have to implement the following:

### `__init__`

Implement an `__init__` method that takes and assigns the most basic attributes of your topic model (`n_components`, `encoder`, `vectorizer`).
Some of these attributes are optional, and to align your model's behaviour with the rest of Turftopic, here's some minimal boilerplate.

!!! note
    Your model might not need an `n_components` attribute if it always discovers the number of topics automatically.


```python
from typing import Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from rich.console import Console

from turftopic.base import ContextualModel, Encoder
from turftopic.vectorizers.default import default_vectorizer

class CustomModel(ContextualModel):
    def __init__(
        self,
        n_components: int,
        # You could of course change this to a 
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.random_state = random_state
        if isinstance(encoder, str):
            # Note that we assign the actual encoder to encoder_
            # This is because scikit-learn requires that the attributes and init parameters match
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            # Assign the default vectorizer from Turftopic
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
```

### `fit_transform`

You will also have to implement a `fit_transform` method. This method does the following things:

1. Learns the vocabulary by training the vectorizer.
2. Encodes the documents using the encoder if the embeddings are not provided.
3. Fits the topic model, then assigns the topic-term-matrix to `components_`.
4. Returns the document-topic matrix.

!!! tip
    Turftopic models also use the `rich` Python library for progress tracking during model fitting.
    Note that this is entirely optional, but it makes your model more streamlined with the rest of the library.

Here's a minimal example with some boilerplate code:

```python
    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        # We track progress with a rich console
        console = Console()
        with console.status("Fitting model") as status:
            if embeddings is None:
                status.update("Encoding documents")
                # The encode_documents method is implemented in ContextualModel
                embeddings = self.encode_documents(raw_documents)
                console.log("Documents encoded.")
            status.update("Extracting terms.")
            # It is very important that you fit your vectorizer on the corpus
            # this is how you get vocabulary items by calling get_vocab()
            document_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            status.update("Fitting model")
            # ===========HERE COMES ALL YOUR MODEL FITTING CODE================
            # I'm assigning None to these since we are not implementing a model here,
            # but in your implementation these should be assigned a numpy array
            self.components_ = None
            document_topic_matrix = None
            console.log("Model fitting done.")
        return document_topic_matrix
```

## Example: Wrapper for Latent Dirichlet Allocation

LDA is the most well known and historically the most used topic model.
It is not implemented in Turftopic, since it is not a contextual topic model, but you might need to compare it with Turftopic models,
and it might be convenient to have access to the same interface.

Here's a minimal wrapper for LDA in Turftopic using the boilerplate code above.
I will not implement most hyperparameters, since this would be trivial to do but takes more code.

```python
from sklearn.decomposition import LatentDirichletAllocation

class LDA(ContextualModel):
    """Latent Dirichlet Allocation model wrapper in Turftopic."""
    def __init__(
        self,
        n_components: int,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.random_state = random_state
        # Since LDA only uses bag-of-words, we do not load an encoder
        self.encoder_ = None
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
    
    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        console = Console()
        with console.status("Fitting model") as status:
            status.update("Extracting terms.")
            document_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            status.update("Fitting model")
            self._lda = LatentDirichletAllocation(self.n_components, random_state=self.random_state)
            document_topic_matrix = self._lda.fit_transform(document_term_matrix)
            # Since the scikit-learn API matches perfectly, we won't have to do much.
            self.components_ = self._lda.components_
            console.log("Model fitting done.")
        return document_topic_matrix
```

This model can now be used the same way you would use any other Turftopic model:

```python
from sklearn.datasets import fetch_20newsgroups

ds = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
)
corpus = ds.data

model = LDA(10, random_state=42)
doc_topic = model.fit_transform(corpus)

model.print_topics()
```

```
[10:08:44] Term extraction done.                                                                                                                                                  lda_mess.py:43
[10:09:48] Model fitting done.                                                                                                                                                    lda_mess.py:53
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Topic ID ┃ Highest Ranking                                                                 ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        0 │ game, team, year, games, play, don, think, season, good, hockey                 │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        1 │ drive, use, like, just, used, scsi, don, time, card, power                      │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        2 │ 00, 10, 25, 20, 15, 11, 12, 14, 16, space                                       │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        3 │ people, god, don, think, just, know, like, say, does, time                      │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        4 │ ax, max, g9v, b8f, a86, pl, 145, 1d9, 34u, 1t                                   │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        5 │ know, thanks, like, car, mail, edu, db, just, good, new                         │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        6 │ gun, people, government, law, right, use, fbi, don, guns, control               │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        7 │ windows, dos, use, file, window, program, using, problem, does, like            │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        8 │ edu, image, available, com, data, ftp, information, file, graphics, mail        │
├──────────┼─────────────────────────────────────────────────────────────────────────────────┤
│        9 │ people, said, president, armenian, israel, government, armenians, mr, jews, war │
└──────────┴─────────────────────────────────────────────────────────────────────────────────┘
```


## Dynamic Topic Models

If you want to implement dynamic functionality you will have to inherit from `DynamicTopicModel`,
and implement a `fit_transform_dynamic` method, that also takes timestamps.

!!! note
    You will have to implement the rest of the methods too, as outlined before.

Here's some boilerplate code:

```python
from turftopic.dynamic import DynamicTopicModel

class CustomDynamic(ContextualModel, DynamicTopicModel):
    ...

    def fit_transform_dynamic(
        self,
        raw_documents,
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
    ):
        # bin_timestamps sorts your data into time bins based on the bins the user provided
        # time_labels will be time bin indices for each document
        time_labels, self.time_bin_edges = self.bin_timestamps(
            timestamps, bins
        )
        n_bins = len(self.time_bin_edges) - 1
        # fit vectorizer as before
        doc_term_matrix = self.vectorizer.transform(raw_documents)
        # Overall topics, non-time sensitive
        self.components_ = ...
        # Here I'm assigning zeros,
        # but this attribute should contain the topic-word distributions for each time bin
        self.temporal_components_ = np.zeros(
            (n_bins, self.n_components, len(self.get_vocab))
        )
        # This attribute should contain the importance of a topic for each time bin
        self.temporal_importance_ = np.zeros((n_bins, n_comp))
        # You should of course assign this too
        doc_topic_matrix = ...
        return doc_topic_matrix
```
