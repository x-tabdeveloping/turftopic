# Late Interaction Topic Models

Late interaction, or multi-vector models use token representations from a Sentence Transformer before pooling them all together into a single document embedding.
This can be particularly useful for clustering models, as they, by default assign one topic to a single document, but when accessing token representations, can assign topics on a per-token basis.

!!! info
    There are currently no native late-interaction models in Turftopic, meaning models that explicitly model token representations in the context of a document.
    We are currently working on implementing such models, but for the time being, wrappers are included, that can force regular models to use embeddings of higher granularity.
    **Visualization utilities** are also on the way.

## Encoding Tokens, and Ragged Array Manipulation

Turftopic provides a convenience class for encoding documents on a token-level using Sentence Transformers instead of pooling them together into document embeddings.
In order to initialize an encoder, load `LateSentenceTransformer`, and specify which model you would like to use:

!!! tip
    While you could use any encoder model with `LateSentenceTransformer`, we recommend that you stick to ones that have mean pooling, and normalize embeddings.
    This is because in these models, you can be sure that the pooled document embeddings and the token embeddings will be in the same semantic space.

### Token Embeddings

```python
from turftopic.late import LateSentenceTransformer

documents = ["This is a text", "This is another but slightly longer text"]

encoder = LateSentenceTransformer("all-MiniLM-L6-v2")
token_embeddings, offsets = encoder.encode_tokens(documents)
print(token_embeddings)
print(offsets)
```

```python
[
  array([[-0.01135089,  0.04170538,  0.00379963, ...,  0.01383126,
        -0.00274855, -0.05360783],
        ...
       [ 0.05069249,  0.03840942, -0.03545087, ...,  0.03142243,
         0.01929936, -0.09216172]],
        shape=(6, 384), dtype=float32),
  array([[-0.00047079,  0.03402771,  0.00037086, ...,  0.0228903 ,
        -0.01734272, -0.04073172],
       ...,
       [-0.02586325,  0.03737643,  0.02260585, ...,  0.05613737,
        -0.01032581, -0.03799873]], shape=(9, 384), dtype=float32)
]
[[(0, 0), (0, 4), (5, 7), (8, 9), (10, 14), (0, 0)], [(0, 0), (0, 4), (5, 7), (8, 15), (16, 19), (20, 28), (29, 35), (36, 40), (0, 0)]]
```

As you can see, `encode_tokens` returns two arrays, one of them being the token embeddings. This is a ragged array, where longer document can have more embeddings.
`offsets` contains a list of tuples for each document, where the first element of the tuple is the start character of the given token, and the second element is the end character.

### Rolling Window Embeddings

You can also pool these embeddings over a rolling window of tokens.
This way, you still represent your document with multiple vectors, but don't need to model each token individually:

```python
window_embeddings, window_offsets = encoder.encode_windows(documents, window_size=5, step_size=4)
for doc_emb, doc_off in zip(window_embeddings, window_offsets):
    print(doc_emb.shape, doc_off)
```

```python
(2, 384) [(0, 14), (10, 0)]
(3, 384) [(0, 19), (16, 0), (0, 0)]
```

### Ragged array manipulation

These ragged datastructures are hard to deal with, especially when using array operations, so we include convenience functions for manipulating them:
**`flatten_repr`** flattens the ragged array into a single large array, and returns the length of each sub-array:

```python
from turftopic.late import flatten_repr, unflatten_repr

flat_token_embeddings, lengths = flatten_repr(token_embeddings)
print(flat_token_embeddings.shape) 
# (15, 384)
```

**`unflatten_repr`** will turn a flattened representation array into a ragged array:
```python
token_embeddings = unflatten_repr(flat_token_embeddings, lengths)
```

**`pool_flat`** will pool a document representations in a flattened array using a given aggregation function:
```python
import numpy as np
from turftopic.late import pool_flat

pooled = pool_flat(flat_token_embeddings, lengths, agg=np.nanmean)
print(pooled.shape)
# (2, 384)
```

## Turning Regular Models into Multi-Vector Models

The `LateWrapper` class can turn your regular topic models into ones that can utilize windowed or token-level embeddings.
Here's how `LateWrapper` works:

  1. It encodes documents at a token or window-level based on its parameters.
  2. It flattens the embedding array, and feeds the this into the topic model, along with the token/window text.
  3. It unflattens the output of the topic model (`doc_topic_matrix`) into a ragged array, where you get topic importance for each token.
  4. *\[OPTIONAL\]* It pools token-level topic content on the document level, so that you get one document-topic vector for each document instead of each token.

Let's see how this works in practice, and create a [Topeax](Topeax.md) model that uses windowed embeddings instead of document-level embeddings:

```python
from sklearn.datasets import fetch_20newsgroups
from turftopic import Topeax
from turftopic.late import LateWrapper, LateSentenceTransformer

corpus = fetch_20newsgroups(subset="all", categories=["alt.atheism"]).data

model = LateWrapper(
    Topeax(encoder=LateSentenceTransformer("all-MiniLM-L6-v2")),
    window_size=50, # If we don't specify window size, it will use token-level embeddings
    step_size=40, # Since the step size is smaller than the window, we will get overlapping windows
)
doc_topic_matrix, offsets = model.fit_transform(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | morality, moral, morals, immoral, objective, behavior, instinctive, species, inherent, animals |
| 1 | matthew, luke, bible, text, passages, mormon, texts, translations, copy, john |
| 2 | atheism, agnostics, atheist, beliefs, belief, faith, contradictory, believers, contradictions, theists |
| 3 | punishment, cruel, abortion, penalty, death, constitution, homosexuality, painless, capital, punish |
| 4 | war, arms, invaded, gulf, hussein, civilians, military, kuwait, peace, sell |
| 5 | islam, islamic, muslim, qur, muslims, imams, rushdie, quran, koran, khomeini |

The document-topic matrix, we created, is now a ragged array and contains document-topic proportions for each window in a document.
Let's see what this means in practice for the first document in our corpus:
```python
import pandas as pd

# We select document 0, then collect all information into a dataframe:
window_topic_matrix = doc_topic_matrix[0]
window_offs = offsets[0]
document = corpus[0]
# We extract the text for each window based on the offsets
window_text = [document[window_start: window_end] for window_start, window_end in window_offs]
df = pd.DataFrame(window_topic_matrix, index=window_text, columns=model.topic_names)
print(df)
```

```python
                                                    0_morality_moral_morals_immoral  1_matthew_luke_bible_text  ...  4_war_arms_invaded_gulf  5_islam_islamic_muslim_qur
From: acooper@mac.cc.macalstr.edu (Turin Turamb...                         0.334267               1.287207e-13  ...             2.626869e-26                1.459101e-04
alester College\nLines: 55\n\nIn article <C5sA2...                         0.360400               8.898302e-14  ...             3.290858e-26                1.382718e-04
u (Mike Cobb) writes:\n> I guess I'm delving in...                         0.847002               5.002921e-22  ...             4.852574e-41                3.141366e-07
this you just have a spiral.  What\nwould then ...                         0.848413               5.819050e-22  ...             8.139559e-41                3.286224e-07
, even though this would hardly seem moral.  Fo...                         0.863685               1.272204e-21  ...             2.823941e-41                2.815930e-07
whatever helps this goal is\n"moral", whatever ...                         0.864913               1.584558e-21  ...             5.780971e-41                3.003952e-07
a "hyper-morality" to apply to just the methods...                         0.865558               1.919885e-21  ...             1.251694e-40                3.231265e-07
not doing something because it is\n> a personal...                         0.868360               2.951441e-21  ...             3.085662e-40                3.494368e-07
we only consider something moral or immoral if ...                         0.872827               5.444738e-21  ...             4.708349e-40                3.580695e-07
here we have a way to discriminate\nmorals.  I ...                         0.876951               1.021014e-20  ...             3.486096e-40                3.411401e-07
enough and\nlistened to the arguments, I could ...                         0.878680               2.302363e-20  ...             5.866410e-40                3.565728e-07
.  Or, as you brought out,\n> if whatever is ri...                         0.878953               3.004052e-20  ...             5.977738e-40                3.566668e-07
> *******************************                                          0.647793               5.664651e-17  ...             1.805073e-19                4.612731e-04
```

## C-Top2Vec

Contextual Top2Vec [(Angelov and Inkpen, 2024)](https://aclanthology.org/2024.findings-emnlp.790/) is a late-interaction topic model, that uses windowed representations.
The model is essentially the same as wrapping a regular Top2vec model in `LateWrapper`, but we provide a convenience class in Turftopic, so that it's easy for you to initialize this model.
It comes pre-loaded with the following features:

   - Same hyperparameters as in Angelov and Inkpen (2024)
   - Phrase-vectorizer that finds regular phrases based on PMI
   - `LateSentenceTransformer` by default, you can specify any model.

Our implementation is much more flexible than the original top2vec package, and you might be able to use much more powerful or novel embedding models.

```python
from turftopic import CTop2Vec

model = CTop2Vec(n_reduce_to=5)
doc_topic_matrix = model.fit_transform(corpus)

model.print_topics()
```


| Topic ID | Highest Ranking |
| - | - |
| -1 | caused atheism organization, genocide caused atheism, atheism organization, atheism, subject political atheists, alt atheism, caused atheism, political atheists organization, subject amusing atheists, amusing atheists |
| 166 | atheists organization, political atheists organization, christian morality organization, caused atheism organization, morality organization, atheism organization, atheists organization california, subject amusing atheists, cwru edu article, alt atheism |
| 172 | biblical, read bible, caused atheism, agnostics, caused atheism organization, atheists agnostics, christianity, alt atheism, atheism, christian morality organization |
| 173 | objective morality, morality, subject christian morality, christian morality, natural morality, say christian morality, morality organization, christian morality organization, behavior moral, moral |
| 175 | atheism, atheism organization, caused atheism organization, atheists agnostics, caused atheism, subject political atheists, alt atheism, genocide caused atheism, subject amusing atheists, amusing atheists |
| 176 | rushdie islamic law, subject rushdie islamic, islamic genocide, islamic law, genocide caused atheism, subject islamic, islamic law organization, islamic genocide organization, rushdie islamic, islamic authority |

You might also observe that the output of this model is a regular document-topic matrix, and isn't ragged.
```python
print(doc_topic_matrix.shape)
# (1024, 6)
```

This is because this way the model has the same API, as other Turftopic models, and works the same way as the top2vec package, making migration easier.

## API Reference 

### Encoder

::: turftopic.late.LateSentenceTransformer

### Wrapper

::: turftopic.late.LateWrapper

### Utility functions

::: turftopic.late.flatten_repr

::: turftopic.late.unflatten_repr

::: turftopic.late.pool_flat

::: turftopic.late.get_document_chunks

