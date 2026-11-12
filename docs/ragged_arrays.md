# Ragged Arrays

As mentioned before, late interaction models in Turftopic use ragged arrays to represent spans either with embeddings or document-topic matrices.
While some third-party tools are available for this, we opted for a very simple model for ragged arrays with standard Python objects.

## Data Model

Ragged arrays in Turftopic are made up of two very important parts, a list of 2D arrays, that include the vectors for each span in each document, and an offsets object, that contains the start and end index of each span in each document.

```python
# List of arrays of length n_documents,
# each with shape (n_spans, n_dimensions)
ArrayList = list[np.ndarray | spr.sparray]

# Offsets in a single document
Offsets = list[tuple[int, int]]

# You might encounter the RaggedArray type descriptor in the documentation,
# this is just an alias for a tuple of array list and offsets
RaggedArray = tuple[ArrayList, list[Offsets]]
```

Let's look at some examples of what ragged arrays are used for in Turftopic's late interaction API.

## Embeddings

In the late API you initialize the encoder in the same way as anywhere else.
We recommend that you pass the encoder as a string, and pass its arguments in `trf_kwargs` and `encode_kwargs`.
For more elaboration see our page on [Encoders](encoders.md).

```python
from turftopic import vSLDA

model = vSLDA(
    10,
    encoder="paraphrase-multilingual-mpnet-base-v2",
    encode_kwargs=dict(show_progress_bar=True),
)
```

The encoder wil be loaded as a `MultiVectorEncoder` from sentence-transformers.
If you pass a `SentenceTransformer` it will be implicitly converted to a `LateSentenceTransformer` object,
which is a shallow wrapper around sentence transformers so that one can extract token embeddings out of them.

### Token embeddings

Embeddings in late interaction models are represented as ragged arrays of token embeddings.
You can extract these by using the `encode_tokens` method:

```python
token_embeddings, token_offsets = model.encode_tokens(corpus)
print(token_embeddings)
print(token_offsets)
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
  ...
]
[[(0, 0), (0, 4), (5, 7), (8, 9), (10, 14), (0, 0)], [(0, 0), (0, 4), (5, 7), (8, 15), (16, 19), (20, 28), (29, 35), (36, 40), (0, 0)], ...]
```

You will, however rarely be dealing with token embeddings in their pure form.
Most of the times late interaction models use span embeddings.

### Span embeddings

Late interaction models typically model Span embeddings.
Spans are defined by the user, and determine the granularity of the analysis.
To read more about spans, read our documentation page on [Span extraction](spans.md).

In the model defined above, the default spans are sentences.
To extract span embeddings, use the `encode_late` method:

```python
span_embeddings, span_offsets = model.encode_late(corpus)
```

### Document-topic matrices

If you intend to use already-computed span embeddings to fit a model, you can pass them along to the model's `fit` or `fit_transform method`:

```python
doc_topic_matrix, span_offsets = model.fit_transform(
    corpus,
    embeddings=span_embeddings,
    offsets=span_offsets
)
```

If the model's `pooling` attribute is `None` (default), then the returned `doc_topic_matrix` will also function as a ragged array.

### Document-term matrices

Internally, late interaction topic models also use ragged document-term matrices produced by their [Vectorizer](vectorizers.md).
Vectorizers don't have to be modified to work in this context.
You can access the ragged document-term matrix by using the `ragged_vectorize` method:

```python
dtm = model.ragged_vectorize(corpus, span_offsets)
```

In this case the matrix will be a list of sparse arrays.


## Ragged array manipulation

Ragged arrays can be hard to work with, especially when it comes to using them in different computations.
As such late-interaction models come with a couple of utilites for flattening, unflattening and pooling ragged arrays.

You can flatten ragged arrays using `flatten_repr`:
```python
# You get an array of all span vectors in the corpus and
# The number of spans in each document, so that it can be unflattened again.
flat_span_embeddings, lengths = model.flatten_repr(span_embeddings)
print(flat_span_embeddings.shape) 
# (n_tokens, n_dimensions)
```

`unflatten_repr` will turn a flattened representation array into a ragged array:
```python
unflattened_span_embeddings = model.unflatten_repr(flat_span_embeddings, lengths)
assert all([np.all(s==us) for s, us in zip(span_embeddings, unflattened_span_embeddings)])
```

`pool_flat` will pool a document representations in a flattened array using a given aggregation function:
```python
pooled = pool_flat(flat_span_embeddings, lengths, agg=np.nanmean)
print(pooled.shape)
# (n_documents, n_dimensions)
```

