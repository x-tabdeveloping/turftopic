# Encoders

!!! warning
    We now recommend that you do NOT pre-load `SentenceTransformer` encoder models, but rather pass them by name to the topic models.
    This allows the model not to store the encoder model on disk, and load it when loading the model.
    This can lead to extreme reductions in disk space usage.
    For example, instead of writing:
    ```python
    model = SensTopic(
        encoder=SentenceTransformer("intfloat/multilingual-e5-large-instruct", default_prompt_name="query")
    )
    ```
    Write:
    ```python
    model = SensTopic(
        encoder="intfloat/multilingual-e5-large-instruct",
        trf_kwargs=dict(default_prompt_name="query")
    )
    ```


Turftopic by default encodes documents using sentence transformers.
You can always change the encoder model either by passing the name of a sentence transformer from the Huggingface Hub to a model, or by passing a `SentenceTransformer` instance.

Here's an example of building a multilingual topic model by using multilingual embeddings:

```python
from turftopic import GMM

model = GMM(10, encoder="paraphrase-multilingual-MiniLM-L12-v2")
```

Different encoders have different performance and model sizes.
To make an informed choice about which embedding model you should be using check out the [Massive Text Embedding Benchmark](https://huggingface.co/blog/mteb).

## Asymmetric and Instruction-tuned Embedding Models

Some embedding models can be used together with prompting, or encode queries and passages differently.
Microsoft's E5 models are, for instance, all prompted by default, and it would be detrimental to performance not to do so yourself.

In these cases, you're better off NOT passing a string to Turftopic models, but explicitly loading the model using `sentence-transformers`.

Here's an example of using instruct models for keyword retrieval with KeyNMF.
In this case, documents will serve as the queries and words as the passages:

```python
from turftopic import KeyNMF
from sentence_transformers import SentenceTransformer

model = KeyNMF(
    10,
    encoder="intfloat/multilingual-e5-large-instruct",
    # These are the arguments that get applied when loading the encoder.
    trf_kwargs=dict(
        prompts={
            "query": "Instruct: Retrieve relevant keywords from the given document. Query: "
            "passage": "Passage: "
        },
        # Make sure to set default prompt to query!
        default_prompt_name="query",
    )
)
```

And a regular, asymmetric example:

```python
model = KeyNMF(
    10,
    encoder="intfloat/e5-large-v2",
    trf_kwargs=dict(
        prompts={
            "query": "query: "
            "passage": "passage: "
        },
        # Make sure to set default prompt to query!
        default_prompt_name="query",
    )
)
```

## Performance tips

From `sentence-transformers` version `3.2.0` you can significantly speed up some models by using
the `onnx` backend instead of regular torch.

```
pip install sentence-transformers[onnx, onnx-gpu]
```

```python
from turftopic import SemanticSignalSeparation
from sentence_transformers import SentenceTransformer


model = SemanticSignalSeparation(10, encoder="all-MiniLM-L6-v2", trf_kwargs=dict(backend="onnx"))
```

## External Embeddings

If you do not have the computational resources to run embedding models on your own infrastructure, you can also use high quality 3rd party embeddings.
Turftopic currently supports OpenAI, Voyage and Cohere embeddings.

:::turftopic.encoders.base.ExternalEncoder

:::turftopic.encoders.CohereEmbeddings

:::turftopic.encoders.OpenAIEmbeddings

:::turftopic.encoders.VoyageEmbeddings
