# Encoders

Turftopic by default encodes documents using sentence transformers.
You can always change the encoder model either by passing the name of a sentence transformer from the Huggingface Hub to a model, or by passing a `SentenceTransformer` instance.

Here's an example of building a multilingual topic model by using multilingual embeddings:

```python
from sentence_transformers import SentenceTransformer
from turftopic import GMM

trf = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = GMM(10, encoder=trf)

# or

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

encoder = SentenceTransformer(
    "intfloat/multilingual-e5-large-instruct",
    prompts={
        "query": "Instruct: Retrieve relevant keywords from the given document. Query: "
        "passage": "Passage: "
    },
    # Make sure to set default prompt to query!
    default_prompt_name="query",
)
model = KeyNMF(10, encoder=encoder)
```

And a regular, asymmetric example:

```python
encoder = SentenceTransformer(
    "intfloat/e5-large-v2",
    prompts={
        "query": "query: "
        "passage": "passage: "
    },
    # Make sure to set default prompt to query!
    default_prompt_name="query",
)
model = KeyNMF(10, encoder=encoder)
```


## External Embeddings

If you do not have the computational resources to run embedding models on your own infrastructure, you can also use high quality 3rd party embeddings.
Turftopic currently supports OpenAI, Voyage and Cohere embeddings.

:::turftopic.encoders.base.ExternalEncoder

:::turftopic.encoders.CohereEmbeddings

:::turftopic.encoders.OpenAIEmbeddings

:::turftopic.encoders.VoyageEmbeddings
