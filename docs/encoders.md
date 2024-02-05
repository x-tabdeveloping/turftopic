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

## External Embeddings

If you do not have the computational resources to run embedding models on your own infrastructure, you can also use high quality 3rd party embeddings.
Turftopic currently supports OpenAI, Voyage and Cohere embeddings.

:::turftopic.encoders.base.ExternalEncoder

:::turftopic.encoders.CohereEmbeddings

:::turftopic.encoders.OpenAIEmbeddings

:::turftopic.encoders.VoyageEmbeddings
