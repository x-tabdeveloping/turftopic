# KeyNMF

KeyNMF is a topic model that relies on contextually sensitive embeddings for keyword retrieval and term importance estimation,
while taking inspiration from classical matrix-decomposition approaches for extracting topics.

## The Model

<figure>
  <img src="/images/keynmf.png" width="90%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Schematic overview of KeyNMF</figcaption>
</figure>

### 1. Keyword Extraction

The first step of the process is gaining enhanced representations of documents by using contextual embeddings.
Both the documents and the vocabulary get encoded with the same sentence encoder.

Keywords are assigned to each document based on the cosine similarity of the document embedding to the embedded words in the document.
Only the top K words with positive cosine similarity to the document are kept.

These keywords are then arranged into a document-term importance matrix where each column represents a keyword that was encountered in at least one document,
and each row is a document.
The entries in the matrix are the cosine similarities of the given keyword to the document in semantic space.

### 2. Topic Discovery

Topics in this matrix are then discovered using Non-negative Matrix Factorization.
Essentially the model tries to discover underlying dimensions/factors along which most of the variance in term importance
can be explained.

## Considerations

### Strengths

 - Stability, Robustness and Quality: KeyNMF extracts very clean topics even when a lot of noise is present in the corpus, and the model's performance remains relatively stable across domains.
 - Scalability: The model can be fitted in an online fashion, and we recommend that you choose KeyNMF when the number of documents is large (over 100 000).
 - Fail Safe and Adjustable: Since the modelling process consists of multiple easily separable steps it is easy to repeat one if something goes wrong. This also makes it an ideal choice for production usage.
 - Can capture multiple topics in a document.

### Weaknesses

 - Lack of Multilingual Capabilities: KeyNMF as it is currently implemented cannot be used in a multilingual context. Changes to the model that allow this are possible, and will likely be ijmplemented in the future.
 - Lack of Nuance: Since only the top K keywords are considered and used for topic extraction some of the nuances, especially in long texts might get lost. We therefore recommend that you scale K with the average length of the texts you're working with. For tweets it might be worth it to scale it down to 5, while with longer documents, a larger number (let's say 50) might be advisable.
 - Practitioners have to choose the number of topics a priori.
