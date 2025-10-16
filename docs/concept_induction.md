# Concept Induction (BETA)

Concept induction is the idea that higher-level concepts can be discovered and described in detail in corpora using the power of Large Language Models ([Lam et al. 2024](https://arxiv.org/abs/2404.12259)).
These high-level concepts in corpora can also be discovered from particular angles, using seeds.
The original study, and the [Lloom package](https://stanfordhci.github.io/lloom/) uses LLMs all the way, and therefore requires excessive computational resources, and aggressive down-sampling of the original corpus.

In order to account for this scalability issue, we use a [seeded topic model](seeded.md) ([KeyNMF](keynmf.md)) to discover the concepts, and only use LLMs to describe and use them.
This allows us to get similar results to Lloom with a fraction of the costs.

In addition, we allow users to generate a **Concept Browser** programmatically, with which these concepts and their related documents can be explored.

<figure>
  <iframe src="../images/concept_induction.html", title="Concepts discovered on the political ideologies dataset", style="height:1000px;width:1200px;padding:0px;border:none;"></iframe>
  <figcaption> Figure 1: Concepts discovered on the political ideologies dataset. </figcaption>
</figure>

## Example Usage

The example bellow uses a synthetically generated political ideologies dataset, that we examine from the following angles:

  - Taxation
  - Stance on immigration
  - Environmental policy

We use an OpenAI analyzer and KeyNMF, with the `paraphrase-MiniLM-L12-v2` embedding model.
The code runs in about ten minutes.

Install dependencies and set API Key:

```bash
pip install turftopic[openai] datasets
export OPENAI_API_KEY="sk-<your API key here>"
```

```python
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from turftopic import KeyNMF, create_concept_browser
from turftopic.analyzers import OpenAIAnalyzer

# Loading the dataset from huggingface
ds = load_dataset("JyotiNayak/political_ideologies", split="train")
corpus = list(ds["statement"])

# Embedding all documents in the corpus
encoder = SentenceTransformer("paraphrase-MiniLM-L12-v2")
embeddings = encoder.encode(corpus, show_progress_bar=True)

# Running separate seeded KeyNMF models for each tab and saving them
seeds = ["Taxation", "Stance on immigration", "Environmental policy"]
models = []
doc_topic = []
for seed in seeds:
    model = KeyNMF(
        3, encoder=encoder, seed_phrase=seed, seed_exponent=2, random_state=42
    )
    doc_topic_matrix = model.fit_transform(corpus, embeddings=embeddings)
    doc_topic.append(doc_topic_matrix)
    models.append(model)

# Calculating topic sizes
sizes = []
top_documents = []
topic_sizes = []
for doc_topic_matrix in doc_topic:
    # We say that if a document has at least five percent of the max importance
    # then it contains the topic
    rescaled = doc_topic_matrix / doc_topic_matrix.max()
    sizes = (rescaled >= 0.05).sum(axis=0)
    topic_sizes.append(sizes)
    # Finding representative documents for each topic
    docs = []
    for doc_t in rescaled.T:
        # Extracting top 10 documents for each topic
        top = np.argsort(-doc_t)[:10]
        # Making sure only those documents get in,
        # that we have marked to contain the topic
        top = top[doc_t[top] >= 0.05]
        docs.append([corpus[i] for i in top])
    top_documents.append(docs)
topic_sizes = np.stack(topic_sizes)

# Running topic analysis on all models using GPT-5-Nano
analyzer = OpenAIAnalyzer()
analysis_results = []
for model, docs in zip(models, top_documents):
    res = analyzer.analyze_topics(
        keywords=model.get_top_words(), documents=docs
    )
    analysis_results.append(res)

# Creating the concept browser:
browser = create_concept_browser(
    seeds=seeds,
    topic_names=[res.topic_names for res in analysis_results],
    keywords=[model.get_top_words() for model in models],
    topic_descriptions=[res.topic_descriptions for res in analysis_results],
    topic_sizes=topic_sizes,
    top_documents=top_documents,
)
browser.show()
```

_See Figure 1 for the results_

## API reference

::: turftopic._concept_browser.create_browser
