# Getting Started

Turftopic is a topic modeling library which intends to simplify and streamline the usage of contextually sensitive topic models.
We provide stable, minimal and scalable implementations of several types of models along with extensive documentation,
so that you can make an informed choice about which model suits you best in the light of a given task or research question.

## Installation

Turftopic can be installed from PyPI.

```bash
pip install turftopic
```

If you intend to use CTMs, make sure to install the package with Pyro as an optional dependency.

```bash
pip install turftopic[pyro-ppl]
```

## Models

You can use most transformer-based topic models in Turftopic, these include:

 - [Semantic Signal Separation - $S^3$](s3.md) :compass:
 - [KeyNMF](KeyNMF.md) :key:
 - [Gaussian Mixture Models (GMM)](gmm.md)
 - [Clustering Topic Models](clustering.md):
    - [BERTopic](clustering.md#bertopic_and_top2vec)
    - [Top2Vec](clustering.md#bertopic_and_top2vec)
 - [Auto-encoding Topic Models](ctm.md):
    - CombinedTM
    - ZeroShotTM
 - [FASTopic](fastopic.md) :zap:



## Basic Usage

Turftopic's models follow the scikit-learn API conventions, and as such they are quite easy to use if you are familiar with
scikit-learn workflows.

Here's an example of how you use KeyNMF, one of our models on the 20Newsgroups dataset from scikit-learn.

```python
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
)
corpus = newsgroups.data
```

Turftopic also comes with interpretation tools that make it easy to display and understand your results.

```python
from turftopic import KeyNMF

model = KeyNMF(20).fit(corpus)
model.print_topics()
```

<center>

| Topic ID | Top 10 Words                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------- |
|        0 | armenians, armenian, armenia, turks, turkish, genocide, azerbaijan, soviet, turkey, azerbaijani |
|        1 | sale, price, shipping, offer, sell, prices, interested, 00, games, selling                      |
|        2 | christians, christian, bible, christianity, church, god, scripture, faith, jesus, sin           |
|        3 | encryption, chip, clipper, nsa, security, secure, privacy, encrypted, crypto, cryptography      |
|         | ....                                |

</center>


