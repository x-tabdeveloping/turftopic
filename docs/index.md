# Getting Started

Turftopic is a topic modeling library which intends to simplify and streamline the usage of contextually sensitive topic models.
We provide stable, minimal and scalable implementations of several types of models along with extensive documentation.

<center>

| | | |
| - | - | - |
|   :house: [Build and Train Topic Models](model_definition_and_training.md) |  :art: [Explore, Interpret and Visualize your Models](model_interpretation.md) | :wrench: [Modify and Fine-tune Topic Models](finetuning.md) |
|  :pushpin:  [Choose the Right Model for your Use-Case](model_overview.md) |  :chart_with_upwards_trend: [Explore Topics Changing over Time](dynamic.md)   |  :newspaper: [Use Phrases or Lemmas for Topic Models](vectorizers.md) |
| :ocean: [Extract Topics from a Stream of Documents](online.md) |  :evergreen_tree: [Find Hierarchical Order in Topics](hierarchical.md) |  :whale: [Name Topics with Large Language Models](namers.md) |

</center>

## Basic Usage

Turftopic can be installed from PyPI.

```bash
pip install turftopic
```

Turftopic's models follow the scikit-learn API conventions, and as such they are quite easy to use if you are familiar with
scikit-learn workflows.

Here's an example of how you use KeyNMF, one of our models on the 20Newsgroups dataset from scikit-learn.

```python
from turftopic import KeyNMF
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
)
corpus = newsgroups.data
model = KeyNMF(20).fit(corpus)
model.print_topics()
```

<center>

| Topic ID | Top 10 Words                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------- |
|        0 | armenians, armenian, armenia, turks, turkish, genocide, azerbaijan, soviet, turkey, azerbaijani |
|        1 | sale, price, shipping, offer, sell, prices, interested, 00, games, selling                      |
|         | ....                                |

</center>



