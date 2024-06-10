<p align="center">
<img align="center" height="200" src="assets/logo_w_text.svg">
<br>
 <b>Topic modeling is your turf too.</b> <br> <i> Contextual topic models with representations from transformers. </i></p>


## Features
 - Novel transformer-based topic models:
   - Semantic Signal Separation - S³ (paper in progress ⏳)
   - KeyNMF 🔑
   - GMM
 - Implementations of existing transformer-based topic models
   - Clustering Topic Models: BERTopic and Top2Vec
   - Autoencoding Topic Models: CombinedTM and ZeroShotTM
 - Streamlined scikit-learn compatible API 🛠️
 - Easy topic interpretation 🔍
 - Dynamic Topic Modeling 📈 (GMM, ClusteringTopicModel and KeyNMF)
 - Visualization with [topicwizard](https://github.com/x-tabdeveloping/topicwizard) 🖌️

> This package is still work in progress and scientific papers on some of the novel methods are currently undergoing peer-review. If you use this package and you encounter any problem, let us know by opening relevant issues.

#### New in version 0.3.0: Dynamic KeyNMF
KeyNMF can now be used for dynamic topic modeling.

```python
from datetime import datetime
from turftopic import KeyNMF

corpus: list[str] = [...]
timestamps = list[datetime] = [...]

model = KeyNMF(10)
doc_topic_matrix = model.fit_transform_dynamic(corpus, timestamps=timestamps, bins=10)

model.print_topics_over_time()

# This needs Plotly: pip install plotly
model.plot_topics_over_time()
```

## Basics [(Documentation)](https://x-tabdeveloping.github.io/turftopic/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/x-tabdeveloping/turftopic/blob/main/examples/basic_example_20newsgroups.ipynb)

### Installation

Turftopic can be installed from PyPI.

```bash
pip install turftopic
```

If you intend to use CTMs, make sure to install the package with Pyro as an optional dependency.

```bash
pip install turftopic[pyro-ppl]
```

### Fitting a Model

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
```

### Interpreting Models

Turftopic comes with a number of pretty printing utilities for interpreting the models.

To see the highest the most important words for each topic, use the `print_topics()` method.

```python
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

```python
# Print highest ranking documents for topic 0
model.print_representative_documents(0, corpus, document_topic_matrix)
```

<center>

| Document                                                                                             | Score |
| -----------------------------------------------------------------------------------------------------| ----- |
| Poor 'Poly'. I see you're preparing the groundwork for yet another retreat from your...              |  0.40 |
| Then you must be living in an alternate universe. Where were they? An Appeal to Mankind During the... |  0.40 |
| It is 'Serdar', 'kocaoglan'. Just love it. Well, it could be your head wasn't screwed on just right... |  0.39 |

</center>

```python
model.print_topic_distribution(
    "I think guns should definitely banned from all public institutions, such as schools."
)
```

<center>

| Topic name                                | Score |
| ----------------------------------------- | ----- |
| 7_gun_guns_firearms_weapons               |  0.05 |
| 17_mail_address_email_send                |  0.00 |
| 3_encryption_chip_clipper_nsa             |  0.00 |
| 19_baseball_pitching_pitcher_hitter       |  0.00 |
| 11_graphics_software_program_3d           |  0.00 |

</center>

### Visualization

Turftopic does not come with built-in visualization utilities, [topicwizard](https://github.com/x-tabdeveloping/topicwizard), an interactive topic model visualization library, is compatible with all models from Turftopic.

```bash
pip install topic-wizard
```

By far the easiest way to visualize your models for interpretation is to launch the topicwizard web app.

```python
import topicwizard

topicwizard.visualize(corpus, model=model)
```

<figure>
  <img src="https://x-tabdeveloping.github.io/topicwizard/_images/screenshot_topics.png" width="70%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Screenshot of the topicwizard Web Application</figcaption>
</figure>

Alternatively you can use the [Figures API](https://x-tabdeveloping.github.io/topicwizard/figures.html) in topicwizard for individual HTML figures.

## Models

| Model | Description | Usage |
| - | - | - |
| KeyNMF | Non-negative Matrix Factorization enhanced with keyword extraction using sentence embeddings | `model = KeyNMF(n_components=10).fit(corpus)` |
| GMM | Gaussian Mixture Model over contextual embeddings + post-hoc term importance estimation | `model = GMM(n_components=10).fit(corpus)` |
| S³ | Separates semantic signals, aka. axes of semantics in a corpus using independent component analysis. | `model = SemanticSignalSeparation(n_components=10).fit(corpus)` |
| Autoencoding Models | Learn topics using amortized variational inference enhanced by contextual representations.  | `model = AutoEncodingTopicModel(n_components=10, combined=False).fit(corpus)` |
| Clustering Models | Clusters semantic embeddings, and estimates term importances for clusters.  | `model = ClusteringTopicModel(feature_importance="ctfidf").fit(corpus)` |

For extensive comparison see our [Model Overview](https://x-tabdeveloping.github.io/turftopic/model_overview/).
