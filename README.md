<p align="center">
<img align="center" height="200" src="assets/logo_w_text.svg">
<br>
 <b>Topic modeling is your turf too.</b> <br> <i> Contextual topic models with representations from transformers. </i></p>


## Features
| | |
| - | - |
| SOTA Transformer-based Topic Models | :compass: [S³](https://x-tabdeveloping.github.io/turftopic/s3/), :key: [KeyNMF](https://x-tabdeveloping.github.io/turftopic/KeyNMF/),  :gem: [GMM](https://x-tabdeveloping.github.io/turftopic/GMM/), [Clustering Models (BERTopic and Top2Vec)](https://x-tabdeveloping.github.io/turftopic/GMM/), [Autoencoding models (ZeroShotTM and CombinedTM)](https://x-tabdeveloping.github.io/turftopic/ctm/), [FASTopic](https://x-tabdeveloping.github.io/turftopic/FASTopic/) |
| Models for all Scenarios | :chart_with_upwards_trend: [Dynamic](https://x-tabdeveloping.github.io/turftopic/dynamic/), :ocean: [Online](https://x-tabdeveloping.github.io/turftopic/online/), :herb: [Seeded](https://x-tabdeveloping.github.io/turftopic/seeded/), :evergreen_tree: [Hierarchical](https://x-tabdeveloping.github.io/turftopic/hierarchical/), and :camera: [Multimodal](https://x-tabdeveloping.github.io/turftopic/multimodal/)  topic modeling |
| [Easy Interpretation](https://x-tabdeveloping.github.io/turftopic/model_interpretation/) | :bookmark_tabs: Pretty Printing, :bar_chart: Interactive Figures, :art: [topicwizard](https://github.com/x-tabdeveloping/topicwizard) compatible |
| [Topic Naming](https://x-tabdeveloping.github.io/turftopic/namers/) | :robot: LLM-based, N-gram Retrieval, :wave: Manual |
| [Informative Topic Descriptions](https://x-tabdeveloping.github.io/turftopic/vectorizers/) | :key: Keyphrases, Noun-phrases, Lemmatization, Stemming |


## Basics [(Documentation)](https://x-tabdeveloping.github.io/turftopic/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/x-tabdeveloping/turftopic/blob/main/examples/basic_example_20newsgroups.ipynb)

For detailed model comparison and tutorials, consult our documentation: [Click here](https://x-tabdeveloping.github.io/turftopic/)

### Installation

Turftopic can be installed from PyPI.

```bash
pip install turftopic
```

If you intend to use CTMs, make sure to install the package with Pyro as an optional dependency.

```bash
pip install turftopic[pyro-ppl]
```

If you want to use clustering models like BERTopic or Top2Vec, install:

```bash
pip install turftopic[umap-learn]
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

#### Automated Topic Naming

Turftopic now allows you to automatically assign human readable names to topics using LLMs or n-gram retrieval!

```python
from turftopic import KeyNMF
from turftopic.namers import OpenAITopicNamer

model = KeyNMF(10).fit(corpus)

namer = OpenAITopicNamer("gpt-4o-mini")
model.rename_topics(namer)
model.print_topics()
```

| Topic ID | Topic Name | Highest Ranking |
| - | - | - |
| 0 | Operating Systems and Software  | windows, dos, os, ms, microsoft, unix, nt, memory, program, apps |
| 1 | Atheism and Belief Systems | atheism, atheist, atheists, belief, religion, religious, theists, beliefs, believe, faith |
| 2 | Computer Architecture and Performance | motherboard, ram, memory, cpu, bios, isa, speed, 486, bus, performance |
| 3 | Storage Technologies | disk, drive, scsi, drives, disks, floppy, ide, dos, controller, boot |
| | ... |

### Vectorizers Module

You can use a set of custom vectorizers for topic modeling over **phrases**, as well as **lemmata** and **stems**.

```python
from turftopic import BERTopic
from turftopic.vectorizers.spacy import NounPhraseCountVectorizer

model = BERTopic(
    n_components=10,
    vectorizer=NounPhraseCountVectorizer("en_core_web_sm"),
)
model.fit(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| | ... |
| 3 | fanaticism, theism, fanatism, all fanatism, theists, strong theism, strong atheism, fanatics, precisely some theists, all theism |
| 4 | religion foundation darwin fish bumper stickers, darwin fish, atheism, 3d plastic fish, fish symbol, atheist books, atheist organizations, negative atheism, positive atheism, atheism index |
| | ... |

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

## References
- Kardos, M., Kostkan, J., Vermillet, A., Nielbo, K., Enevoldsen, K., & Rocca, R. (2024, June 13). $S^3$ - Semantic Signal separation. arXiv.org. https://arxiv.org/abs/2406.09556
- Wu, X., Nguyen, T., Zhang, D. C., Wang, W. Y., & Luu, A. T. (2024). FASTopic: A Fast, Adaptive, Stable, and Transferable Topic Modeling Paradigm. ArXiv Preprint ArXiv:2405.17978.
 - Grootendorst, M. (2022, March 11). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv.org. https://arxiv.org/abs/2203.05794
 - Angelov, D. (2020, August 19). Top2VEC: Distributed representations of topics. arXiv.org. https://arxiv.org/abs/2008.09470
 - Bianchi, F., Terragni, S., & Hovy, D. (2020, April 8). Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence. arXiv.org. https://arxiv.org/abs/2004.03974
 - Bianchi, F., Terragni, S., Hovy, D., Nozza, D., & Fersini, E. (2021). Cross-lingual Contextualized Topic Models with Zero-shot Learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume (pp. 1676–1683). Association for Computational Linguistics.
 - Kristensen-McLachlan, R. D., Hicke, R. M. M., Kardos, M., & Thunø, M. (2024, October 16). Context is Key(NMF): Modelling Topical Information Dynamics in Chinese Diaspora Media. arXiv.org. https://arxiv.org/abs/2410.12791
