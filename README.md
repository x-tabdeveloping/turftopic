<p align="center">
<img align="center" height="200" src="assets/logo_w_text.svg">
<br>
 <b>Topic modeling is your turf too.</b> <br> <i> Contextual topic models with representations from transformers. </i></p>


## Features
 - Novel transformer-based topic models:
   - Semantic Signal Separation - S³ 🧭
   - KeyNMF 🔑 (paper in progress ⏳)
   - GMM :gem: (paper soon)
 - Implementations of other transformer-based topic models
   - Clustering Topic Models: BERTopic and Top2Vec
   - Autoencoding Topic Models: CombinedTM and ZeroShotTM
   - FASTopic
 - Streamlined scikit-learn compatible API 🛠️
 - Easy topic interpretation 🔍
 - Dynamic Topic Modeling 📈 (GMM, ClusteringTopicModel and KeyNMF)
 - Visualization with [topicwizard](https://github.com/x-tabdeveloping/topicwizard) 🖌️

> This package is still work in progress and scientific papers on some of the novel methods are currently undergoing peer-review. If you use this package and you encounter any problem, let us know by opening relevant issues.

### New in version 0.6.0

#### Prompting Embedding Models

KeyNMF and clustering topic models can now efficiently utilise asymmetric and instruction-finetuned embedding models.
This, in combination with the right embedding model, can enhance performance significantly.

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

## References
- Kardos, M., Kostkan, J., Vermillet, A., Nielbo, K., Enevoldsen, K., & Rocca, R. (2024, June 13). $S^3$ - Semantic Signal separation. arXiv.org. https://arxiv.org/abs/2406.09556
- Wu, X., Nguyen, T., Zhang, D. C., Wang, W. Y., & Luu, A. T. (2024). FASTopic: A Fast, Adaptive, Stable, and Transferable Topic Modeling Paradigm. ArXiv Preprint ArXiv:2405.17978.
 - Grootendorst, M. (2022, March 11). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv.org. https://arxiv.org/abs/2203.05794
 - Angelov, D. (2020, August 19). Top2VEC: Distributed representations of topics. arXiv.org. https://arxiv.org/abs/2008.09470
 - Bianchi, F., Terragni, S., & Hovy, D. (2020, April 8). Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence. arXiv.org. https://arxiv.org/abs/2004.03974
 - Bianchi, F., Terragni, S., Hovy, D., Nozza, D., & Fersini, E. (2021). Cross-lingual Contextualized Topic Models with Zero-shot Learning. In Proceedings of the 16th Conference of the European 
 - Chapter of the Association for Computational Linguistics: Main Volume (pp. 1676–1683). Association for Computational Linguistics.
