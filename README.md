<p align="center">
<img align="center" height="200" src="assets/logo_w_text.svg">
<br>
 <b>Topic modeling is your turf too.</b> <br> <i> Contextual topic models with representations from transformers. </i></p>


### Intentions
 - Provide simple, robust and fast implementations of existing approaches (BERTopic, Top2Vec, CTM) with minimal dependencies.
 - Implement state-of-the-art approaches from my papers. (papers work-in-progress)
 - Put all approaches in a broader conceptual framework.
 - Provide clear and extensive documentation about the best use-cases for each model.
 - Make the models' API streamlined and compatible with topicwizard and scikit-learn.
 - Develop smarter, transformer-based evaluation metrics.

!!!This package is still a prototype, and no papers are published about the models. Until these are out, and most features are implemented
I DO NOT recommend using this package for production and academic use!!!

### Roadmap
 - [x] Model Implementation
 - [x] Pretty Printing
 - [ ] Publish papers :hourglass_flowing_sand: (in progress..)
 - [ ] Thorough documentation and good tutorials ⏳
 - [ ] Implement visualization utilites for these models in topicwizard ⏳
 - [ ] High-level topic descriptions with LLMs.
 - [ ] Contextualized evaluation metrics.


### Implemented Models
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/x-tabdeveloping/turftopic/blob/main/examples/basic_example_20newsgroups.ipynb)
#### Mixture of Gaussians (GMM)

Topic models where topics are assumed to be Multivariate Normal components,
and term importances are estimated with Soft-c-TF-IDF.

```python
from turftopic import GMM

model = GMM(10).fit(texts)
model.print_topics()
```

#### KeyNMF

Nonnegative Matrix Factorization over keyword importances based on transformer representations.

```python
from turftopic import KeyNMF

model = KeyNMF(10).fit(texts)
model.print_topics()
```

#### Semantic Signal Separation (S³)

Interprets topics as dimensions of semantics.
Obtains these dimensions with ICA or PCA.

```python
from turftopic import SemanticSignalSeparation

model = SemanticSignalSeparation(10).fit(texts)
model.print_topics()
```

#### Clustering Topic Models

Topics are clusters in embedding space and term importances are either estimated with c-TF-IDF (BERTopic)
or proximity to cluster centroid (Top2Vec).

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel().fit(texts)
model.print_topics()
```

#### Variational Autoencoders (CTM)

Contextual representations are used as ProdLDA encoder network inputs,
either alone (ZeroShotTM) or concatenated to BoW (CombinedTM).

```bash
pip install turftopic[pyro-ppl]
```

```python
from turftopic import AutoencodingTopicModel

model = AutoencodingTopicModel(10).fit(texts)
model.print_topics()
```


