# C-Top2Vec

Contextual Top2Vec [(Angelov and Inkpen, 2024)](https://aclanthology.org/2024.findings-emnlp.790/) is a [late-interaction topic model](late_interaction.md), that uses windowed representations.

!!! info
    This part of the documentation is still in the works.
    More information, visualizations and benchmark results are on their way.

The model is essentially the same as wrapping a regular Top2vec model in `LateWrapper`, but we provide a convenience class in Turftopic, so that it's easy for you to initialize this model.
It comes pre-loaded with the following features:

   - Same hyperparameters as in Angelov and Inkpen (2024)
   - Phrase-vectorizer that finds regular phrases based on PMI
   - `LateSentenceTransformer` by default, you can specify any model.

Our implementation is much more flexible than the original top2vec package, and you might be able to use much more powerful or novel embedding models.

!!! tip
    For more info about multi-vector/late-interaction models, read our [User Guide](late-interaction.md).

## Example Usage 

You should install Turftopic with UMAP in order to be able to use C-Top2Vec:

```bash
pip install turftopic[umap-learn]
```

Then use the topic model as you would use any other model in Turftopic:

```python
from turftopic import CTop2Vec

model = CTop2Vec(n_reduce_to=5)
doc_topic_matrix = model.fit_transform(corpus)

model.print_topics()
```

## API Reference 

::: turftopic.models.cluster.CTop2Vec

