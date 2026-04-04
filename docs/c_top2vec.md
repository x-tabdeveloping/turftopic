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

## Citation

Please cite Angelov and Inkpen (2024) and Turftopic when using C-Top2Vec in publications:

```bibtex
@article{
  Kardos2025,
  title = {Turftopic: Topic Modelling with Contextual Representations from Sentence Transformers},
  doi = {10.21105/joss.08183},
  url = {https://doi.org/10.21105/joss.08183},
  year = {2025},
  publisher = {The Open Journal},
  volume = {10},
  number = {111},
  pages = {8183},
  author = {Kardos, Márton and Enevoldsen, Kenneth C. and Kostkan, Jan and Kristensen-McLachlan, Ross Deans and Rocca, Roberta},
  journal = {Journal of Open Source Software} 
}

@inproceedings{angelov-inkpen-2024-topic,
    title = "Topic Modeling: Contextual Token Embeddings Are All You Need",
    author = "Angelov, Dimo  and
      Inkpen, Diana",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.790/",
    doi = "10.18653/v1/2024.findings-emnlp.790",
    pages = "13528--13539",
    abstract = "The goal of topic modeling is to find meaningful topics that capture the information present in a collection of documents. The main challenges of topic modeling are finding the optimal number of topics, labeling the topics, segmenting documents by topic, and evaluating topic model performance. Current neural approaches have tackled some of these problems but none have been able to solve all of them. We introduce a novel topic modeling approach, Contextual-Top2Vec, which uses document contextual token embeddings, it creates hierarchical topics, finds topic spans within documents and labels topics with phrases rather than just words. We propose the use of BERTScore to evaluate topic coherence and to evaluate how informative topics are of the underlying documents. Our model outperforms the current state-of-the-art models on a comprehensive set of topic model evaluation metrics."
}

```



## API Reference 

::: turftopic.models.cluster.CTop2Vec

