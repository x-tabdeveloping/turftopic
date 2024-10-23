# Clustering Topic Models

Clustering topic models conceptualize topic modeling as a clustering task.
Essentially a topic for these models is a tightly packed group of documents in semantic space.

The first contextually sensitive clustering topic model was introduced with Top2Vec, and BERTopic has also iterated on this idea.

Turftopic contains flexible implementations of these models where you have control over each of the steps in the process,
while sticking to a minimal amount of extra dependencies.
While the models themselves can be equivalent to BERTopic and Top2Vec implementations, Turftopic might not offer some of the implementation-specific features,
that the other libraries boast.

## How do clustering models work?

### Dimensionality Reduction

```python
from sklearn.manifold import TSNE
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel(clustering=TSNE())
```

It is common practice to reduce the dimensionality of the embeddings before clustering them.
This is to avoid the curse of dimensionality, an issue, which many clustering models are affected by.
Dimensionality reduction by default is done with scikit-learn's **TSNE** implementation in Turftopic,
but users are free to specify the model that will be used for dimensionality reduction.

??? note "What reduction model should I choose?"
    Our knowledge about the impacts of choice of dimensionality reduction is limited, and has not yet been explored in the literature.
    Top2Vec and BERTopic both use UMAP, which has a number of desirable properties over alternatives (arranging data points into cluster-like structures, better preservation of global structure than TSNE, speed).

### Clustering

```python
from sklearn.cluster import OPTICS
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel(clustering=OPTICS())
```

After reducing the dimensionality of the embeddings, they are clustered with a clustering model.
As HDBSCAN  has only been part of scikit-learn since version 1.3.0, Turftopic uses **OPTICS** as its default.

??? note "What clustering model should I choose?"
    Some clustering models are capable of discovering the number of clusters in the data (HDBSCAN, DBSCAN, OPTICS, etc.).
    Practice suggests, however, that in large corpora, this frequently results in a very large number of topics, which is impractical for interpretation.
    Models' hyperparameters can be adjusted to account for this behaviour, but the impact of choice of hyperparameters on topic quality is more or less unknown.
    You can also use models that have predefined numbers of clusters, these, however, typically produce lower topic quality (e.g. KMeans)

### Term importance

Clustering topic models rely on post-hoc term importance estimation.
Multiple methods can be used for this in Turftopic.

!!! failure inline end "Weaknesses"
    - Topics can be too specific => low within-topic coverage
    - Assumes spherical clusters => could give incorrect results

!!! success inline end "Strengths"
    - Clean topics
    - Highly specific topics

#### Proximity to Cluster Centroids


The solution introduced in Top2Vec (Angelov, 2020) is that of estimating terms' importances for a given topic from their
embeddings' cosine similarity to the centroid of the embeddings in a cluster.

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel(feature_importance="centroid")
```


!!! failure inline end "Weaknesses"
    - Topics can be contaminated with stop words
    - Lower topic quality

!!! success inline end "Strengths"
    - Theoretically correct
    - More within-topic coverage

#### c-TF-IDF


c-TF-IDF (Grootendorst, 2022) is a weighting scheme based on the number of occurrences of terms in each cluster.
Terms which frequently occur in other clusters are inversely weighted so that words, which are specific to a topic gain larger importance.

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel(feature_importance="soft-c-tf-idf")
# or
model = ClusteringTopicModel(feature_importance="c-tf-idf")
```


By default, Turftopic uses a modified version of c-TF-IDF, called Soft-c-TF-IDF.

??? info "Click to see formula"
    - Let $X$ be the document term matrix where each element ($X_{ij}$) corresponds with the number of times word $j$ occurs in a document $i$.
    - Estimate weight of term $j$ for topic $z$: <br>
    $tf_{zj} = \frac{t_{zj}}{w_z}$, where 
    $t_{zj} = \sum_{i \in z} X_{ij}$ is the number of occurrences of a word in a topic and 
    $w_{z}= \sum_{j} t_{zj}$ is all words in the topic <br>
    - Estimate inverse document/topic frequency for term $j$:  
    $idf_j = log(\frac{N}{\sum_z |t_{zj}|})$, where
    $N$ is the total number of documents.
    - Calculate importance of term $j$ for topic $z$:   
    $Soft-c-TF-IDF{zj} = tf_{zj} \cdot idf_j$

You can also use the original c-TF-IDF formula, if you intend to replicate the behaviour of BERTopic exactly. The two formulas tend to give similar results, though the implications of choosing one over the other has not been thoroughly evaluated.

??? info "Click to see formula"
    - Let $X$ be the document term matrix where each element ($X_{ij}$) corresponds with the number of times word $j$ occurs in a document $i$.
    - $tf_{zj} = \frac{t_{zj}}{w_z}$, where 
    $t_{zj} = \sum_{i \in z} X_{ij}$ is the number of occurrences of a word in a topic and 
    $w_{z}= \sum_{j} t_{zj}$ is all words in the topic <br>
    - Estimate inverse document/topic frequency for term $j$:  
    $idf_j = log(1 + \frac{A}{\sum_z |t_{zj}|})$, where
    $A = \frac{\sum_z \sum_j t_{zj}}{Z}$ is the average number of words per topic, and $Z$ is the number of topics.
    - Calculate importance of term $j$ for topic $z$:   
    $c-TF-IDF{zj} = tf_{zj} \cdot idf_j$

#### Recalculating Term Importance

You can also choose to recalculate term importances with a different method after fitting the model:

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel().fit(corpus)
model.estimate_components(feature_importance="centroid")
model.estimate_components(feature_importance="soft-c-tf-idf")
```

### Hierarchical Topic Merging

A weakness of clustering approaches based on density-based clustering methods, is that all too frequently they find a very large number of topics.
To limit the number of topics in a topic model you can use hierarchical topic merging.

#### Merge Smallest
The approach used in the Top2Vec package is to always merge the smallest topic into the one closest to it (except the outlier-cluster) until the number of topics is down to the desired amount.

You can achieve this behaviour like so:

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel(n_reduce_to=10, reduction_method="smallest")
```

#### Agglomerative Clustering

In BERTopic topics are merged based on agglomerative clustering using average linkage, and then term importances are reestimated.
You can do this in Turftopic as well:

```python
model = ClusteringTopicModel(n_reduce_to=10, reduction_method="agglomerative")
```

You can also merge topics after having run the models using the `reduce_topics()` method.

```python
model = ClusteringTopicModel().fit(corpus)
model.reduce_topics(n_reduce_to=20, reduction_method="smallest")
```

To reset topics to the original clustering, use the `reset_topics()` method:

```python
model.reset_topics()
```

### Manual Topic Merging

You can also manually merge topics using the `join_topics()` method.

```python
model = ClusteringTopicModel()
model.fit(texts, embeddings=embeddings)
# This joins topics 0, 1, 2 to be cluster 0
model.join_topics([0, 1, 2])
```

### How do I use BERTopic and Top2Vec in Turftopic?

You can create BERTopic and Top2Vec models in Turftopic by modifying all model parameters and hyperparameters to match the defaults in those other packages.

You will need UMAP and scikit-learn>=1.3.0 to be able to use HDBSCAN and UMAP:
```bash
pip install umap-learn scikit-learn>=1.3.0
```

#### BERTopic

You will need to set the clustering model to HDBSCAN and dimensionality reduction to UMAP.
BERTopic also uses the original c-tf-idf formula and agglomerative topic joining.

??? info "Show code"

    ```python
    from turftopic import ClusteringTopicModel
    from sklearn.cluster import HDBSCAN
    import umap

    berttopic = ClusteringTopicModel(
        dimensionality_reduction=umap.UMAP(
            n_neighbors=10,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
        ),
        clustering=HDBSCAN(
            min_cluster_size=15,
            metric="euclidean",
            cluster_selection_method="eom",
        ),
        feature_importance="c-tf-idf",
        reduction_method="agglomerative"
    )
    ```

#### Top2Vec

You will need to set the clustering model to HDBSCAN and dimensionality reduction to UMAP.
Top2Vec uses `centroid` feature importance and `smallest` topic merging method.

??? info "Show code"
    ```python
    top2vec = ClusteringTopicModel(
        dimensionality_reduction=umap.UMAP(
            n_neighbors=15,
            n_components=5,
            metric="cosine"
        ),
        clustering=HDBSCAN(
            min_cluster_size=15,
            metric="euclidean",
            cluster_selection_method="eom",
        ),
        feature_importance="centroid",
        reduction_method="smallest"
    )
    ```

Theoretically the model descriptions above should result in the same behaviour as the other two packages, but there might be minor changes in implementation.
We do not intend to keep up with changes in Top2Vec's and BERTopic's internal implementation details indefinitely.

### Dynamic Modeling

Clustering models are also capable of dynamic topic modeling. This happens by fitting a clustering model over the entire corpus, as we expect that there is only one semantic model generating the documents.

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel().fit_dynamic(corpus, timestamps=ts, bins=10)
model.print_topics_over_time()
```

## API Reference

::: turftopic.models.cluster.ClusteringTopicModel
