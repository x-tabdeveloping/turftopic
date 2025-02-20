# Clustering Topic Models

Clustering topic models conceptualize topic modeling as a clustering task.
Essentially a topic for these models is a tightly packed group of documents in semantic space.
The first contextually sensitive clustering topic model was introduced with Top2Vec, and BERTopic has also iterated on this idea.

If you are looking for a probabilistic/soft-clustering model you should also check out [GMM](GMM.md).

<figure>
  <iframe src="../images/cluster_datamapplot.html", title="Cluster visualization", style="height:600px;width:800px;padding:0px;border:none;"></iframe>
  <figcaption> Figure 1: Interactive figure to explore cluster structure in a clustering topic model. </figcaption>
</figure>

## How do clustering models work?

### Step 1: Dimensionality Reduction

It is common practice to reduce the dimensionality of the embeddings before clustering them.
This is to avoid the curse of dimensionality, an issue, which many clustering models are affected by.
Dimensionality reduction by default is done with [**TSNE**](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne) in Turftopic,
but users are free to specify the model that will be used for dimensionality reduction.

!!! quote "Choose a dimensionality reduction method"

    === "TSNE (default)"

        ```python
        from sklearn.manifold import TSNE
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(dimensionality_reduction=TSNE(n_components=2, metric="cosine"))
        ```
        TSNE is a classic method for producing non-linear lower-dimensional representations of high-simensional embeddings.
        TSNE has an inherent clustering property, which helps clustering models find groups of data.
        While it is widely used, it has many well-known issues, such as poor representation of global relations, and artificial clusters.

        !!! tip "Use openTSNE for better performance!"
            By default, a scikit-learn implementation is used, but if you have the [openTSNE](https://github.com/pavlin-policar/openTSNE) package installed on your system, Turftopic will automatically use it.
            You can potentially speed up your clustering topic models by multiple orders of magnitude.
            ```bash
            pip install turftopic[opentsne]
            ```


    === "UMAP (Top2Vec; BERTopic)"

        ```bash
        pip install umap-learn
        ```

        ```python
        from umap import UMAP
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(dimensionality_reduction=UMAP(n_components=2, metric="cosine"))
        ```

        UMAP is universally usable non-linear dimensionality reduction method and is typically the default choice for topic discovery in clustering topic models.
        UMAP is faster than TSNE and is also substantially better at representing global structures in your dataset.

    === "PCA (fast)"

        ```python
        from sklearn.decomposition import PCA
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(dimensionality_reduction=PCA(n_components=2))
        ```

        Principal Component Analysis is one of the most widely used dimensionality reduction techniques in machine learning.
        It is a linear method, that projects embeddings onto the first N principal components by the amount of variance they capture in the data.
        PCA is substantially faster than manifold methods, but is not as good at aiding clustering models as TSNE and UMAP.

### Step 2: Document Clustering

After the dimensionality of document embeddings is reduced, topics are discovered by clustering document-embeddings in this lower dimensional space.
Turftopic is entirely clustering-model agnostic, and as such, any type of model may be used.

!!! quote "Choose a clustering method"

    === "HDBSCAN (default)"

        ```python
        from sklearn.cluster import HDBSCAN
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(clustering=HDBSCAN())
        ```

        HDBSCAN is a density-based clustering method, that can find clusters with varying densities.
        It can find the number of clusters in the data, and can also find outliers.
        While HDBSCAN has many advantageous properties, it can be hard to make an informed choice about its hyperparameters.

    === "KMeans (fast)"

        ```python
        from sklearn.cluster import KMeans
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(clustering=KMeans(n_clusters=10))
        ```

        The KMeans algorithm finds clusters by locating a prespecified number of mean vectors that minimize square distance of embeddings in a cluster to their mean.
        KMeans is a very fast algorithm, but makes very strong assumptions about cluster shapes, can't detect outliers and you have to specify the number of clusters prior to model fitting.

### Step 3: Calculate term importance scores

Clustering topic models rely on post-hoc term importance estimation, meaning that topic descriptions are calculated based on already discovered clusters.
Multiple methods are available in Turftopic for estimating words'/phrases' importance scores for topics.


!!! quote "Choose a term importance estimation method"

    === "c-TF-IDF (BERTopic)"

        ```python
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(feature_importance="soft-c-tf-idf")
        # or
        model = ClusteringTopicModel(feature_importance="c-tf-idf")
        ```
        !!! failure inline end "Weaknesses"
            - Topics can be contaminated with stop words
            - Lower topic quality

        !!! success inline end "Strengths"
            - Theoretically more correct
            - More within-topic coverage
        c-TF-IDF (Grootendorst, 2022) is a weighting scheme based on the number of occurrences of terms in each cluster.
        Terms which frequently occur in other clusters are inversely weighted so that words, which are specific to a topic gain larger importance.
        By default, Turftopic uses a modified version of c-TF-IDF, called Soft-c-TF-IDF, which is more robust to stop-words.

        <br>

        ??? info "Click to see formulas"
            #### Soft-c-TF-IDF
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

            #### c-TF-IDF
            - Let $X$ be the document term matrix where each element ($X_{ij}$) corresponds with the number of times word $j$ occurs in a document $i$.
            - $tf_{zj} = \frac{t_{zj}}{w_z}$, where 
            $t_{zj} = \sum_{i \in z} X_{ij}$ is the number of occurrences of a word in a topic and 
            $w_{z}= \sum_{j} t_{zj}$ is all words in the topic <br>
            - Estimate inverse document/topic frequency for term $j$:  
            $idf_j = log(1 + \frac{A}{\sum_z |t_{zj}|})$, where
            $A = \frac{\sum_z \sum_j t_{zj}}{Z}$ is the average number of words per topic, and $Z$ is the number of topics.
            - Calculate importance of term $j$ for topic $z$:   
            $c-TF-IDF{zj} = tf_{zj} \cdot idf_j$


    === "Centroid Proximity (Top2Vec)"

        ```python
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(feature_importance="centroid")
        ```

        !!! failure inline end "Weaknesses"
            - Low within-topic coverage
            - Assumes spherical clusters

        !!! success inline end "Strengths"
            - Clean topics
            - Highly specific topics

        In Top2Vec (Angelov, 2020) term importance scores are estimated from word embeddings' similarity to centroid vector of clusters.
        This approach typically produces cleaner and more specific topic descriptions, but might not be the optimal choice, since it makes assumptions about cluster shapes, and only describes the centers of clusters accurately.





You can also choose to recalculate term importances with a different method after fitting the model:

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel().fit(corpus)
model.estimate_components(feature_importance="centroid")
model.estimate_components(feature_importance="soft-c-tf-idf")
```

## Hierarchical Topic Merging

A weakness of clustering approaches based on density-based clustering methods, is that all too frequently they find a very large number of topics.
To limit the number of topics in a topic model you can hierarchically merge topics, until you get the desired number.
Turftopic allows you to use a number of popular methods for merging topics in clustering models.

!!! quote "Choose a topic reduction method"

    === "Agglomerative Clustering (BERTopic)"

        ```python
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(n_reduce_to=10, reduction_method="average")
        # or 
        model.reduce_topics(10, reduction_method="single", metric="cosine")
        ```

        Topics discovered by a clustering model can be merged using agglomerative clustering.
        For a detailed discussion of linkage methods and hierarchical clustering, consult [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html).
        All linkage methods compatible with SciPy can be used as topic reduction methods in Turftopic.

    === "Smallest -> Closest (Top2Vec)"

        ```python
        from turftopic import ClusteringTopicModel

        model = ClusteringTopicModel(n_reduce_to=10, reduction_method="smallest")
        # or 
        model.reduce_topics(10, reduction_method="smallest", metric="cosine")
        ```

        The approach used in the Top2Vec package is to always merge the smallest topic into the one closest to it (except the outlier-cluster) until the number of topics is down to the desired amount.
        This approach is remarkably fast, and usually quite effective, since it doesn't require computing full linkages.

As such, all clustering models have a `hierarchy` property, with which you can explore the topic hierarchy discovered by your models. For a detailed discussion of hierarchical modeling, check out the [Hierarchical modeling](hierarchical.md) page.

```python
print(model.hierarchy)
```

<div style="background-color: #F5F5F5; padding: 10px; padding-left: 20px; padding-right: 20px;">
<tt style="font-size: 11pt">
<b>Root</b>: <br>
├── <b style="color:blue">-1</b>: documented, obsolete, et4000, concerns, dubious, embedded, hardware, xfree86, alternative, seeking<br>
├── <b style="color:blue">20</b>: hitter, pitching, batting, hitters, pitchers, fielder, shortstop, inning, baseman, pitcher<br>
├── <b style="color:blue">284</b>: nhl, goaltenders, canucks, sabres, hockey, bruins, puck, oilers, canadiens, flyers<br>
│   ├── <b style="color:magenta">242</b>: sportschannel, espn, nbc, nhl, broadcasts, broadcasting, broadcast, mlb, cbs, cbc<br>
│   │   ├── <b style="color:green">171</b>: stadium, tickets, mlb, ticket, sportschannel, mets, inning, nationals, schedule, cubs<br>
│   │   │   └── ...<br>
│   │   └── <b style="color:green">21</b>: sportschannel, nbc, espn, nhl, broadcasting, broadcasts, broadcast, hockey, cbc, cbs<br>
│   └── <b style="color:magenta">236</b>: nhl, goaltenders, canucks, sabres, puck, oilers, andreychuk, bruins, goaltender, leafs<br>
...
</tt>
</div>

You can also manually merge topics by using the `join_topics()` method of cluster hierarchies.

```python
# Joins topics 0,1 and 2 and creates a merged topics with ID 4
model.hierarchy.join_topics([0, 1, 2], joint_id=4)
```

If you want to reset topics to their original state, you can call `reset_topics()`
```python
model.reset_topics()
```

## Dynamic Topic Modeling

Clustering models are also capable of dynamic topic modeling. This happens by fitting a clustering model over the entire corpus, as we expect that there is only one semantic model generating the documents.

For a detailed discussion, see [Dynamic Models](dynamic.md).
```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel().fit_dynamic(corpus, timestamps=ts, bins=10)
model.print_topics_over_time()
```


## Visualization

You can interactively explore clusters using [datamapplot](https://github.com/TutteInstitute/datamapplot) directly in Turftopic!
You will first have to install `datamapplot` for this to work:

```bash
pip install turftopic[datamapplot]
```

```python
from turftopic import ClusteringTopicModel
from turftopic.namers import OpenAITopicNamer

model = ClusteringTopicModel(feature_importance="centroid").fit(corpus)

namer = OpenAITopicNamer("gpt-4o-mini")
model.rename_topics(namer)

fig = model.plot_clusters_datamapplot()
fig.save("clusters_visualization.html")
fig
```

_See Figure 1_

!!! info
    If you are not running Turftopic from a Jupyter notebook, make sure to call `fig.show()`. This will open up a new browser tab with the interactive figure.



## BERTopic and Top2Vec-like models in Turftopic

You can create BERTopic and Top2Vec models in Turftopic by modifying all model parameters and hyperparameters to match the defaults in those other packages.
You will need UMAP and scikit-learn>=1.3.0 to be able to use HDBSCAN and UMAP:
```bash
pip install umap-learn scikit-learn>=1.3.0
```

!!! quote "Create BERTopic and Top2Vec models"

    === "BERTopic"

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
            reduction_method="average"
            reduction_distance_metric="cosine",
            reduction_topic_representation="component",
        )
        ```

    === "Top2Vec"

        ```python
        from turftopic import ClusteringTopicModel
        from sklearn.cluster import HDBSCAN
        import umap

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
            reduction_distance_metric="cosine",
            reduction_topic_representation="centroid",
        )
        ```

Theoretically the model descriptions above should result in the same behaviour as the other two packages, but there might be minor changes in implementation.
We do not intend to keep up with changes in Top2Vec's and BERTopic's internal implementation details indefinitely.

## API Reference

::: turftopic.models.cluster.ClusteringTopicModel
