# Model Overview

Turftopic contains implementations of a number of contemporary topic models.
Some of these models might be similar to each other in a lot of aspects, but they might be different in others.
It is quite important that you choose the right topic model for your use case.

!!! tip "Looking for Model Performance?"

    If you are interested in seeing how these models perform on a bunch of datasets, and would like to base your model choice on evaluations,
    make sure to check out the [Model Leaderboard](benchmark.md) tab:

    <br>
    <center>
    <img src="../images/leaderboard_screenshot.png" width="700"></img>
    </center>

<div style="text-align: center" markdown>

| Model | Summary | Strengths | Weaknesses |
| -  | - | - | - |
| [Topeax](Topeax.md)  | Density peak detection + Gaussian mixture approximation | Cluster quality, Topic quality, Stability, Automatic n-topics | Underestimates N topics, Slower, No inference for new documents |
| [KeyNMF](KeyNMF.md)  | Keyword importance estimation + matrix factorization | Reliability, Topic quality, Scalability to large corpora and long documents | Automatic topic number detection, Multilingual performance, Sometimes includes stop words |
| [SensTopic(BETA)](SensTopic.md)  | Regularized Semi-nonnegative matrix factorization in embedding space | Very fast, High quality topics and clusters, Can assign multiple soft clusters to documents, GPU support | Automatic n-topics is not too good |
| [GMM](GMM.md)  | Soft clustering with Gaussian Mixtures and soft-cTF-IDF | Reliability, Speed, Cluster quality | Manual n-topics, Lower quality keywords, [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality) |
| [FASTopic](FASTopic.md)  | Neural topic modelling with Dual Semantic-relation Reconstruction | High quality topics and clusters, GPU support | Very slow, Memory hungry, Manual n-topics |
| [$S^3$](s3.md)  | Semantic axis discovery in embedding space | Fastest, Human-readable topics | Axes can be very unintuitive, Manual n-topics |
| [BERTopic and Top2Vec](clustering.md)  | Embed -> Reduce -> Cluster | Flexible, Feature rich | Slow, Unreliable and unstable, Wildly overestimates number of clusters, Low topic and cluster quality |
| [AutoEncodingTopicModel](ctm.md)  | Discover topics by generating BoW with a variational autoencoder | GPU-support | Slow, Sometimes low quality topics |

</div>

Different models will naturally be good at different things, because they conceptualize topics differently for instance:

- `BERTopic`, `Top2Vec`, `GMM` and `Topeax` find **clusters** of documents and treats those as topics
- `KeyNMF`, `SensTopic`, `FASTopic` and `AutoEncodingTopicModel` conceptualize topics as latent nonnegative **factors** that generate the documents.
- `SemanticSignalSeparation`($S^3$) conceptualizes topics as **semantic axes**, along which topics are distributed.

You can find a detailed overview of how each of these models work in their respective tabs.

## Model Features

Some models are also capable of being used in a dynamic context, some can be fitted online, some can detect the number of topics for you and some can detect topic hierarchies. You can find an overview of these features in the table below.


| Model | :1234: Multiple Topics per Document  | :hash: Detecting Number of Topics  | :chart_with_upwards_trend: Dynamic Modeling  | :evergreen_tree: Hierarchical Modeling  | :star: Inference over New Documents  | :globe_with_meridians: Cross-Lingual  | :ocean: Online Fitting  |
| - | - | - | - | - | - | - | - |
| **[KeyNMF](KeyNMF.md)** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: |
| **[SensTopic](SensTopic.md)** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark:  | :x: |
| **[Topeax](Topeax.md)** | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :x:  | :x: |
| **[SemanticSignalSeparation](s3.md)** | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: |
| **[ClusteringTopicModel](clustering.md)** | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: |
| **[GMM](GMM.md)** | :heavy_check_mark: | :x: | :heavy_check_mark: | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: |
| **[AutoEncodingTopicModel](ctm.md)** | :heavy_check_mark: | :x: | :x: | :x: | :heavy_check_mark: | :heavy_check_mark:  | :x: |
| **[FASTopic](fastopic.md)** | :heavy_check_mark: | :x: | :x: | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: |


## Model API Reference

:::turftopic.base.ContextualModel
