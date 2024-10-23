# Model Overview

In any use case it is important that practicioners understand the implications of their choices.
This page is dedicated to giving an overview of the models in the package, so you can find the right one for your particular application.

### What is a topic?

Models in Turftopic provide answers to this question that can at large be assigned into two categories:

1. A topic is a __dimension/factor of semantics__. 
 These models try to find the axes along which most of the variance in semantics can be explained.
 These include S³ and KeyNMF.
 A clear advantage of using these models is that they can capture multiple topics in a document and usually capture nuances in semantics better.
2. A topic is a __cluster of documents__. These models conceptualize a topic as a group of documents that are closely related to each other.
 The advantage of using these models is that they are perhaps more aligned with human intuition about what a "topic" is.
 On the other hand, they can only capture nuances in topical content in documents to a limited extent.
3. A topic is a __probability distribution__ of words. This conception is characteristic of autencoding models.

### Document Representations

All models in Turftopic at some point in the process use contextualized representations from transformers to learn topics.
Documents, however have different representations internally, and this has an effect on how the models behave:

1. In most models the documents are __directly represented by the embeddings__ (S³, Clustering, GMM).
 The advantage of this is that at no point in the process do we loose contextual information.
2. In KeyNMF documents are represented with __keyword importances__. This means that some of the contextual nuances get lost in the process before topic discovery.
 As a result of this, KeyNMF models dimensions of semantics in word content, not the continuous semantic space.
 In practice this rarely presents a challenge, but topics in KeyNMF might be less interesting or novel than in other models, and might resemble classical topic models more.
3. In Autoencoding Models _embeddings are only used in the encoder network_, but the models describe the generative process of __Bag-of-Words representations__.
 This is not ideal, as all too often contextual nuances get lost in the modeling process.

<center>

| Model | Conceptualization | #N Topics | Term Importance | Document Representation | Inference | Multilingual :globe_with_meridians: |
| - | - | - | - | - | - | - |
| [S³](s3.md) | Factor | Manual | Decomposition | Embedding | Inductive | :heavy_check_mark: |
| [KeyNMF](KeyNMF.md) | Factor | Manual | Parameters | Keywords | Inductive | :x:  |
| [GMM](GMM.md) | Mixture Component | Manual | c-TF-IDF | Embedding | Inductive | :heavy_check_mark: |
| [Clustering Models](clustering.md) | Cluster | **Automatic** | c-TF-IDF/ <br> Centroid Proximity | Embedding | Transductive | :heavy_check_mark: |
| [Autoencoding Models](ctm.md) | Probability Distribution | Manual | Parameters | Embedding + <br> BoW | Inductive | :heavy_check_mark:  |

_Comparison of the models on a number of theoretical aspects_

</center>

### Inference

Models in Turftopic use two different types of inference, which has a number of implications.

1. Most models are __inductive__. Meaning that they aim to recover some underlying structure which results in the observed data.
 Inductive models can be used for inference over novel data at any time.
2. Clustering models that use HDBSCAN, DBSCAN or OPTICS are __transductive__. This means that the models have no theory of underlying semantic structures,
 but simply desdcribe the dataset at hand. This has the effect that direct inference on unseen documents is not possible.

### Term Importance

Term importances in different models are calculated differently.

1. Some models (KeyNMF, Autoencoding) __infer__ term importances, as they are model parameters.
2. Other models (GMM, Clustering, $S^3$) use __post-hoc__ measures for determining term importance.

## API Reference

:::turftopic.base.ContextualModel
