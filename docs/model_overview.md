# Model Overview

In any use case it is important that practicioners understand the implications of their choices.
This page is dedicated to giving an overview of the models in the package, so you can find the right one for your particular application.


## What is a topic?

Models in Turftopic provide answers to this question that can at large be assigned into two categories:

1. A topic is a __dimension/factor of semantics__. 
 These models try to find the axes along which most of the variance in semantics can be explained.
 These include S³, KeyNMF and Autoencoding Models.
 A clear advantage of using these models is that they can capture multiple topics in a document and usually capture nuances in semantics better.
2. A topic is a __cluster of documents__. These models conceptualize a topic as a group of documents that are closely related to each other.
 The advantage of using these models is that they are perhaps more aligned with human intuition about what a "topic" is.
 On the other hand, they can only capture nuances in topical content in documents to a limited extent.

## Document Representations

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

##### Theoretical Comparison

| Model | Conceptualization | #N Topics | Term Importance | Document Representation | Inference | Multilingual :globe_with_meridians: |
| - | - | - | - | - | - | - |
| S³ | Dimension/Factor | Manual | Decomposition | Embedding | Inductive | :heavy_check_mark: |
| KeyNMF | Dimension/Factor | Manual | Parameters | Keywords | Inductive | :x:  |
| GMM | Cluster/Mixture Component | Manual | c-TF-IDF | Embedding | Inductive | :heavy_check_mark: |
| Clustering Models | Cluster/Mixture Component | **Automatic** | c-TF-IDF/ <br> Centroid Proximity | Embedding | Transductive | :heavy_check_mark: |
| Autoencoding Models | Dimension/Factor | Manual | Parameters | Embedding + <br> BoW | Inductive | :heavy_check_mark:  |

</center>

## Inference

Models in Turftopic use two different types of inference, which has a number of implications.

1. Most models are __inductive__. Meaning that they aim to recover some underlying structure which results in the observed data.
 Inductive models can be used for inference over novel data at any time.
2. Clustering models that use HDBSCAN, DBSCAN or OPTICS are __transductive__. This means that the models have no theory of underlying semantic structures,
 but simply desdcribe the dataset at hand. This has the effect that direct inference on unseen documents is not possible.

## Term Importance

Term importances in different models are calculated differently.

1. Some models (KeyNMF, Autoencoding) have built-in term importance estimation, as term importances are literally in the models' parameters.
 This means that term importances are __inferential__. Meaning that they make a claim about underlying semantical structures.
 A potential drawback is, that if the vocabulary is very large, the models can be impacted by the curse of dimensionality, resulting in poor convergence or slow inference.
2. Other models (GMM, Clustering) use post-hoc measures for determining term importance.
 In other words term importances are __descriptive__. Inference of term importance is much more efficient for these methods, but
 they make no claims about the underlying semantics that result in these term importances.
3. S³ __decomposes the vocabulary__ with a fitted model. The result of this is that the model can generalize over all sorts of corpora,
 and can be described in different ways in different vocabularies. This is somewhere in between inferential and descriptive methods.


## Which model should I choose?

The model that you should be using for any particular application will of course be influenced by a number of factors, that you should consider.
The tables on this page give you a general overview of a handful of practical aspects of the models.

Here is an opinionated guide for common use cases:

### 1. When in doubt **use KeyNMF**.

When you can't make an informed decision about which model is optimal for your use case, or you just want to get your hands dirty with topic modeling,
KeyNMF is the best option.
It is very stable, gives high quality topics, and is incredibly robust to noise.
It is also the closest to classical topic models and thus conforms to your intuition about topic modeling.

Another advantage is that KeyNMF is the most scalable and fail-safe option, meaning that you can use it on enormous corpora.

### 2. Short Texts - **use Clustering or GMM**

On tweets and short texts in general, making the assumption that a document only contains one topic is very reasonable.
Clustering models and GMM are very good in this context and should be preferred over other options.

### 3. Want to understand variation? **use S³**

S³ is by far the best model to explain variations in semantics.
If you are looking for a model that can help you establish a theory of semantics in a corpus, S³ is an excellent choice.

### 4. Avoid using Autoencoding Models.

In my anecdotal experience and all experiments I've done with topic models, Autoencoding Models were consistently outclassed by all else,
and their behaviour is also incredbly opaque.
Convergence issues or overlapping topics are a common occurrence. And as such, unless you have reasons to do so I would recommend that your first choice is another model on the list.

<center>

##### Practical Comparison

| Model | Scalability | Ideal Document Length | Speed | Stability | Robustness to Noise | Embedding Size |
| - | - | - | - | - | - | - |
| S³ | Moderate | **Short, Medium, Long** | **Fast** | Moderate | Good | Any |
| KeyNMF | **Very High** | Medium, Long | Moderate | **Stable** | **Very Good** | Any |
| GMM | Moderate | Short, Medium | Moderate | Moderate | Good | Limited |
| Clustering Models | Low | Short, Medium | Moderate | Volatile | **Very Good**(_centroid_) <br>  Moderate(_c-TF-IDF_) | Any |
| Autoencoding Models | Low | Hard to Tell | Slow | Volatile | Poor | Limited |

</center>
