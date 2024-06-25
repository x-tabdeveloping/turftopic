# Model Overview

In any use case it is important that practicioners understand the implications of their choices.
This page is dedicated to giving an overview of the models in the package, so you can find the right one for your particular application.

## Theory

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


## Performance

Here's a table with the models' performance on a number of quantitative metrics with the `all-MiniLM-L6-v2` embedding model.
Results were obtained with the `topic-benchmark` package.

<center>

<table>
  <tr>
    <th></th>
    <th colspan="4"><strong>20 Newsgroups Raw</strong></th>
    <th colspan="4"><strong>BBC News</strong></th>
    <th colspan="4"><strong>ArXiv ML Papers</strong></th>
  </tr>
  <tr>
    <th></th>
    <th>C<sub>NPMI</sub></th>
    <th>Diversity</th>
    <th>WEC<sub>in</sub></th>
    <th>WEC<sub>ex</sub></th>
    <th>C<sub>NPMI</sub></th>
    <th>Diversity</th>
    <th>WEC<sub>in</sub></th>
    <th>WEC<sub>ex</sub></th>
    <th>C<sub>NPMI</sub></th>
    <th>Diversity</th>
    <th>WEC<sub>in</sub></th>
    <th>WEC<sub>ex</sub></th>
  </tr>
  <tr>
    <td><strong>KeyNMF</strong></td>
    <td><strong>0.148</strong></td>
    <td>0.898</td>
    <td><strong>0.531</strong></td>
    <td>0.273</td>
    <td><strong>0.073</strong></td>
    <td><u>0.907</u></td>
    <td><u>0.925</u></td>
    <td><strong>0.302</strong></td>
    <td><u>-0.010</u></td>
    <td>0.756</td>
    <td>0.821</td>
    <td>0.172</td>
  </tr>
  <tr>
    <td><strong>S³</strong></td>
    <td>-0.211</td>
    <td><strong>0.990</strong></td>
    <td>0.449</td>
    <td><u>0.278</u></td>
    <td>-0.292</td>
    <td><strong>0.923</strong></td>
    <td>0.923</td>
    <td>0.237</td>
    <td>-0.320</td>
    <td><strong>0.943</strong></td>
    <td><strong>0.907</strong></td>
    <td><strong>0.205</strong></td>
  </tr>
  <tr>
    <td><strong>Top2Vec (Clustering)</strong></td>
    <td>-0.137</td>
    <td><u>0.949</u></td>
    <td><u>0.486</u></td>
    <td><strong>0.373</strong></td>
    <td>-0.296</td>
    <td>0.780</td>
    <td><strong>0.931</strong></td>
    <td><u>0.264</u></td>
    <td>-0.266</td>
    <td>0.466</td>
    <td><u>0.844</u></td>
    <td>0.166</td>
  </tr>
  <tr>
    <td><strong>BERTopic (Clustering)</strong></td>
    <td>0.041</td>
    <td>0.532</td>
    <td>0.416</td>
    <td>0.196</td>
    <td>-0.010</td>
    <td>0.500</td>
    <td>0.609</td>
    <td>0.256</td>
    <td>-0.010</td>
    <td>0.354</td>
    <td>0.571</td>
    <td>0.189</td>
  </tr>
  <tr>
    <td><strong>CombinedTM (Autoencoding)</strong></td>
    <td>-0.038</td>
    <td>0.883</td>
    <td>0.401</td>
    <td>0.180</td>
    <td>-0.028</td>
    <td>0.905</td>
    <td>0.859</td>
    <td>0.161</td>
    <td>-0.058</td>
    <td><u>0.808</u></td>
    <td>0.744</td>
    <td>0.132</td>
  </tr>
  <tr>
    <td><strong>ZeroShotTM (Autoencoding)</strong></td>
    <td>-0.016</td>
    <td>0.890</td>
    <td>0.446</td>
    <td>0.183</td>
    <td>-0.018</td>
    <td>0.822</td>
    <td>0.828</td>
    <td>0.174</td>
    <td>-0.062</td>
    <td>0.767</td>
    <td>0.754</td>
    <td>0.130</td>
  </tr>
</table>

_Model Comparison on 3 Corpora: Best bold, second best underlined_
</center>

### 1. When in doubt **use KeyNMF**.

When you can't make an informed decision about which model is optimal for your use case, or you just want to get your hands dirty with topic modeling,
KeyNMF is by far the best option.
It is very stable, gives high quality topics, and is incredibly robust to noise.
It is also the closest to classical topic models and thus conforms to your intuition about topic modeling.

Another advantage is that KeyNMF is the most scalable and fail-safe option, meaning that you can use it on enormous corpora.

### 2. Short Texts - **use Clustering or GMM**

On tweets and short texts in general, making the assumption that a document only contains one topic is very reasonable.
Clustering models and GMM are very good in this context and should be preferred over other options.

### 3. Want to understand variation? **use $S^3$**

$S^3$ is by far the best model to explain variations in semantics.
If you are looking for a model that can help you establish a theory of semantics in a corpus, $S^3$ is an excellent choice.

### 4. Avoid using Autoencoding Models.

In my anecdotal experience and all experiments I've done with topic models, Autoencoding Models were consistently outclassed by all else,
and their behaviour is also incredbly opaque.
Convergence issues or overlapping topics are a common occurrence. And as such, unless you have reasons to do so I would recommend that your first choice is another model on the list.

## API Reference

:::turftopic.base.ContextualModel
