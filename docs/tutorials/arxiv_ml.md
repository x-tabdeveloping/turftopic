# Building a Taxonomy of Machine Learning Papers

Topic models can often be used to gain an overall understanding of what kinds of topics are being discussed in a corpus.
If one wanted to build a taxonomy of the field of machine learning, a good starting point would be to investigate what sorts of machine learning articles have been put on ArXiv, the largest preprint server.

In this tutorial, I will demonstrate how you can use Turftopic to discover themes in machine learning abstracts using a clustering topic model, we will look at:

- Building and training a clustering topic model
- Interpreting the model's output, and
- Topic-based filtering and retrieval 

## Installation

For this investigation we will use two additional libraries. `datasets` to fetch the dataset from HF Hub, as well as `plotly` for creating interactive plots of our results.
We also need to install optional dependencies to be able to use umap-based clustering models, and then create a map using datamapplot.

```bash
pip install datasets plotly turftopic[umap-learn, datamapplot]
```

## Data Preparation

We will download the dataset from HuggingFace Hub, using `datasets`.

!!! note
    In this example, we will downsample to a subset of the data (10000 examples).
    We do this to make the tutorial run smoothly. However, often downsampling can be 
    a great strategy for speeding up topic modelling.
```python
from datasets import load_dataset

ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
# Subsampling dataset
ds = ds.train_test_split(seed=42, test_size=10_000)["test"]
abstracts = ds["abstract"]
```

Turftopic uses contextual embeddings of documents in order to understand what is in the corpus.
We will use the small and fast `all-MiniLM-L6-v2` sentence transformer to produce embeddings of our data.

!!! tip 
    Sometimes you might want to reuse embeddings in different topic models, or save them to disk.
    It is thus recommended, but not necessary to precompute them before running a topic model.

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(abstracts, show_progress_bar=True)
```

## Model Training

For building a hierarchy, we will assume that a paper is centered around one central theme, and we would like to categorize our articles into distinct groups.
[Clustering topic models](../clustering.md), such as BERTopic and Top2Vec are well suited for this task, as they assume that a document belongs to a single cluster.
They can also be used in settings, where we expect there to be a hierarchy of topics.
This is certainly the case for machine learning, where, for instance, *auto-encoders* could be a sub-topic of *unsupervised learning*.

In this example I am going to use the Top2Vec topic model, which discovers topic using UMAP and HDBSCAN, and produces very clean topic descriptions.
Top2Vec learns the number of topics from the data, so we won't have to specify it a-priori.

```python
from turftopic import Top2Vec

model = Top2Vec(encoder=encoder, random_state=42)
topic_data = model.prepare_topic_data(abstracts, embeddings=embeddings)
```

## Interpreting Results

!!! tip
    For a more detailed discussion, see the [Model Interpretation](../model_interpretation.md) page in the documentation.

Let's print the topic in our model in order to see what sorts of topics have been discovered.

```python
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| -1 | softmax, supervised, cnn, cnns, autoencoders, learns, classifiers, imagenet, rnns, learning |
| 0 | quantum, quantization, qubit, quantized, encoders, backpropagation, learns, disentangling, learning, decoders |
| 1 | backpropagation, robustness, lyapunov, robust, adversarial, softmax, controllers, learnable, robustly, controllable |
| 2 | fairness, discrimination, bias, unfairness, biases, discriminating, discriminate, adversarially, classifiers, penalized |
| 3 | investment, learns, learning, reinforcement, learnt, finance, rnns, trading, memorization, bandit |
| | ... |


We can already see some clear themes emerge, for instance we can see that a group of ML papers are dedicated to quantum computing, and there are also some researchers investigating bias and fairness in machine learning approaches.

### Building a Topic Hierarchy


!!! tip
    For a more detailed discussion, see the [Hierarchical Modelling](../hierarchical.md) page in the documentation.

Our model has discovered over 100 topics, which can be a bit hard to interpret.
Luckily, we can reduce the number of top level topics in clustering models by iteratively merging them until we obtain a desired number.
This will also build a cluster hierarchy, which we can then investigate using the model's `hierarchy` property.

```python
model.reduce_topics(n_reduce_to=25)
print(model.hierarchy.cut(3))
```

```
Root: 
├── -1: softmax, supervised, cnn, cnns, autoencoders, learns, classifiers, imagenet, rnns, learning
├── 6: privacy, adversarially, randomization, adversarial, softmax, randomized, distributed, supervised, rnns, regularization
├── 50: gans, generative, gan, adversarial, adversarially, autoencoders, inception, cyclegan, autoencoder, imagenet
├── 105: minimizers, optimizations, minimizes, optimization, minimizer, optimizers, optimizer, minimization, minimize, optimizing
├── 161: learns, reinforcement, learning, learnt, planning, ai, memorization, learnable, rnns, supervised
│   ├── 8: mobilenet, networking, mobilenetv2, scheduling, bandwidth, 5g, resnets, networks, network, congestion
│   └── 129: learns, reinforcement, learning, learnt, planning, ai, memorization, learnable, rnns, supervised
│       ├── 1: backpropagation, robustness, lyapunov, robust, adversarial, softmax, controllers, learnable, robustly, controllable
...
```

We can also look at a particular part of the hierarchy, we might be interested in.
If I would like to gain a more complete picture of what approaches exist for graph learning,
I can look at that part of the hierarchy, specifically:

```python
fig = model.hierarchy[193].plot_tree()
fig.show()
```

<center>
  <iframe src="../images/graph_learning_tree.html", title="Tree plot of hierarchy", style="height:420px;width:820px;padding:0px;border:none;"></iframe>
</center>


### Investigating the Topic Landscape

We can also gain more detailed insights by looking at the document clusters on an interactive map, that way, not all topics have to be seen at once, but one can zoom in to gain deeper insights about a certain area.
We will do this by using `datamapplot`, and make sure that document names can be seen when we hover over their respective datapoints.

This plot will also allow us to see how far or close clusters are to each other, as well as what kinds of paper belong to each cluster.

```python
# We will reset the hierarchy, so that we can see all topics at once.
model.reset_topics()
fig = model.plot_clusters_datamapplot(hover_text=ds["title"])
fig.show()
```

<center>
  <iframe src="../images/arxiv_ml_datamapplot.html", title="Map of ArXiv ML Papers", style="height:800px;width:800px;padding:0px;border:none;"></iframe>
</center>

## Topic-based Retrieval and Filtering

Suppose that I am a cognitive neuroscientist, and would like to incorporate machine learning methods into my work.
I would like to gain an overview of methods, get some paper recommendations, and get an understanding of how prevalent neuroscience is in machine learning literature.

First, we need to know which topics are relevant.
We can do this by estimating topic importance scores for a phrase that captures what we want to find.
Let's use `cognitive neuroscience, imaging` as a search term.

```python
model.print_topic_distribution("cognitive neuroscience imaging")
``` ```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Topic name                                             ┃ Score ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ 48_fmri_neural_neuroimaging_cortex                     │  1.64 │
│ 53_imaging_mri_deconvolution_imagenet                  │  1.47 │
│ 47_neuron_neurons_neural_neuronal                      │  1.43 │
│ 25_electroencephalogram_electroencephalography_eeg_bci │  1.41 │
│ 54_tomography_imaging_imagenet_cnn                     │  1.38 │
│ 49_deconvolution_regularization_denoising_compressive  │  1.34 │
│ 51_denoising_deconvolution_cnn_imagenet                │  1.32 │
│ 58_imagenet_cnn_cnns_inception                         │  1.29 │
│ 59_imagenet_cnn_cnns_supervised                        │  1.18 │
│ 74_cnns_convolutions_cnn_imagenet                      │  1.17 │
└────────────────────────────────────────────────────────┴───────┘
```

All of the topics that show up are related to image processing, but only the first 5 seem to revolve around neuroimaging specifically.

If we want these documents to belong to one cluster, we can join these clusters together in the model manually into the topic with the smallest ID, then give it a descriptive name.

```python
model.join_topics([48, 53, 47, 25, 54], joint_id=25)
model.rename_topics({25: "Neuroimaging"})
```

We can collect documents that belong to this topic:

```python
import numpy as np

is_relevant = model.labels_ == 25

relevant_documents = np.array(abstracts)[is_relevant]
relevant_titles = np.array(ds["title"])[is_relevant]

# We calculate document-topic importance scores to retrieve most relevant documents
doc_topic_importance = model.transform(relevant_documents, embeddings=embeddings[is_relevant])

print(len(relevant_documents))
# 162
```
We have now collected 162 papers that are relevant to our inquiry.
If we wish to get paper recommendations, we can check, which documents rank highest on this topic:

```python
topic_idx = list(model.classes_).index(25)
most_relevant = np.argsort(-doc_topic_importance[:, 25])

for idx in most_relevant[:10]:
    print(relevant_titles[idx])
```

#### Top 10 matches:
- neuro2vec: Masked Fourier Spectrum Prediction for Neurophysiological
  Representation Learning
- Synthetic Epileptic Brain Activities Using Generative Adversarial
  Networks
- Reconstructing ERP Signals Using Generative Adversarial Networks for
  Mobile Brain-Machine Interface
- Deep learning approaches for neural decoding: from CNNs to LSTMs and
  spikes to fMRI
- Real-time EEG-based Emotion Recognition using Discrete Wavelet
  Transforms on Full and Reduced Channel Signals
- Evaluation of Preference of Multimedia Content using Deep Neural
  Networks for Electroencephalography
- A Compact and Interpretable Convolutional Neural Network for
  Cross-Subject Driver Drowsiness Detection from Single-Channel EEG
- Personalized Automatic Sleep Staging with Single-Night Data: a Pilot
  Study with KL-Divergence Regularization
- SeizureNet: Multi-Spectral Deep Feature Learning for Seizure Type
  Classification
- Towards physiology-informed data augmentation for EEG-based BCIs

### Filtering New Documents

Suppose that, in the future, more and more papers get published, but we are only interested in the ones that have a *neuroimaging* theme.
Since our topic model has learned a good representation of what this means in relation to other machine learning papers, we can use the model to filter documents in the future based on their topical content.

This means that we can use topic models to create a classification model without labelling documents or training a classifier.

```python
new_documents = [
    "We utilized fMRI and unsupervised learning to uncover patterns in the development of schizophrenia in adolescents.",
    "Our approach utilizies fluid dynamics for hyperparameter optimization, and achieves state-of-the-art results on multiple neural network architectures."
]

doc_topic_importance = model.transform(new_documents)
topic_labels = [model.topic_names[topic_idx] for topic_idx in np.argmax(doc_topic_importance, axis=1)]
print(topic_labels)
# ['Neuroimaging', '41_autonomous_driving_planning_vehicles']
```


