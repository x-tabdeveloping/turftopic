# GMM

GMM is a generative probabilistic model over the contextual embeddings.
The model assumes that contextual embeddings are generated from a mixture of underlying Gaussian components.
These Gaussian components are assumed to be the topics.

<figure>
  <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_gmm_pdf_001.png" width="80%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Components of a Gaussian Mixture Model <br>(figure from scikit-learn documentation)</figcaption>
</figure>

## The Model

### 1. Generative Modeling

GMM assumes that the embeddings are generated according to the following stochastic process:

1. Select global topic weights: $\Theta$
2. For each component select mean $\mu_z$ and covariance matrix $\Sigma_z$ .
3. For each document:
    - Draw topic label: $z \sim Categorical(\Theta)$
    - Draw document vector: $\rho \sim \mathcal{N}(\mu_z, \Sigma_z)$

Priors are optionally imposed on the model parameters.
The model is fitted either using expectation maximization or variational inference.

### 2. Topic Inference over Documents

After the model is fitted, soft topic labels are inferred for each document.
A document-topic-matrix ($T$) is built from the likelihoods of each component given the document encodings.

Or in other words for document $i$ and topic $z$ the matrix entry will be: $T_{iz} = p(\rho_i|\mu_z, \Sigma_z)$

### 3. Soft c-TF-IDF

Term importances for the discovered Gaussian components are estimated post-hoc using a technique called __Soft c-TF-IDF__,
an extension of __c-TF-IDF__, that can be used with continuous labels.

Let $X$ be the document term matrix where each element ($X_{ij}$) corresponds with the number of times word $j$ occurs in a document $i$.
Soft Class-based tf-idf scores for terms in a topic are then calculated in the following manner:

- Estimate weight of term $j$ for topic $z$: <br>
$tf_{zj} = \frac{t_{zj}}{w_z}$, where 
$t_{zj} = \sum_k T_{iz} \cdot X_{ij}$ and 
$w_{z}= \sum_i(|T_{iz}| \cdot \sum_j X_{ij})$ <br>
- Estimate inverse document/topic frequency for term $j$:  
$idf_j = log(\frac{N}{\sum_z |t_{zj}|})$, where
$N$ is the total number of documents.
- Calculate importance of term $j$ for topic $z$:   
$Soft-c-TF-IDF{zj} = tf_{zj} \cdot idf_j$

### Similarities with Clustering Models

Gaussian Mixtures can in some sense be considered a fuzzy clustering model.

Since we assume the existence of a ground truth label for each document, the model technically cannot capture multiple topics in a document,
only uncertainty around the topic label.

This makes GMM better at accounting for documents which are the intersection of two or more semantically close topics.

Another important distinction is that clustering topic models are typically transductive, while GMM is inductive.
This means that in the case of GMM we are inferring some underlying semantic structure, from which the different documents are generated,
instead of just describing the corpus at hand.
In practical terms this means that GMM can, by default infer topic labels for documents, while (some) clustering models cannot.

## Considerations

### Strengths

 - Efficiency, Stability: GMM relies on a rock solid implementation in scikit-learn, you can rest assured that the model will be fast and reliable.
 - High Quality: GMM typically finds very high quality and easily interpretable topics.
 - Coverage of Ingroup Variance: The model is very efficient at describing the extracted topics in all their detail.
 This means that the topic descriptions will typically cover most of the documents generated from the topic fairly well.
 - Uncertainty: GMM is capable of expressing and modeling uncertainty around topic labels for documents.

### Weaknesses

 - Curse of Dimensionality: The dimensionality of embeddings can vary wildly from model to model. High-dimensional embeddings might decrease the efficiency and performance of GMM, as it is sensitive to the curse of dimensionality. Dimensionality reduction can help mitigate these issues.
 - Assumption of Gaussianity: The model assumes that topics are Gaussian components, it might very well be that this is not the case.
 Fortunately enough this rarely effects real-world perceived performance of models, and typically does not present an issue in practical settings.
 - Scalability: While the model is scalable to a certain extent, it is not nearly as scalable as some of the other options. If you experience issues with computational efficiency or convergence, try another model.
