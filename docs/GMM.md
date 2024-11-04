# GMM (Gaussian Mixture Model)

GMM is a generative probabilistic model over the contextual embeddings.
The model assumes that contextual embeddings are generated from a mixture of underlying Gaussian components.
These Gaussian components are assumed to be the topics.

<figure>
  <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_gmm_pdf_001.png" width="80%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Components of a Gaussian Mixture Model <br>(figure from scikit-learn documentation)</figcaption>
</figure>

## How does GMM work?

### Generative Modeling

GMM assumes that the embeddings are generated according to the following stochastic process from a number of Gaussian components.
Priors are optionally imposed on the model parameters.
The model is fitted either using expectation maximization or variational inference.

??? info "Click to see formula"
    1. Select global topic weights: $\Theta$
    2. For each component select mean $\mu_z$ and covariance matrix $\Sigma_z$ .
    3. For each document:
        - Draw topic label: $z \sim Categorical(\Theta)$
        - Draw document vector: $\rho \sim \mathcal{N}(\mu_z, \Sigma_z)$


### Calculate Topic Probabilities

After the model is fitted, soft topic labels are inferred for each document.
A document-topic-matrix ($T$) is built from the likelihoods of each component given the document encodings.

??? info "Click to see formula"
    - For document $i$ and topic $z$ the matrix entry will be: $T_{iz} = p(\rho_i|\mu_z, \Sigma_z)$

### Soft c-TF-IDF

Term importances for the discovered Gaussian components are estimated post-hoc using a technique called __Soft c-TF-IDF__,
an extension of __c-TF-IDF__, that can be used with continuous labels.

??? info "Click to see formula"

    Let $X$ be the document term matrix where each element ($X_{ij}$) corresponds with the number of times word $j$ occurs in a document $i$.
    Soft Class-based tf-idf scores for terms in a topic are then calculated in the following manner:

    - Estimate weight of term $j$ for topic $z$: <br>
    $tf_{zj} = \frac{t_{zj}}{w_z}$, where 
    $t_{zj} = \sum_i T_{iz} \cdot X_{ij}$ and 
    $w_{z}= \sum_i(|T_{iz}| \cdot \sum_j X_{ij})$ <br>
    - Estimate inverse document/topic frequency for term $j$:  
    $idf_j = log(\frac{N}{\sum_z |t_{zj}|})$, where
    $N$ is the total number of documents.
    - Calculate importance of term $j$ for topic $z$:   
    $Soft-c-TF-IDF{zj} = tf_{zj} \cdot idf_j$

### Dynamic Modeling

GMM is also capable of dynamic topic modeling. This happens by fitting one underlying mixture model over the entire corpus, as we expect that there is only one semantic model generating the documents.
To gain temporal representations for topics, the corpus is divided into equal, or arbitrarily chosen time slices, and then term importances are estimated using Soft-c-TF-IDF for each of the time slices separately.

### Similarities with Clustering Models

Gaussian Mixtures can in some sense be considered a fuzzy clustering model.

Since we assume the existence of a ground truth label for each document, the model technically cannot capture multiple topics in a document,
only uncertainty around the topic label.

This makes GMM better at accounting for documents which are the intersection of two or more semantically close topics.

Another important distinction is that clustering topic models are typically transductive, while GMM is inductive.
This means that in the case of GMM we are inferring some underlying semantic structure, from which the different documents are generated,
instead of just describing the corpus at hand.
In practical terms this means that GMM can, by default infer topic labels for documents, while (some) clustering models cannot.

## Performance Tips

GMM can be a bit tedious to run at scale. This is due to the fact, that the dimensionality of parameter space increases drastically with the number of mixture components, and with embedding dimensionality.
To counteract this issue, you can use dimensionality reduction. We recommend that you use PCA, as it is a linear and interpretable method, and it can function efficiently at scale.

> Through experimentation on the 20Newsgroups dataset I found that with 20 mixture components and embeddings from the `all-MiniLM-L6-v2` embedding model
 reducing the dimensionality of the embeddings to 20 with PCA resulted in no performance decrease, but ran multiple times faster.
 Needless to say this difference increases with the number of topics, embedding and corpus size.

```python
from turftopic import GMM
from sklearn.decomposition import PCA

model = GMM(20, dimensionality_reduction=PCA(20))

# for very large corpora you can also use Incremental PCA with minibatches

from sklearn.decomposition import IncrementalPCA

model = GMM(20, dimensionality_reduction=IncrementalPCA(20))
```

## API Reference

::: turftopic.models.gmm.GMM
