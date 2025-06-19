from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as spr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics.pairwise import cosine_similarity


def cluster_centroid_distance(
    cluster_centroids: np.ndarray,
    vocab_embeddings: np.ndarray,
) -> np.ndarray:
    """Computes feature importances based on distances between
    topic vectors (cluster centroids) and term embeddings

    Parameters
    ----------
    cluster_centroids: np.ndarray
        Coordinates of cluster centroids of shape (n_topics, embedding_size)
    vocab_embeddings: np.ndarray
        Term embeddings of shape (vocab_size, embedding_size)

    Returns
    -------
    ndarray of shape (n_topics, vocab_size)
        Term importance matrix.
    """
    n_components = cluster_centroids.shape[0]
    n_vocab = vocab_embeddings.shape[0]
    components = np.full((n_components, n_vocab), np.nan)
    valid_centroids = np.all(np.isfinite(cluster_centroids), axis=1)
    similarities = cosine_similarity(
        cluster_centroids[valid_centroids], vocab_embeddings
    )
    components[valid_centroids, :] = similarities
    return components


def linear_classifier(
    doc_topic_matrix: np.ndarray,
    embeddings: np.ndarray,
    vocab_embeddings: np.ndarray,
) -> np.ndarray:
    """Computes feature importances based on embedding directions
    obtained with a linear classifier.

    Parameters
    ----------
    doc_topic_matrix: np.ndarray
        Document-topic matrix.
    embeddings: np.ndarray
        Document embeddings.
    vocab_embeddings: np.ndarray
        Term embeddings of shape (vocab_size, embedding_size)

    Returns
    -------
    ndarray of shape (n_topics, vocab_size)
        Term importance matrix.
    """
    labels = np.argmax(doc_topic_matrix, axis=1)
    model = LinearDiscriminantAnalysis().fit(embeddings, labels)
    components = cosine_similarity(model.coef_, vocab_embeddings)
    if len(set(labels)) == 2:
        # Binary is a special case
        components = np.concatenate([-components, components], axis=0)
    return components


def fighting_words(
    doc_topic_matrix: np.ndarray,
    doc_term_matrix: spr.csr_matrix,
    prior: float | Literal["corpus"] = "corpus",
) -> np.ndarray:
    """Computes feature importance using the *Fighting Words* algorithm.

    Parameters
    ----------
    doc_topic_matrix: np.ndarray
        Document-topic matrix of shape (n_documents, n_topics)
    doc_term_matrix: np.ndarray
        Document-term matrix of shape (n_documents, vocab_size)
    prior: float or "corpus", default "corpus"
        Dirichlet prior to use. When a float, it indicates the alpha
        parameter of a symmetric Dirichlet, if "corpus",
        word frequencies from the background corpus are used.

    Returns
    -------
    ndarray of shape (n_topics, vocab_size)
        Term importance matrix.
    """
    labels = np.argmax(doc_topic_matrix, axis=1)
    n_topics = doc_topic_matrix.shape[1]
    n_vocab = doc_term_matrix.shape[1]
    components = []
    if prior == "corpus":
        priors = np.ravel(np.asarray(doc_term_matrix.sum(axis=0)))
    else:
        priors = np.full(n_vocab, prior)
    a0 = np.sum(priors)  # prior * n_vocab
    for i_topic in range(n_topics):
        topic_freq = np.ravel(
            np.asarray(doc_term_matrix[labels == i_topic].sum(axis=0))
        )
        rest_freq = np.ravel(
            np.asarray(doc_term_matrix[labels != i_topic].sum(axis=0))
        )
        n1 = np.sum(topic_freq)
        n2 = np.sum(rest_freq)
        topic_logodds = np.log(
            (topic_freq + priors) / (n1 + a0 - topic_freq - priors)
        )
        rest_logodds = np.log(
            (rest_freq + priors) / (n2 + a0 - rest_freq - priors)
        )
        delta = topic_logodds - rest_logodds
        delta_var = 1 / (topic_freq + priors) + 1 / (rest_freq + priors)
        zscore = delta / np.sqrt(delta_var)
        components.append(zscore)
    return np.stack(components)


def soft_ctf_idf(
    doc_topic_matrix: np.ndarray,
    doc_term_matrix: spr.csr_matrix,
    return_idf: bool = False,
) -> np.ndarray:
    """Computes feature importances using Soft C-TF-IDF

    Parameters
    ----------
    doc_topic_matrix: np.ndarray
        Document-topic matrix of shape (n_documents, n_topics)
    doc_term_matrix: np.ndarray
        Document-term matrix of shape (n_documents, vocab_size)

    Returns
    -------
    ndarray of shape (n_topics, vocab_size)
        Term importance matrix.
    """
    eps = np.finfo(float).eps
    term_importance = doc_topic_matrix.T @ doc_term_matrix
    overall_in_topic = np.abs(term_importance).sum(axis=1)
    n_docs = len(doc_topic_matrix)
    tf = (term_importance.T / (overall_in_topic + eps)).T
    idf = np.log(n_docs / (np.abs(term_importance).sum(axis=0) + eps))
    ctf_idf = tf * idf
    idf_diag = spr.diags(
        idf,
        offsets=0,
        shape=(doc_term_matrix.shape[1], doc_term_matrix.shape[1]),
        format="csr",
        dtype=tf.dtype,
    )
    if not return_idf:
        return ctf_idf
    else:
        return ctf_idf, idf_diag


def ctf_idf(
    doc_topic_matrix: np.ndarray,
    doc_term_matrix: spr.csr_matrix,
    return_idf: bool = False,
) -> np.ndarray:
    """Computes feature importances using standard C-TF-IDF

    Parameters
    ----------
    doc_topic_matrix: np.ndarray
        Document-topic matrix of shape (n_documents, n_topics)
    doc_term_matrix: np.ndarray
        Document-term matrix of shape (n_documents, vocab_size)

    Returns
    -------
    ndarray of shape (n_topics, vocab_size)
        Term importance matrix.
    """
    labels = np.argmax(doc_topic_matrix, axis=1)
    n_topics = doc_topic_matrix.shape[1]
    components = []
    overall_freq = np.ravel(np.asarray(doc_term_matrix.sum(axis=0)))
    average = overall_freq.sum() / n_topics
    overall_freq[overall_freq == 0] = np.finfo(float).eps
    for i_topic in range(n_topics):
        freq = np.ravel(
            np.asarray(doc_term_matrix[labels == i_topic].sum(axis=0))
        )
        component = freq * np.log(1 + average / overall_freq)
        components.append(component)
    idf = np.log((average / overall_freq) + 1)
    idf_diag = spr.diags(
        idf,
        offsets=0,
        shape=(doc_term_matrix.shape[1], doc_term_matrix.shape[1]),
        format="csr",
        dtype=doc_term_matrix.dtype,
    )
    if not return_idf:
        return np.stack(components)
    else:
        return np.stack(components), idf_diag


def bayes_rule(
    doc_topic_matrix: np.ndarray, doc_term_matrix: spr.csr_matrix
) -> np.ndarray:
    """Computes feature importance based on Bayes' rule.
    The importance of a word for a topic is the probability of the topic conditional on the word.

    $$p(t|w) = \\frac{p(w|t) * p(t)}{p(w)}$$

    Parameters
    ----------
    doc_topic_matrix: np.ndarray
        Document-topic matrix of shape (n_documents, n_topics)
    doc_term_matrix: np.ndarray
        Document-term matrix of shape (n_documents, vocab_size)

    Returns
    -------
    ndarray of shape (n_topics, vocab_size)
        Term importance matrix.
    """
    eps = np.finfo(float).eps
    p_w = np.squeeze(np.asarray(doc_term_matrix.sum(axis=0)))
    p_w = p_w / p_w.sum()
    p_w[p_w <= 0] = eps
    p_t = doc_topic_matrix.sum(axis=0)
    p_t = p_t / p_t.sum()
    term_importance = doc_topic_matrix.T @ doc_term_matrix
    overall_in_topic = np.abs(term_importance).sum(axis=1)
    overall_in_topic[overall_in_topic <= 0] = eps
    p_wt = (term_importance.T / (overall_in_topic)).T
    p_wt /= p_wt.sum(axis=1)[:, None]
    p_tw = (p_wt.T * p_t).T / p_w
    p_tw /= np.nansum(p_tw, axis=0)
    return p_tw
