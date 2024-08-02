import numpy as np
import scipy.sparse as spr
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def cluster_centroid_distance(
    cluster_centroids: np.ndarray,
    vocab_embeddings: np.ndarray,
    metric="cosine",
) -> np.ndarray:
    """Computes feature importances based on distances between
    topic vectors (cluster centroids) and term embeddings

    Parameters
    ----------
    cluster_centroids: np.ndarray
        Coordinates of cluster centroids of shape (n_topics, embedding_size)
    vocab_embeddings: np.ndarray
        Term embeddings of shape (vocab_size, embedding_size)
    metric: str, defaul 'cosine'
        Metric used to compute distance from centroid.
        See documentation for sklearn.metrics.pairwise.distance_metrics
        for valid values.

    Returns
    -------
    ndarray of shape (n_topics, vocab_size)
        Term importance matrix.
    """
    distances = pairwise_distances(
        cluster_centroids, vocab_embeddings, metric=metric
    )
    similarities = -distances / np.max(distances)
    # Z-score transformation
    similarities = (similarities - np.mean(similarities)) / np.std(
        similarities
    )
    return similarities


def soft_ctf_idf(
    doc_topic_matrix: np.ndarray, doc_term_matrix: spr.csr_matrix
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
    return ctf_idf


def ctf_idf(
    doc_topic_matrix: np.ndarray, doc_term_matrix: spr.csr_matrix
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
    for i_topic in range(n_topics):
        freq = np.ravel(
            np.asarray(doc_term_matrix[labels == i_topic].sum(axis=0))
        )
        component = freq * np.log(1 + average / overall_freq)
        components.append(component)
    return np.stack(components)


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
    p_t = doc_topic_matrix.sum(axis=0)
    p_t = p_t / p_t.sum()
    term_importance = doc_topic_matrix.T @ doc_term_matrix
    overall_in_topic = np.abs(term_importance).sum(axis=1)
    p_wt = (term_importance.T / (overall_in_topic + eps)).T
    p_wt = normalize(p_wt, norm="l1", axis=1)
    p_tw = (p_wt.T * p_t).T / p_w
    p_tw = normalize(p_tw, axis=0, norm="l1")
    return p_tw
