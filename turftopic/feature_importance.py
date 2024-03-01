import numpy as np
import scipy.sparse as spr
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def cluster_centroid_distance(
    cluster_centroids: np.ndarray,
    vocab_embeddings: np.ndarray,
    metric="cosine",
) -> np.ndarray:
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
