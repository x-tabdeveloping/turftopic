import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def cluster_centroid_distance(
    cluster_labels, embeddings, vocab_embeddings, metric="euclidean"
):
    centroids = []
    unique_labels = np.unique(cluster_labels)
    unique_labels = np.sort(unique_labels)
    for label in unique_labels:
        centroid = np.mean(embeddings[cluster_labels == label], axis=0)
        centroids.append(centroid)
    centroids = np.stack(centroids)
    distances = pairwise_distances(centroids, vocab_embeddings, metric=metric)
    similarities = -distances / np.max(distances)
    # Z-score transformation
    similarities = (similarities - np.mean(similarities)) / np.std(
        similarities
    )
    return similarities
