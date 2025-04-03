import pytest
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from turftopic import (
    GMM,
    AutoEncodingTopicModel,
    ClusteringTopicModel,
    KeyNMF,
    SemanticSignalSeparation,
)

encoder = "sentence-transformers/clip-ViT-B-16"
MULTIMODAL_MODELS = [
    AutoEncodingTopicModel(
        2, combined=True, encoder=encoder, vectorizer=CountVectorizer()
    ),
    GMM(2, encoder=encoder, vectorizer=CountVectorizer()),
    KeyNMF(2, encoder=encoder, vectorizer=CountVectorizer()),
    SemanticSignalSeparation(2, encoder=encoder, vectorizer=CountVectorizer()),
    ClusteringTopicModel(
        dimensionality_reduction=PCA(10),
        clustering=KMeans(3),
        feature_importance="c-tf-idf",
        encoder=encoder,
    ),
    ClusteringTopicModel(
        dimensionality_reduction=PCA(10),
        clustering=KMeans(3),
        feature_importance="centroid",
        encoder=encoder,
    ),
]

flowers = load_dataset(
    "Multimodal-Fatima/OxfordFlowers_test", split="test"
).train_test_split(test_size=256, seed=42)["test"]
texts = flowers["blip_caption"]
images = flowers["image"]


@pytest.mark.parametrize("model", MULTIMODAL_MODELS)
def test_multimodal(model):
    doc_topic_matrix = model.fit_transform_multimodal(texts, images=images)
    fig = model.plot_topics_with_images()
    assert len(model.top_images) == model.components_.shape[0]
    assert doc_topic_matrix.shape[1] == model.components_.shape[0]
