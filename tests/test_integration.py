import itertools
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA

from turftopic import (
    GMM,
    AutoEncodingTopicModel,
    ClusteringTopicModel,
    FASTopic,
    KeyNMF,
    SemanticSignalSeparation,
    SensTopic,
    Topeax,
    load_model,
)


def batched(iterable, n: int):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def generate_dates(
    n_dates: int,
) -> list[datetime]:
    """Generate random dates to test dynamic models"""
    dates = []
    for n in range(n_dates):
        d = np.random.randint(low=1, high=29)
        m = np.random.randint(low=1, high=13)
        y = np.random.randint(low=2000, high=2020)
        date = datetime(year=y, month=m, day=d)
        dates.append(date)
    return dates


newsgroups = fetch_20newsgroups(
    subset="all",
    categories=[
        "misc.forsale",
    ],
    remove=("headers", "footers", "quotes"),
)
texts = newsgroups.data
trf = SentenceTransformer("paraphrase-MiniLM-L3-v2")
embeddings = np.asarray(trf.encode(texts))
timestamps = generate_dates(n_dates=len(texts))

models = [
    GMM(3, encoder=trf),
    SemanticSignalSeparation(3, encoder=trf),
    KeyNMF(3, encoder=trf),
    KeyNMF(3, encoder=trf, cross_lingual=True),
    ClusteringTopicModel(
        dimensionality_reduction=PCA(10),
        clustering=KMeans(3),
        feature_importance="c-tf-idf",
        encoder=trf,
        reduction_method="average",
    ),
    ClusteringTopicModel(
        dimensionality_reduction=PCA(10),
        clustering=KMeans(3),
        feature_importance="centroid",
        encoder=trf,
        reduction_method="smallest",
    ),
    AutoEncodingTopicModel(3, combined=True),
    FASTopic(3, batch_size=None),
    SensTopic(),
    Topeax(),
]

dynamic_models = [
    GMM(3, encoder=trf),
    ClusteringTopicModel(
        dimensionality_reduction=PCA(10),
        clustering=KMeans(3),
        feature_importance="centroid",
        encoder=trf,
        reduction_method="smallest",
    ),
    ClusteringTopicModel(
        dimensionality_reduction=PCA(10),
        clustering=KMeans(3),
        feature_importance="soft-c-tf-idf",
        encoder=trf,
        reduction_method="smallest",
    ),
    KeyNMF(3, encoder=trf),
]

online_models = [KeyNMF(3, encoder=trf)]


@pytest.mark.parametrize("model", dynamic_models)
def test_fit_dynamic(model):
    doc_topic_matrix = model.fit_transform_dynamic(
        texts,
        embeddings=embeddings,
        timestamps=timestamps,
    )
    table = model.export_topics(format="csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_path = Path(tmpdirname).joinpath("topics.csv")
        with out_path.open("w") as out_file:
            out_file.write(table)
        df = pd.read_csv(out_path)


@pytest.mark.parametrize("model", online_models)
def test_fit_online(model):
    for epoch in range(5):
        for batch in batched(zip(texts, embeddings), 50):
            batch_text, batch_embedding = zip(*batch)
            batch_text = list(batch_text)
            batch_embedding = np.stack(batch_embedding)
            model.partial_fit(batch_text, embeddings=batch_embedding)
    table = model.export_topics(format="csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_path = Path(tmpdirname).joinpath("topics.csv")
        with out_path.open("w") as out_file:
            out_file.write(table)
        df = pd.read_csv(out_path)


OPTIONAL_FIELDS = [
    "topic_names",
    "classes",
    "corpus",
    "transform",
    "time_bin_edges",
    "temporal_components",
    "temporal_importance",
    "has_negative_side",
    "hierarchy",
]


@pytest.mark.parametrize("model", models)
def test_prepare_topic_data_export_table(model):
    topic_data = model.prepare_topic_data(texts, embeddings=embeddings)
    for key, value in topic_data.items():
        if key in OPTIONAL_FIELDS:
            continue
        if value is None:
            raise TypeError(f"Field {key} is None in topic_data.")
    table = model.export_topics(format="csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_path = Path(tmpdirname).joinpath("topics.csv")
        with out_path.open("w") as out_file:
            out_file.write(table)
        df = pd.read_csv(out_path)


def test_hierarchical():
    model = KeyNMF(2).fit(texts, embeddings=embeddings)
    model.hierarchy.divide_children(3)
    model.hierarchy[0][0].divide(3)
    repr = str(model.hierarchy)


def test_hierarchical_clustering():
    model = ClusteringTopicModel(
        n_reduce_to=5,
        dimensionality_reduction=PCA(10),
        clustering=KMeans(20),
        feature_importance="c-tf-idf",
        encoder=trf,
        reduction_method="smallest",
        reduction_topic_representation="centroid",
    )
    topic_data = model.prepare_topic_data(texts, embeddings=embeddings)
    assert model.components_.shape[0] == 5
    fig = model.hierarchy.plot_tree()
    print(model.hierarchy.cut(2))


def test_naming():
    model = KeyNMF(2).fit(texts, embeddings=embeddings)
    topic_names = ["Topic 1", "Topic 2"]
    model.rename_topics(topic_names)
    assert topic_names == model.topic_names
    model.rename_topics(
        {
            topic_id: topic_name
            for topic_id, topic_name in enumerate(topic_names)
        }
    )
    assert topic_names == model.topic_names


def test_topic_joining():
    model = ClusteringTopicModel(
        dimensionality_reduction=PCA(2),
        clustering=KMeans(5),
        feature_importance="c-tf-idf",
        encoder=trf,
        reduction_method="smallest",
    )
    model.fit(texts, embeddings=embeddings)
    model.join_topics([0, 1, 2])
    assert set(model.classes_) == {0, 3, 4}


def test_refitting():
    model = SemanticSignalSeparation(10)
    model.fit(texts, embeddings=embeddings)
    model.refit(texts, embeddings=embeddings, n_components=20)
    assert model.components_.shape[0] == 20


def test_serialization():
    model = SemanticSignalSeparation(10)
    model.fit(texts, embeddings=embeddings)
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.to_disk(tmp_dir)
        model = load_model(tmp_dir)
