import itertools
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

from turftopic import (
    GMM,
    AutoEncodingTopicModel,
    ClusteringTopicModel,
    KeyNMF,
    SemanticSignalSeparation,
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
trf = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.asarray(trf.encode(texts))
timestamps = generate_dates(n_dates=len(texts))

models = [
    GMM(5, encoder=trf),
    SemanticSignalSeparation(5, encoder=trf),
    KeyNMF(5, encoder=trf),
    ClusteringTopicModel(
        n_reduce_to=5,
        feature_importance="c-tf-idf",
        encoder=trf,
        reduction_method="agglomerative",
    ),
    ClusteringTopicModel(
        n_reduce_to=5,
        feature_importance="centroid",
        encoder=trf,
        reduction_method="smallest",
    ),
    AutoEncodingTopicModel(5, combined=True),
]

dynamic_models = [
    GMM(5, encoder=trf),
    ClusteringTopicModel(
        n_reduce_to=5,
        feature_importance="centroid",
        encoder=trf,
        reduction_method="smallest",
    ),
    ClusteringTopicModel(
        n_reduce_to=5,
        feature_importance="soft-c-tf-idf",
        encoder=trf,
        reduction_method="smallest",
    ),
    KeyNMF(5, encoder=trf),
]

online_models = [KeyNMF(5, encoder=trf)]


@pytest.mark.parametrize("model", models)
def test_fit_export_table(model):
    doc_topic_matrix = model.fit_transform(texts, embeddings=embeddings)
    table = model.export_topics(format="csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_path = Path(tmpdirname).joinpath("topics.csv")
        with out_path.open("w") as out_file:
            out_file.write(table)
        df = pd.read_csv(out_path)


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
