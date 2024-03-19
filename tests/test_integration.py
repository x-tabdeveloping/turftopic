from datetime import datetime
import tempfile
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
    SemanticSignalSeparation
)


def generate_dates(
        n_dates: int,
) -> list[datetime]:
        """ Generate random dates to test dynamic models """
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
    KeyNMF(5, encoder=trf, keyword_scope='document'),
    KeyNMF(5, encoder=trf, keyword_scope='corpus'),
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
        reduction_method="smallest"
    ),
    ClusteringTopicModel(
        n_reduce_to=5,
        feature_importance="soft-c-tf-idf",
        encoder=trf,
        reduction_method="smallest"
    )
]


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
         texts, embeddings=embeddings, timestamps=timestamps,
    )
    table = model.export_topics(format="csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_path = Path(tmpdirname).joinpath("topics.csv")
        with out_path.open("w") as out_file:
            out_file.write(table)
        df = pd.read_csv(out_path)
