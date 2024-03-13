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
    SemanticSignalSeparation,
)

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


@pytest.mark.parametrize("model", models)
def test_fit_export_table(model):
    doc_topic_matrix = model.fit_transform(texts, embeddings=embeddings)
    table = model.export_topics(format="csv")
    with tempfile.TemporaryDirectory() as tmpdirname:
        out_path = Path(tmpdirname).joinpath("topics.csv")
        with out_path.open("w") as out_file:
            out_file.write(table)
        df = pd.read_csv(out_path)
