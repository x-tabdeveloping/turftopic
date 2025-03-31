import json
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional, Union

import joblib
import numpy as np
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from turftopic.container import TopicContainer
from turftopic.data import TopicData
from turftopic.encoders import ExternalEncoder
from turftopic.serialization import create_readme, get_package_versions

Encoder = Union[ExternalEncoder, SentenceTransformer]


class ContextualModel(BaseEstimator, TransformerMixin, TopicContainer):
    """Base class for contextual topic models in Turftopic."""

    @property
    def has_negative_side(self) -> bool:
        return False

    def encode_documents(self, raw_documents: Iterable[str]) -> np.ndarray:
        """Encodes documents with the sentence encoder of the topic model.

        Parameters
        ----------
        raw_documents: iterable of str
            Textual documents to encode.

        Return
        ------
        ndarray of shape (n_documents, n_dimensions)
            Matrix of document embeddings.
        """
        if not hasattr(self.encoder_, "encode"):
            return self.encoder.get_text_embeddings(list(raw_documents))
        return self.encoder_.encode(raw_documents)

    @abstractmethod
    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fits model and infers topic importances for each document.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        y: None
            Ignored, exists for sklearn compatibility.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.

        Returns
        -------
        ndarray of shape (n_documents, n_topics)
            Document-topic matrix.
        """
        pass

    def fit(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        """Fits model on the given corpus.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        y: None
            Ignored, exists for sklearn compatibility.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.
        """
        self.fit_transform(raw_documents, y, embeddings)
        return self

    def get_vocab(self) -> np.ndarray:
        """Get vocabulary of the model.

        Returns
        -------
        ndarray of shape (n_vocab)
            All terms in the vocabulary.
        """
        return self.vectorizer.get_feature_names_out()

    def get_feature_names_out(self) -> np.ndarray:
        """Get topic ids.

        Returns
        -------
        ndarray of shape (n_topics)
            IDs for each output feature of the model.
            This is useful, since some models have outlier
            detection, and this gets -1 as ID, instead of
            its index.
        """
        n_topics = self.components_.shape[0]
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(n_topics))
        return np.asarray(classes)

    def prepare_topic_data(
        self,
        corpus: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> TopicData:
        """Produces topic inference data for a given corpus, that can be then used and reused.
        Exists to allow visualizations out of the box with topicwizard.

        Parameters
        ----------
        corpus: list of str
            Documents to infer topical content for.
        embeddings: ndarray of shape (n_documents, n_dimensions)
            Embeddings of documents.

        Returns
        -------
        TopicData
            Information about topical inference in a dictionary.
        """
        if embeddings is None:
            embeddings = self.encode_documents(corpus)
        try:
            document_topic_matrix = self.transform(
                corpus, embeddings=embeddings
            )
        except (AttributeError, NotFittedError):
            document_topic_matrix = self.fit_transform(
                corpus, embeddings=embeddings
            )
        dtm = self.vectorizer.transform(corpus)  # type: ignore
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(self.components_.shape[0]))
        res = TopicData(
            corpus=corpus,
            document_term_matrix=dtm,
            vocab=self.get_vocab(),
            document_topic_matrix=document_topic_matrix,
            document_representation=embeddings,
            topic_term_matrix=self.components_,  # type: ignore
            transform=getattr(self, "transform", None),
            topic_names=self.topic_names,
            classes=classes,
            has_negative_side=self.has_negative_side,
            hierarchy=getattr(self, "hierarchy", None),
        )
        return res

    def to_disk(self, out_dir: Union[Path, str]):
        """Persists model to directory on your machine.

        Parameters
        ----------
        out_dir: Path | str
            Directory to save the model to.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
        package_versions = get_package_versions()
        with out_dir.joinpath("package_versions.json").open("w") as ver_file:
            ver_file.write(json.dumps(package_versions))
        joblib.dump(self, out_dir.joinpath("model.joblib"))

    def push_to_hub(self, repo_id: str):
        """Uploads model to HuggingFace Hub

        Parameters
        ----------
        repo_id: str
            Repository to upload the model to.
        """
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            readme_path = Path(tmp_dir).joinpath("README.md")
            with readme_path.open("w") as readme_file:
                readme_file.write(create_readme(self, repo_id))
            self.to_disk(tmp_dir)
            api.upload_folder(
                folder_path=tmp_dir,
                repo_id=repo_id,
                repo_type="model",
            )
