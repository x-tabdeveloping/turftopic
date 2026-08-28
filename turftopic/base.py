import json
import tempfile
from abc import ABC, abstractmethod
from functools import partial
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
from turftopic.vectorizers.merging import merge_vectorizers

Encoder = Union[ExternalEncoder, SentenceTransformer]


class ContextualModel(BaseEstimator, TransformerMixin, TopicContainer):
    """Base class for contextual topic models in Turftopic."""

    def load_encoder(self):
        if isinstance(self.encoder, str):
            if self.trf_kwargs is None:
                trf_kwargs = dict()
            else:
                trf_kwargs = self.trf_kwargs
            self.encoder_ = SentenceTransformer(self.encoder, **trf_kwargs)
            self._encoder_preloaded = False
        else:
            self.encoder_ = self.encoder
            self._encoder_preloaded = True

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
        if getattr(self, "encode_kwargs", None) is None:
            encode_kwargs = dict()
        else:
            encode_kwargs = self.encode_kwargs
        return self.encoder_.encode(list(raw_documents), **encode_kwargs)

    def encode_vocabulary_items(self, vocab: Iterable[str]) -> np.ndarray:
        return self.encode_documents(list(vocab))

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

    def _update_vocab_embeddings(self, diff_terms, diff_vocab_embeddings=None):
        if (diff_vocab_embeddings is None) or (
            diff_vocab_embeddings.shape[1] != self.vocab_embeddings.shape[1]
        ):
            diff_vocab_embeddings = self.encode_vocabulary_items(diff_terms)
        self.vocab_embeddings = np.concatenate(
            [self.vocab_embeddings, diff_vocab_embeddings], axis=0
        )

    def _pad_components(self, n_diff_terms: int):
        NO_PADDING = (0, 0)
        if getattr(self, "components_", None) is not None:
            self.components_ = np.pad(
                self.components_,
                # (n_timebins, n_components, n_terms)
                [NO_PADDING, (0, n_diff_terms)],
            )
        if getattr(self, "temporal_components_", None) is not None:
            self.temporal_components_ = np.pad(
                self.temporal_components_,
                # (n_timebins, n_components, n_terms)
                [NO_PADDING, NO_PADDING, (0, n_diff_terms)],
            )
        if getattr(self, "axial_temporal_components_", None) is not None:
            self.axial_temporal_components_ = np.pad(
                self.axial_temporal_components_,
                # (n_timebins, n_components, n_terms)
                [NO_PADDING, NO_PADDING, (0, n_diff_terms)],
            )

    def update_vocabulary(self, other_vectorizer_or_model):
        """Updates the model's vocabulary with another model's or vectorizer's vocab.

        Parameters
        ----------
        other_vectorizer_or_model: ContextualModel or Vectorizer
            Other topic model or vectorizer to update the model's vocabulary with.
        """
        if getattr(self.vectorizer, "vocabulary_", None) is None:
            raise ValueError(
                "Can't update vocabulary because the model's vectorizer has not been fitted yet."
                "Perhaps you should pass the vectorizer to the model upon initialization."
            )
        if getattr(other_vectorizer_or_model, "vectorizer", None) is not None:
            other_vectorizer = other_vectorizer_or_model.vectorizer
        else:
            other_vectorizer = other_vectorizer_or_model
        # Joining vectorizers
        joint_vectorizer = merge_vectorizers(self.vectorizer, other_vectorizer)
        # Finding term difference
        old_vectorizer = self.vectorizer
        old_terms = old_vectorizer.get_feature_names_out()
        joint_terms = joint_vectorizer.get_feature_names_out()
        n_old = len(old_terms)
        diff_terms = joint_terms[n_old:]
        if len(diff_terms) == 0:
            self.vectorizer = joint_vectorizer
            return 0
        # Encoding new vocabulary items.
        diff_to_other = [other_vectorizer.vocabulary_[t] for t in diff_terms]
        # Updating vocab_embeddings if the model has them.
        if getattr(self, "vocab_embeddings", None) is not None:
            if (
                getattr(other_vectorizer_or_model, "vocab_embeddings", None)
                is not None
            ):
                diff_vocab_embeddings = (
                    other_vectorizer_or_model.vocab_embeddings[
                        diff_to_other, :
                    ]
                )
            else:
                diff_vocab_embeddings = None
            self._update_vocab_embeddings(diff_terms, diff_vocab_embeddings)
        # Padding all components with NaNs where they haven't been fitted yet.
        self._pad_components(n_diff_terms=len(diff_terms))
        # Setting joint vectorizer
        self.vectorizer = joint_vectorizer
        return len(diff_terms)

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
        if getattr(self, "_encoder_preloaded", True):
            joblib.dump(self, out_dir.joinpath("model.joblib"))
        else:
            encoder_ = self.encoder_
            delattr(self, "encoder_")
            joblib.dump(self, out_dir.joinpath("model.joblib"))
            self.encoder_ = encoder_

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
