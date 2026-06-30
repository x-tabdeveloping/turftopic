import json
import tempfile
from pathlib import Path
from typing import Union

import joblib
import numpy as np
from huggingface_hub import HfApi
from sklearn.base import BaseEstimator, TransformerMixin

from turftopic.late import (
    LateSentenceTransformer,
    flatten_repr,
    pool_flat,
)
from turftopic.serialization import create_readme, get_package_versions
from turftopic.vectorizers.latent_terms.top_k_autoencoder import (
    TopKAutoEncoder,
)


class LatentTermsVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        encoder: str | LateSentenceTransformer,
        autoencoder: TopKAutoEncoder,
        concept_labels: np.ndarray,
        show_progress_bar: bool = True,
    ):
        self.encoder = encoder
        if isinstance(self.encoder, str):
            self._encoder = LateSentenceTransformer(self.encoder)
        else:
            self._encoder = self.encoder
        self.concept_labels = np.array(concept_labels)
        self.autoencoder = autoencoder
        self.show_progress_bar = show_progress_bar
        self.autoencoder.show_progress_bar = show_progress_bar

    def fit(self, raw_documents, y=None):
        # Does nothing, for compatibility
        return self

    def transform(self, raw_documents):
        token_embeddings, offsets = self._encoder.encode_tokens(
            list(raw_documents), show_progress_bar=self.show_progress_bar
        )
        flat_token_embeddings, lengths = flatten_repr(token_embeddings)
        flat_z = self.autoencoder.transform(flat_token_embeddings)
        # Pooling procedure from section 3.2
        pooled_z = pool_flat(flat_z, lengths=lengths, agg=np.sum)
        return np.sqrt(pooled_z)

    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents)

    def get_feature_names_out(self):
        return self.concept_labels

    @classmethod
    def from_dict(cls, data):
        autoencoder = TopKAutoEncoder.from_dict(data["autoencoder"])
        return cls(
            encoder=data["encoder"],
            autoencoder=autoencoder,
            show_progress_bar=data["show_progress_bar"],
            concept_labels=data["concept_labels"],
        )

    def to_dict(self):
        return dict(
            encoder=self.encoder,
            autoencoder=self.autoencoder.to_dict(),
            show_progress_bar=self.show_progress_bar,
            concept_labels=self.concept_labels,
        )

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
