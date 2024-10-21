import math
import random
from typing import Optional, Union

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel, Encoder
from turftopic.vectorizer import default_vectorizer


class EncoderNetwork(nn.Module):
    def __init__(self, contextualized_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(contextualized_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale


class DecoderNetwork(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class Model(nn.Module):
    def __init__(
        self, vocab_size, contextualized_size, num_topics, hidden, dropout
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = EncoderNetwork(
            contextualized_size, num_topics, hidden, dropout
        )
        self.decoder = DecoderNetwork(vocab_size, num_topics, dropout)

    def model(self, bow, contextualized):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", bow.shape[0]):
            logtheta_loc = bow.new_zeros((bow.shape[0], self.num_topics))
            logtheta_scale = bow.new_ones((bow.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta",
                dist.Normal(logtheta_loc, logtheta_scale).to_event(1),
            )
            theta = F.softmax(logtheta, -1)
            count_param = self.decoder(theta)
            total_count = int(bow.sum(-1).max())
            pyro.sample(
                "obs", dist.Multinomial(total_count, count_param), obs=bow
            )

    def guide(self, bow, contextualized):
        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", contextualized.shape[0]):
            logtheta_loc, logtheta_scale = self.encoder(contextualized)
            logtheta = pyro.sample(
                "logtheta",
                dist.Normal(logtheta_loc, logtheta_scale).to_event(1),
            )

    def beta(self):
        return self.decoder.beta.weight.cpu().detach().T


class AutoEncodingTopicModel(ContextualModel):
    """Variational autoencoding topic models
    with contextualized representations (CTM).
    Uses amortized variational inference with neural networks
    to estimate posterior for ProdLDA.

    ```python
    from turftopic import AutoEncodingTopicModel

    corpus: list[str] = ["some text", "more text", ...]

    model = AutoEncodingTopicModel(10, combined=False).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int
        Number of topics.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    combined: bool, default False
        Indicates whether encoder inputs should be combined
        with bow representations.
        When False the model is equivalent to ZeroShotTM,
        when True it is CombinedTM.
    dropout_rate: float, default 0.1
        Dropout in the encoder layers.
    hidden: int, default 100
        Size of hidden layers in the encoder network.
    batch_size: int, default 42
        Batch size when training the network.
    learning_rate: float, default 1e-2
        Learning rate for the optimizer.
    n_epochs: int, default 50
        Number of epochs to run during training.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    """

    def __init__(
        self,
        n_components: int,
        encoder: Union[
            Encoder, SentenceTransformer
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        combined: bool = False,
        dropout_rate: float = 0.1,
        hidden: int = 100,
        batch_size: int = 42,
        learning_rate: float = 1e-2,
        n_epochs: int = 50,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.combined = combined
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.hidden = hidden

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ):
        """Infers topic importances for new documents based on a fitted model.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed document encodings.

        Returns
        -------
        ndarray of shape (n_dimensions, n_topics)
            Document-topic matrix.
        """
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        if self.combined:
            bow = self.vectorizer.fit_transform(raw_documents)
            contextual_embeddings = np.concatenate(
                (embeddings, bow.toarray()), axis=1
            )
        else:
            contextual_embeddings = embeddings
        contextual_embeddings = torch.tensor(contextual_embeddings).float()
        loc, scale = self.model.encoder(contextual_embeddings)
        prob = torch.softmax(loc, dim=-1)
        return prob.cpu().data.numpy()

    def fit(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        console = Console()
        with console.status("Fitting model") as status:
            if embeddings is None:
                status.update("Encoding documents")
                embeddings = self.encoder_.encode(raw_documents)
                console.log("Documents encoded.")
            status.update("Extracting terms.")
            document_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            seed = self.random_state or random.randint(0, 10_000)
            torch.manual_seed(seed)
            pyro.set_rng_seed(seed)
            device = torch.device("cpu")
            pyro.clear_param_store()
            contextualized_size = embeddings.shape[1]
            if self.combined:
                contextualized_size = (
                    contextualized_size + document_term_matrix.shape[1]
                )
            self.model = Model(
                vocab_size=document_term_matrix.shape[1],
                contextualized_size=contextualized_size,
                num_topics=self.n_components,
                hidden=self.hidden,
                dropout=self.dropout_rate,
            )
            self.model.to(device)
            optimizer = pyro.optim.Adam({"lr": self.learning_rate})
            svi = SVI(
                self.model.model,
                self.model.guide,
                optimizer,
                loss=TraceMeanField_ELBO(),
            )
            num_batches = int(
                math.ceil(document_term_matrix.shape[0] / self.batch_size)
            )

            status.update(f"Fitting model. Epoch [0/{self.n_epochs}]")
            for epoch in range(self.n_epochs):
                running_loss = 0.0
                for i in range(num_batches):
                    batch_bow = np.atleast_2d(
                        document_term_matrix[
                            i * self.batch_size : (i + 1) * self.batch_size, :
                        ].toarray()
                    )
                    # Skipping batches that are smaller than 2
                    if batch_bow.shape[0] < 2:
                        continue
                    batch_contextualized = np.atleast_2d(
                        embeddings[
                            i * self.batch_size : (i + 1) * self.batch_size, :
                        ]
                    )
                    if self.combined:
                        batch_contextualized = np.concatenate(
                            (batch_contextualized, batch_bow), axis=1
                        )
                    batch_contextualized = (
                        torch.tensor(batch_contextualized).float().to(device)
                    )
                    batch_bow = torch.tensor(batch_bow).float().to(device)
                    loss = svi.step(batch_bow, batch_contextualized)
                    running_loss += loss / batch_bow.size(0)
                status.update(
                    f"Fitting model. Epoch [{epoch}/{self.n_epochs}], Loss [{running_loss}]"
                )
            self.components_ = self.model.beta()
            console.log("Model fitting done.")
        return self

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.fit(raw_documents, y, embeddings).transform(
            raw_documents, embeddings
        )
