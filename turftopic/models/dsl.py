from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel
from turftopic.encoders.soft_labels import (
    SoftLabelEncoder,
    map_vocabulary,
    soft_labels,
)


class _Network(nn.Module):
    """ProdLDA-style variational autoencoder over language-model embeddings."""

    def __init__(self, input_size, vocab_size, n_components, hidden, dropout):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Dropout(dropout),
        )
        self.f_mu = nn.Linear(hidden, n_components)
        self.f_mu_bn = nn.BatchNorm1d(n_components, affine=False)
        self.f_sigma = nn.Linear(hidden, n_components)
        self.f_sigma_bn = nn.BatchNorm1d(n_components, affine=False)
        self.prior_mean = nn.Parameter(torch.zeros(n_components))
        self.prior_variance = nn.Parameter(
            torch.full((n_components,), 1.0 - 1.0 / n_components)
        )
        self.beta = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(n_components, vocab_size))
        )
        self.beta_bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.theta_dropout = nn.Dropout(dropout)

    def encode(self, x):
        h = self.encoder(x)
        return self.f_mu_bn(self.f_mu(h)), self.f_sigma_bn(self.f_sigma(h))

    def forward(self, x, temperature):
        mu, log_var = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        theta = self.theta_dropout(torch.softmax(z, dim=1))
        word_dist = torch.softmax(
            self.beta_bn(theta @ self.beta) / temperature, dim=1
        )
        return word_dist, mu, log_var

    def prior_kl(self, mu, log_var):
        var = log_var.exp()
        return 0.5 * (
            (var / self.prior_variance).sum(1)
            + ((self.prior_mean - mu) ** 2 / self.prior_variance).sum(1)
            - mu.shape[1]
            + self.prior_variance.log().sum()
            - log_var.sum(1)
        )


class DSL(ContextualModel):
    """Topic model trained by Distilling Soft Labels from a language model
    (Li et al., 2026).

    A small causal language model reads each document in a prompt that asks for its
    theme. Its next-token distribution over the vocabulary is the reconstruction target
    (instead of the bag-of-words), and its last hidden state is the document
    representation. The topic model itself is a ProdLDA-style autoencoder.

    ```python
    from turftopic import DSL

    corpus: list[str] = ["some text", "more text", ...]

    model = DSL(10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int
        Number of topics.
    encoder: str or SoftLabelEncoder
        Causal language model used as the teacher,
        ERNIE-4.5-0.3B is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Defaults to the 2000 most frequent non-stop words.
        Only words that are a single alphabetic token for the language model
        are part of the soft labels, so unigram vocabularies work best.
    temperature: float, default 3.0
        Softmax temperature of the soft labels.
    loss_weight: float, default 1000.0
        Weight of the distillation term against the prior term.
    hidden: int, default 200
        Units in each of the two encoder layers.
    dropout_rate: float, default 0.2
        Dropout in the encoder and on topic proportions.
    batch_size: int, default 64
        Batch size.
    learning_rate: float, default 2e-3
        Learning rate of Adam.
    n_epochs: int, default 100
        Number of training epochs.
    random_state: int, default None
        Random state for reproducible training.
    device: str, default "cpu"
        Device to train the topic model on.
    """

    def __init__(
        self,
        n_components: int,
        encoder: Union[SoftLabelEncoder, str] = "baidu/ERNIE-4.5-0.3B-PT",
        vectorizer: Optional[CountVectorizer] = None,
        temperature: float = 3.0,
        loss_weight: float = 1000.0,
        hidden: int = 200,
        dropout_rate: float = 0.2,
        batch_size: int = 64,
        learning_rate: float = 2e-3,
        n_epochs: int = 100,
        random_state: Optional[int] = None,
        device: str = "cpu",
        trf_kwargs=None,
        encode_kwargs=None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.trf_kwargs = trf_kwargs
        self.encode_kwargs = encode_kwargs
        self.random_state = random_state
        if isinstance(encoder, str):
            self.encoder_ = SoftLabelEncoder(encoder, **(trf_kwargs or {}))
            self._encoder_preloaded = False
        else:
            self.encoder_ = encoder
            self._encoder_preloaded = True
        if vectorizer is None:
            self.vectorizer = CountVectorizer(
                max_features=2000,
                stop_words="english",
                token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            )
        else:
            self.vectorizer = vectorizer
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.hidden = hidden
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device

    def fit_transform(
        self,
        raw_documents,
        y=None,
        embeddings: Optional[np.ndarray] = None,
        logits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fits the model and returns the document-topic matrix.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed hidden states from the teacher.
        logits: ndarray of shape (n_documents, n_vocab), optional
            Precomputed teacher logits over the vectorizer's vocabulary.
        """
        console = Console()
        raw_documents = list(raw_documents)
        with console.status("Fitting model") as status:
            status.update("Extracting terms.")
            self.vectorizer.fit(raw_documents)
            vocab = self.get_vocab()
            console.log("Term extraction done.")
            if embeddings is None or logits is None:
                status.update("Running the language model")
                logits, embeddings, _ = self.encoder_.extract(
                    raw_documents, list(vocab)
                )
                console.log("Soft labels and embeddings extracted.")
            # Words that are not one alphabetic token for the language model
            # cannot be scored exactly, and are left out of the target.
            scored = map_vocabulary(
                self.encoder_.tokenizer, list(vocab)
            ).single_token
            targets = soft_labels(logits, self.temperature, mask=scored)
            if self.random_state is not None:
                torch.manual_seed(self.random_state)
            self.model = _Network(
                embeddings.shape[1],
                len(vocab),
                self.n_components,
                self.hidden,
                self.dropout_rate,
            ).to(self.device)
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(0.99, 0.99),
            )
            x_all = torch.as_tensor(np.asarray(embeddings)).float()
            y_all = torch.as_tensor(targets).float()
            generator = torch.Generator()
            if self.random_state is not None:
                generator.manual_seed(self.random_state)
            self.model.train()
            for epoch in range(self.n_epochs):
                order = torch.randperm(len(x_all), generator=generator)
                running, seen = 0.0, 0
                for start in range(0, len(order), self.batch_size):
                    idx = order[start : start + self.batch_size]
                    if len(idx) < 2:
                        continue
                    x = x_all[idx].to(self.device)
                    target = y_all[idx].to(self.device).clamp_min(1e-9)
                    optimizer.zero_grad()
                    word_dist, mu, log_var = self.model(x, self.temperature)
                    distill = (
                        target * (target / word_dist.clamp_min(1e-9)).log()
                    ).sum(1)
                    loss = (
                        self.model.prior_kl(mu, log_var)
                        + self.loss_weight * distill
                    ).sum()
                    loss.backward()
                    optimizer.step()
                    running += loss.item()
                    seen += len(idx)
                status.update(
                    f"Fitting model. Epoch [{epoch + 1}/{self.n_epochs}], "
                    f"loss [{running / max(seen, 1):.2f}]"
                )
            self.components_ = self.model.beta.detach().cpu().numpy()
            console.log("Model fitting done.")
            doc_topic_matrix = self.transform(
                raw_documents, embeddings=embeddings
            )
            self.top_documents = self.get_top_documents(
                raw_documents, document_topic_matrix=doc_topic_matrix
            )
        return doc_topic_matrix

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Infers topic proportions of documents.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to infer topics for.
        embeddings: ndarray of shape (n_documents, n_dimensions), optional
            Precomputed hidden states from the teacher.

        Returns
        -------
        ndarray of shape (n_documents, n_topics)
            Document-topic matrix.
        """
        if embeddings is None:
            embeddings = self.encoder_.encode(list(raw_documents))
        self.model.eval()
        chunks = []
        with torch.no_grad():
            x_all = torch.as_tensor(np.asarray(embeddings)).float()
            for start in range(0, len(x_all), 1024):
                mu, _ = self.model.encode(
                    x_all[start : start + 1024].to(self.device)
                )
                chunks.append(torch.softmax(mu, dim=1).cpu().numpy())
        return np.concatenate(chunks, axis=0)
