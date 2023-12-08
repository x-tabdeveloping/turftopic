import math
from typing import Optional, Union

import numpy as np
import pyro
import pyro.distributions as dist
import scipy.sparse as spr
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import trange

from turftopic.base import ContextualModel


class Encoder(nn.Module):
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


class Decoder(nn.Module):
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
        self.encoder = Encoder(
            contextualized_size, num_topics, hidden, dropout
        )
        self.decoder = Decoder(vocab_size, num_topics, dropout)

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
    def __init__(
        self,
        n_components: int,
        encoder: Union[
            str, SentenceTransformer
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        combined: bool = False,
        dropout_rate: float = 0.1,
        hidden: int = 100,
        batch_size: int = 42,
        learning_rate: float = 1e-2,
        n_epochs: int = 50,
    ):
        self.n_components = n_components
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = CountVectorizer(min_df=10)
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
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        if self.combined:
            bow = self.vectorizer.fit_transform(raw_documents)
            contextual_embeddings = np.concatenate(
                (embeddings, bow.toarray()), axis=1
            )
        else:
            contextual_embeddings = embeddings
        loc, scale = self.model.encoder(contextual_embeddings)
        prob = torch.softmax(loc, dim=-1)
        return prob.numpy()

    def fit(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ):
        if embeddings is None:
            embeddings = self.encoder_.encode(raw_documents)
        document_term_matrix = self.vectorizer.fit_transform(raw_documents)
        seed = 0
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pyro.clear_param_store()
        self.model = Model(
            vocab_size=document_term_matrix.shape[1],
            contextualized_size=embeddings.shape[1],
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

        bar = trange(self.n_epochs)
        for epoch in bar:
            running_loss = 0.0
            for i in range(num_batches):
                batch_bow = document_term_matrix[
                    i * self.batch_size : (i + 1) * self.batch_size, :
                ]
                batch_contextualized = embeddings[
                    i * self.batch_size : (i + 1) * self.batch_size, :
                ]
                if self.combined:
                    batch_contextualized = np.concatenate(
                        (embeddings, batch_bow.toarray()), axis=1
                    )
                batch_contextualized = (
                    torch.tensor(batch_contextualized).float().to(device)
                )
                batch_bow = (
                    torch.tensor(batch_bow.toarray()).float().to(device)
                )
                loss = svi.step(batch_bow, batch_contextualized)
                running_loss += loss / batch_bow.size(0)
            bar.set_postfix(epoch_loss="{:.2e}".format(running_loss))
        self.components_ = self.model.beta()
        return self

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.fit(raw_documents, y, embeddings).transform(
            raw_documents, embeddings
        )
