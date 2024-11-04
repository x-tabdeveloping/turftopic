import math
from typing import Optional, Union

import numpy as np
import torch
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel, Encoder
from turftopic.models._fastopic import fastopic
from turftopic.vectorizer import default_vectorizer


class FASTopic(ContextualModel):
    """
    Implementation of the FASTopic model with a Turftopic API.
    The implementation is based on the [original FASTopic package](https://github.com/BobXWu/FASTopic/tree/master),
    but is adapted for optimal use in Turftopic (you can pre-compute embeddings for instance).

    You will need to install torch to use this model.

    ```bash
    pip install turftopic[torch]
    ## OR:
    pip install turftopic[pyro-ppl]
    ```

    ```python
    from turftopic import FASTopic

    corpus: list[str] = ["some text", "more text", ...]

    model = FASTopic(10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int
        Number of topics. If you're using priors on the weight,
        feel free to overshoot with this value.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    DT_alpha: float, default 3.0
        Sinkhorn alpha between document embeddings and topic embeddings.
    TW_alpha: float, default 2.0
        Sinkhorn alpha between topic embeddings and word embeddings.
    theta_temp: float, default 1.0
        Temperature parameter of used in softmax to compute topic probabilities in documents.
    n_epochs: int, default 200
        Number of epochs to train the model for.
    learning_rate: float, default 0.002
        Learning rate for the ADAM optimizer.
    device: str, default "cpu"
        Device to run the model on. Defaults to CPU.
    """

    def __init__(
        self,
        n_components: int,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        DT_alpha: float = 3.0,
        TW_alpha: float = 2.0,
        theta_temp: float = 1.0,
        n_epochs: int = 200,
        learning_rate: float = 0.002,
        device: str = "cpu",
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.random_state = random_state
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.DT_alpha = DT_alpha
        self.TW_alpha = TW_alpha
        self.theta_temp = theta_temp
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.model = fastopic(n_components, theta_temp, DT_alpha, TW_alpha)
        self.batch_size = batch_size

    def make_optimizer(self, learning_rate: float):
        args_dict = {
            "params": self.model.parameters(),
            "lr": learning_rate,
        }
        optimizer = torch.optim.Adam(**args_dict)
        return optimizer

    def _train_model(self, embeddings, document_term_matrix, status):
        self.model.init(document_term_matrix.shape[1], embeddings.shape[1])
        self.model = self.model.to(self.device)
        optimizer = self.make_optimizer(self.learning_rate)
        self.model.train()
        if self.batch_size is None:
            batch_size = embeddings.shape[0]
        else:
            batch_size = self.batch_size
        num_batches = int(
            math.ceil(document_term_matrix.shape[0] / batch_size)
        )
        for epoch in range(self.n_epochs):
            running_loss = 0
            for i in range(num_batches):
                batch_bow = np.atleast_2d(
                    document_term_matrix[
                        i * batch_size : (i + 1) * batch_size, :
                    ].toarray()
                )
                # Skipping batches that are smaller than 2
                if batch_bow.shape[0] < 2:
                    continue
                batch_contextualized = np.atleast_2d(
                    embeddings[i * batch_size : (i + 1) * batch_size, :]
                )
                batch_contextualized = (
                    torch.tensor(batch_contextualized).float().to(self.device)
                )
                batch_bow = torch.tensor(batch_bow).float().to(self.device)
                rst_dict = self.model(batch_bow, batch_contextualized)
                batch_loss = rst_dict["loss"]
                running_loss += batch_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            status.update(
                f"Fitting model. Epoch [{epoch}/{self.n_epochs}], Loss [{running_loss}]"
            )
        self.components_ = self.model.get_beta().detach().numpy()

    def fit_transform(
        self, raw_documents, y=None, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        console = Console()
        with console.status("Fitting model") as status:
            if embeddings is None:
                status.update("Encoding documents")
                embeddings = self.encoder_.encode(raw_documents)
                console.log("Documents encoded.")
            self.train_doc_embeddings = embeddings
            status.update("Extracting terms.")
            document_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            status.update("Fitting model")
            self._train_model(embeddings, document_term_matrix, status)
            console.log("Model fitting done.")
            document_topic_matrix = self.transform(
                raw_documents, embeddings=embeddings
            )
        return document_topic_matrix

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
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
        with torch.no_grad():
            self.model.eval()
            theta = self.model.get_theta(
                torch.as_tensor(embeddings),
                torch.as_tensor(self.train_doc_embeddings),
            )
            theta = theta.detach().cpu().numpy()
        return theta
