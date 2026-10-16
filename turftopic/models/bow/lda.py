from typing import Optional

import numpy as np
from rich.console import Console
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from turftopic.base import ContextualModel
from turftopic.vectorizers.default import default_vectorizer


class LDA(ContextualModel):
    """Turftopic wrapper around sklearns's implementation of Latent Dirichlet Allocation.
    Note that LDA is a Bag-of-words model and NOT a contextual model and uses static word representations!
    The encoder and passed embeddings will be ignored.
    For interpretation of hyperparameters refer to [sklearn's documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)

    ```python
    from turftopic import LDA
    corpus: list[str] = ["some text", "more text", ...]

    model = LDA(10).fit(corpus)
    model.print_topics()
    ```

    Parameters
    ----------
    n_components: int
        Number of topics.
    encoder: None
        Ignored.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    trf_kwargs: dict, default None
        Ignored.
    encode_kwargs: dict, default None
        Ignored.

    """

    def __init__(
        self,
        n_components: int,
        encoder=None,
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
        doc_topic_prior=None,
        topic_word_prior=None,
        learning_method="batch",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=10,
        batch_size=128,
        evaluate_every=-1,
        total_samples=1000000.0,
        perp_tol=0.1,
        mean_change_tol=0.001,
        max_doc_update_iter=100,
        n_jobs=None,
        verbose=0,
        trf_kwargs=None,
        encode_kwargs=None,
    ):
        self.n_components = n_components
        self.encoder = encoder
        self.encode_kwargs = encode_kwargs
        self.trf_kwargs = trf_kwargs
        self.random_state = random_state
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.learning_offset = learning_offset
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.total_samples = total_samples
        self.perp_tol = perp_tol
        self.mean_change_tol = mean_change_tol
        self.max_doc_update_iter = max_doc_update_iter
        self.n_jobs = n_jobs
        self.verbose = verbose

    @property
    def components_(self):
        return self.model_.components_

    def fit_transform(
        self,
        raw_documents,
        y=None,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        console = Console()
        with console.status("Fitting model") as status:
            status.update("Extracting terms.")
            self.doc_term_matrix = self.vectorizer.fit_transform(raw_documents)
            console.log("Term extraction done.")
            status.update("Fitting the model")
            self.model_ = LatentDirichletAllocation(
                self.n_components,
                random_state=self.random_state,
                doc_topic_prior=self.doc_topic_prior,
                topic_word_prior=self.topic_word_prior,
                learning_method=self.learning_method,
                learning_decay=self.learning_decay,
                learning_offset=self.learning_offset,
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                evaluate_every=self.evaluate_every,
                total_samples=self.total_samples,
                perp_tol=self.perp_tol,
                mean_change_tol=self.mean_change_tol,
                max_doc_update_iter=self.max_doc_update_iter,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
            self.doc_topic_matrix = self.model_.fit_transform(
                self.doc_term_matrix
            )
            console.log("Model fitting done.")
            self.top_documents = self.get_top_documents(
                raw_documents, document_topic_matrix=self.doc_topic_matrix
            )
        return self.doc_topic_matrix

    def transform(
        self, raw_documents, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Infers topic importances for new documents based on a fitted model.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        embeddings: None
            Ignored.

        Returns
        -------
        ndarray of shape (n_dimensions, n_topics)
            Document-topic matrix.
        """
        doc_term_matrix = self.vectorizer.transform(raw_documents)
        return self.model_.transform(doc_term_matrix)
