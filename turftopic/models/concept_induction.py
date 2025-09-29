from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from rich.progress import track
from sklearn.feature_extraction.text import CountVectorizer

from turftopic._figure import HTMLFigure
from turftopic.analyzers.base import Analyzer
from turftopic.base import Encoder
from turftopic.encoders.multimodal import MultimodalEncoder
from turftopic.models._concept_induction import (
    HTML_WRAPPER,
    STYLE,
    render_models,
)
from turftopic.models.keynmf import KeyNMF


class ConceptInduction:
    """Class for producing high-level concepts from different angles about the same corpus.
    We use a multitude of seeded KeyNMF models to produce topics conditioned
    on the provided aspects that are then analyzed by large-language models.

    Parameters
    ----------
    seed_phrases: list[str]
        List of phrases/aspects to use as seeds for the analysis.
    analyzer: Analyzer
        Large language model used to analyze the concepts produced by KeyNMF.
    n_components: int, default 4
        Number of concepts/topics to create for each aspect.
        We recommend that you stick to a lower number both because of diminishing
        usefulness but also due to lower cost.
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    top_n: int, default 25
        Number of keywords to extract for each document.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.
    metric: "cosine" or "dot", default "cosine"
        Similarity metric to use for keyword extraction.
    seed_exponent: float, default 2.0
        Exponent that is applied to document weight in relation to the provided seed phrase.
    cross_lingual: bool, default False
        Indicates whether KeyNMF should match terms across languages.
        This is useful when you have a corpus containing multiple languages.
    term_match_threshold: float, default 0.9
        Cosine similarity threshold for matching terms across languages.
    """

    def __init__(
        self,
        seed_phrases: list[str],
        analyzer: Analyzer,
        n_components: int = 4,
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        top_n: int = 25,
        random_state: Optional[int] = None,
        metric: Literal["cosine", "dot"] = "cosine",
        seed_exponent: float = 2,
        cross_lingual: bool = False,
        term_match_threshold: float = 0.9,
    ):
        self.seed_phrases = seed_phrases
        self.analyzer = analyzer
        self.n_components = n_components
        self.encoder = encoder
        self.vectorizer = vectorizer
        self.top_n = top_n
        self.random_state = random_state
        self.metric = metric
        self.seed_exponent = seed_exponent
        self.cross_lingual = cross_lingual
        self.term_match_threshold = term_match_threshold

    def fit(
        self,
        raw_documents,
        y=None,
        embeddings: Optional[np.ndarray] = None,
    ):
        self.models = []
        doc_topic_matrices = []
        console = Console()
        for seed in track(
            self.seed_phrases,
            description="Running KeyNMF with all seed phrases.",
        ):
            console.log(f"Initializing model with seed {seed}")
            model = KeyNMF(
                self.n_components,
                encoder=self.encoder,
                seed_phrase=seed,
                vectorizer=self.vectorizer,
            )
            if embeddings is None:
                console.log("Embedding documents.")
                embeddings = model.encode_documents(raw_documents)
            with console.capture() as capture:
                doc_top = model.fit_transform(
                    raw_documents, embeddings=embeddings
                )
            console.log(capture.get())
            console.log("Model fit.")
            doc_topic_matrices.append(doc_top)
            self.models.append(model)
        self.document_topic_matrices = np.stack(doc_topic_matrices)
        self.analysis_results = []
        for model in self.models:
            _res = model.analyze_topics(self.analyzer)
            self.analysis_results.append(_res)
        console.log("Analysis complete.")
        return self

    def _render_html(self, display_summaries: bool) -> str:
        html = HTML_WRAPPER.format(
            style=STYLE,
            body_content=render_models(
                self.models, display_summaries=display_summaries
            ),
        )
        return html

    def plot_concepts(self, display_summaries: bool = True) -> HTMLFigure:
        """Plots concepts in an interactive explorer.

        Parameters
        ----------
        display_summaries: bool, default True
            Displays summaries when they are available (if analyzer.use_summaries == True)
        """
        return HTMLFigure(
            self._render_html(display_summaries=display_summaries)
        )
