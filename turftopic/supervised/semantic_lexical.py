from typing import Literal, Optional, Union

import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import label_binarize, scale

from turftopic.base import ContextualModel, Encoder
from turftopic.feature_importance import fighting_words, semantic_difference
from turftopic.vectorizer import default_vectorizer

NOT_MATCHING_ERROR = (
    "Document embedding dimensionality ({n_dims}) doesn't match term embedding dimensionality ({n_word_dims}). "
    + "Perhaps you are using precomputed embeddings but forgot to pass an encoder to your model. "
    + "Try to initialize the model with the encoder you used for computing the embeddings."
)


class SemanticLexicalAnalysis(ContextualModel):
    """Analyzes groups of texts based on their semantic/lexical differences.

    ```python
    from turftopic import SemanticLexicalAnalysis

    corpus: list[str] = ["some text", "more text", ...]
    labels: list[str] = ["group0", "group1"]

    model = SemanticLexicalAnalysis().fit(corpus, y=labels)
    model.print_topics()
    ```

    Parameters
    ----------
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    """

    def __init__(
        self,
        *,
        encoder: Union[
            Encoder, str
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        random_state: Optional[int] = None,
    ):
        self.encoder = encoder
        if isinstance(encoder, str):
            self.encoder_ = SentenceTransformer(encoder)
        else:
            self.encoder_ = encoder
        if vectorizer is None:
            self.vectorizer = default_vectorizer()
        else:
            self.vectorizer = vectorizer

    def fit_transform(
        self, raw_documents, y, embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        y = np.array(y)
        self.classes_ = np.sort(np.unique(y))
        doc_topic_matrix = label_binarize(y, classes=self.classes_)
        console = Console()
        self.embeddings = embeddings
        with console.status("Fitting model") as status:
            if self.embeddings is None:
                status.update("Encoding documents")
                self.embeddings = self.encoder_.encode(raw_documents)
                console.log("Documents encoded.")
            status.update("Extracting terms.")
            doc_term_matrix = self.vectorizer.fit_transform(raw_documents)
            vocab = self.vectorizer.get_feature_names_out()
            console.log("Term extraction done.")
            status.update("Computing lexical differences.")
            self.lexical_components_ = scale(
                fighting_words(doc_topic_matrix, doc_term_matrix), axis=1
            )
            console.log("Lexical components done.")
            status.update("Encoding vocabulary")
            self.vocab_embeddings = self.encoder_.encode(vocab)
            if self.vocab_embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError(
                    NOT_MATCHING_ERROR.format(
                        n_dims=self.embeddings.shape[1],
                        n_word_dims=self.vocab_embeddings.shape[1],
                    )
                )
            console.log("Vocabulary encoded.")
            status.update("Computing semantic differences.")
            self.semantic_components_ = semantic_difference(
                doc_topic_matrix, self.embeddings, self.vocab_embeddings
            )
            self.components_ = self.semantic_components_
            console.log("Semantic comoponents done.")
            console.log("Model fitting done.")
        return doc_topic_matrix

    def plot_semantic_lexical_square(self, label):
        vocab = self.get_vocab()
        try:
            import plotly.express as px
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        i_component = {lab: i for i, lab in enumerate(self.classes_)}[label]
        # Semantic-lexical compass
        x = self.semantic_components_[i_component]
        y = self.lexical_components_[i_component]
        points = np.array(list(zip(x, y)))
        xx, yy = np.meshgrid(
            np.linspace(np.min(x), np.max(x), 20),
            np.linspace(np.min(y), np.max(y), 20),
        )
        coords = np.array(list(zip(np.ravel(xx), np.ravel(yy))))
        coords = coords + np.random.default_rng(0).normal(
            [0, 0], [0.1, 0.1], size=coords.shape
        )
        dist = euclidean_distances(coords, points)
        idxs = np.argmin(dist, axis=1)
        fig = px.scatter(
            x=x[idxs],
            y=y[idxs],
            text=vocab[idxs],
            template="plotly_white",
        )
        fig = fig.update_traces(
            mode="text", textfont_color="black", marker=dict(color="black")
        ).update_layout(
            xaxis_title="Semantic Importance",
            yaxis_title="Lexical Importance",
        )
        fig = fig.update_layout(
            width=1000,
            height=1000,
            font=dict(family="Times New Roman", color="black", size=21),
            margin=dict(l=5, r=5, t=5, b=5),
        )
        fig = fig.add_hline(y=0, line_color="black", line_width=4)
        fig = fig.add_vline(x=0, line_color="black", line_width=4)
        fig.add_annotation(
            text="Lexical-Semantic",
            x=np.max(x[(x > 0) & (y > 0)]),
            y=np.max(y[(x > 0) & (y > 0)]),
            ax=60,
            ay=-60,
            showarrow=True,
            arrowwidth=3,
            arrowhead=6,
            arrowcolor="black",
            font=dict(size=34, color="black"),
        )
        fig.add_annotation(
            text="Lexical-Nonsemantic",
            x=np.min(x[(x < 0) & (y > 0)]),
            y=np.max(y[(x < 0) & (y > 0)]),
            ax=-60,
            ay=-60,
            showarrow=True,
            arrowwidth=3,
            arrowhead=6,
            arrowcolor="black",
            font=dict(size=34, color="black"),
        )
        fig.add_annotation(
            text="Semantic-Nonlexical",
            x=np.max(x[(x > 0) & (y < 0)]),
            y=np.min(y[(x > 0) & (y < 0)]),
            ax=60,
            ay=60,
            showarrow=True,
            arrowwidth=3,
            arrowhead=6,
            arrowcolor="black",
            font=dict(size=34, color="black"),
        )
        return fig

    def plot_residuals(
        self,
        label,
        independent_variable: Literal["semantic", "lexical"] = "semantic",
    ):
        try:
            import plotly.express as px
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        vocab = self.get_vocab()
        i_component = {lab: i for i, lab in enumerate(self.classes_)}[label]
        semantic_component = self.semantic_components_[i_component]
        lexical_component = self.lexical_components_[i_component]
        if independent_variable == "semantic":
            x, y = semantic_component, lexical_component
        else:
            x, y = lexical_component, semantic_component
        linreg = LinearRegression().fit(x[:, None], y)
        y_pred = linreg.predict(x[:, None])
        residuals = y_pred - y
        absres = np.abs(residuals)
        sorted_res = np.argsort(residuals)
        idxs = [*sorted_res[:100], *sorted_res[-100:]]
        fig = px.scatter(
            x=x,
            y=residuals,
            text=vocab,
            template="plotly_white",
            size=absres,
        )
        for idx in idxs:
            fig.add_annotation(
                text=vocab[idx],
                x=x[idx],
                y=residuals[idx],
                showarrow=False,
                font=dict(size=max(int(8 * np.sqrt(absres[idx])), 8)),
            )
        max_absres = np.max(absres)
        fig.update_yaxes(range=(-max_absres * 1.1, max_absres * 1.1))
        fig.update_traces(mode="text")
        fig.update_traces(
            mode="markers",
            textfont_color="black",
            marker=dict(color="white", line=dict(color="black")),
            hovertemplate="<b>%{text}</b>",
            opacity=1,
        ).update_layout(
            xaxis_title=(
                "Semantic Importance"
                if independent_variable == "semantic"
                else "Lexical Importance"
            ),
            yaxis_title=(
                "Lexical Residual"
                if independent_variable == "semantic"
                else "Semantic Residual"
            ),
        )
        fig.update_layout(
            width=1200,
            height=600,
            font=dict(family="Times New Roman", color="black", size=21),
            margin=dict(l=5, r=5, t=5, b=5),
            hoverlabel=dict(
                bgcolor="white", font_size=24, font_family="Times New Roman"
            ),
        )
        fig.add_hline(y=0, line_color="black", line_width=4)
        return fig

    def _topics_table(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: bool = False,
    ) -> list[list[str]]:
        columns = ["Topic ID"]
        if getattr(self, "topic_names_", None):
            columns.append("Topic Name")
        columns.append("Semantic")
        columns.append("Lexical")
        rows = []
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(self.components_.shape[0]))
        vocab = self.get_vocab()
        for i_topic, (topic_id, sem_component, lex_component) in enumerate(
            zip(classes, self.semantic_components_, self.lexical_components_)
        ):
            semantic = np.argpartition(-sem_component, top_k)[:top_k]
            semantic = semantic[np.argsort(-sem_component[semantic])]
            lexical = np.argpartition(-lex_component, top_k)[:top_k]
            lexical = lexical[np.argsort(-lex_component[lexical])]
            if show_scores:
                concat_semantic = ", ".join(
                    [
                        f"{word}({importance:.2f})"
                        for word, importance in zip(
                            vocab[semantic], sem_component[semantic]
                        )
                    ]
                )
                concat_lexical = ", ".join(
                    [
                        f"{word}({importance:.2f})"
                        for word, importance in zip(
                            vocab[lexical], lex_component[lexical]
                        )
                    ]
                )
            else:
                concat_semantic = ", ".join([word for word in vocab[semantic]])
                concat_lexical = ", ".join([word for word in vocab[lexical]])
            row = [f"{topic_id}"]
            if getattr(self, "topic_names_", None):
                row.append(self.topic_names_[i_topic])
            row.append(concat_semantic)
            row.append(concat_lexical)
            rows.append(row)
        return [columns, *rows]
