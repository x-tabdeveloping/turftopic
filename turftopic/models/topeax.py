from typing import Optional, Union

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    generate_binary_structure,
    maximum_filter,
)
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _estimate_gaussian_parameters,
)

from turftopic.base import Encoder
from turftopic.encoders.multimodal import MultimodalEncoder
from turftopic.models.gmm import GMM, LexicalWordImportance


def detect_peaks(image):
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 25)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # we create the mask of the background
    background = image == 0
    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks


class FixedMeanGaussianMixture(GaussianMixture):
    def _m_step(self, X, log_resp):
        # Skipping mean update
        self.weights_, _, self.covariances_ = _estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )


class Peax(ClusterMixin, BaseEstimator):
    """Clustering model based on density peaks.

    Parameters
    ----------
    random_state: int, default None
        Random seed to use for fitting gaussian mixture to peaks.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def fit(self, X, y=None):
        self.X_range = np.min(X), np.max(X)
        self.density = gaussian_kde(X.T, "scott")
        coord = np.linspace(*self.X_range, num=100)
        z = []
        for yval in coord:
            points = np.stack([coord, np.full(coord.shape, yval)]).T
            prob = np.exp(self.density.logpdf(points.T))
            z.append(prob)
        z = np.stack(z)
        peaks = detect_peaks(z.T)
        peak_ind = np.nonzero(peaks)
        peak_pos = np.stack([coord[peak_ind[0]], coord[peak_ind[1]]]).T
        weights = self.density.pdf(peak_pos.T)
        weights = weights / weights.sum()
        self.gmm_ = FixedMeanGaussianMixture(
            peak_pos.shape[0],
            means_init=peak_pos,
            weights_init=weights,
            random_state=self.random_state,
        )
        self.labels_ = self.gmm_.fit_predict(X)
        # Checking whether there are close to zero components
        is_zero = np.isclose(self.gmm_.weights_, 0)
        n_zero = np.sum(is_zero)
        if n_zero > 0:
            print(
                f"{n_zero} components have zero weight, removing them and refitting."
            )
        peak_pos = peak_pos[~is_zero]
        weights = self.gmm_.weights_[~is_zero]
        weights = weights / weights.sum()
        self.gmm_ = FixedMeanGaussianMixture(
            peak_pos.shape[0],
            means_init=peak_pos,
            weights_init=weights,
            random_state=self.random_state,
        )
        self.labels_ = self.gmm_.fit_predict(X)
        self.classes_ = np.sort(np.unique(self.labels_))
        self.means_ = self.gmm_.means_
        self.weights_ = self.gmm_.weights_
        self.covariances_ = self.gmm_.covariances_
        return self.labels_

    @property
    def n_components(self) -> int:
        return self.gmm_.n_components

    def predict_proba(self, X):
        return self.gmm_.predict_proba(X)

    def score_samples(self, X):
        return self.density.logpdf(X.T)

    def score(self, X):
        return np.mean(self.score_samples(X))


class Topeax(GMM):
    """Topic model based on the Peax clustering algorithm.
    The algorithm discovers the number of topics automatically, and is based on GMM.

    Parameters
    ----------
    encoder: str or SentenceTransformer
        Model to encode documents/terms, all-MiniLM-L6-v2 is the default.
    vectorizer: CountVectorizer, default None
        Vectorizer used for term extraction.
        Can be used to prune or filter the vocabulary.
    perplexity: int, default 50
        Number of neighbours to take into account when running TSNE.
    random_state: int, default None
        Random state to use so that results are exactly reproducible.

    """

    def __init__(
        self,
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        perplexity: int = 50,
        random_state: Optional[int] = None,
    ):
        dimensionality_reduction = TSNE(
            2,
            metric="cosine",
            perplexity=perplexity,
            random_state=random_state,
        )
        self.perplexity = perplexity
        super().__init__(
            n_components=0,
            encoder=encoder,
            vectorizer=vectorizer,
            dimensionality_reduction=dimensionality_reduction,
            random_state=random_state,
        )

    def estimate_components(
        self,
        feature_importance: Optional[LexicalWordImportance] = None,
        doc_topic_matrix=None,
        doc_term_matrix=None,
    ) -> np.ndarray:
        doc_topic_matrix = (
            doc_topic_matrix
            if doc_topic_matrix is not None
            else self.doc_topic_matrix
        )
        doc_term_matrix = (
            doc_term_matrix
            if doc_term_matrix is not None
            else self.doc_term_matrix
        )
        lexical_components = super().estimate_components(
            "npmi", doc_topic_matrix, doc_term_matrix
        )
        vocab = self.get_vocab()
        if getattr(self, "vocab_embeddings", None) is None or (
            self.vocab_embeddings.shape[0] != vocab.shape[0]
        ):
            self.vocab_embeddings = self.encode_documents(vocab)
        topic_embeddings = []
        for weight in doc_topic_matrix.T:
            topic_embeddings.append(
                np.average(self.embeddings, axis=0, weights=weight)
            )
        self.topic_embeddings = np.stack(topic_embeddings)
        semantic_components = cosine_similarity(
            self.topic_embeddings, self.vocab_embeddings
        )
        # Transforming to positive values from 0 to 1
        # Then taking geometric average of the two values
        self.components_ = np.sqrt(
            ((1 + lexical_components) / 2) * ((1 + semantic_components) / 2)
        )
        return self.components_

    def _init_model(self, n_components: int):
        mixture = Peax()
        return mixture

    def plot_steps(self, hover_text=None):
        try:
            import plotly.express as px
            from plotly.subplots import make_subplots
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        dens_3d = self.plot_density_3d()
        component_plot = self.plot_components(
            show_points=True, hover_text=hover_text
        )
        points_plot = px.scatter(
            x=self.reduced_embeddings[:, 0],
            y=self.reduced_embeddings[:, 1],
            template="plotly_white",
        )
        points_plot = points_plot.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
        )
        points_plot = points_plot.update_traces(
            marker=dict(
                color="#B7B7FF",
                size=6,
                opacity=0.5,
                line=dict(color="#01014B", width=2),
            )
        )
        colormap = {
            name: color
            for name, color in zip(
                self.topic_names, px.colors.qualitative.Dark24
            )
        }
        bar = px.bar(
            y=self.topic_names,
            x=self.weights_,
            template="plotly_white",
            color_discrete_map=colormap,
            color=self.topic_names,
            text=[f"{p:.2f}" for p in self.weights_],
        )
        bar = bar.update_traces(
            marker_line_color="black",
            marker_line_width=1.5,
            opacity=0.8,
        )

        def update_annotation(a):
            name = a.text.removeprefix("<b>").split("<")[0]
            return a.update(
                # text=name,
                font=dict(size=8, color=colormap[name]),
                arrowsize=1,
                arrowhead=1,
                arrowwidth=1,
                bgcolor=None,
                opacity=0.7,
                # bgcolor=colormap[name],
                bordercolor=colormap[name],
                borderwidth=0,
            )

        fig = make_subplots(
            horizontal_spacing=0.0,
            vertical_spacing=0.1,
            rows=2,
            cols=2,
            subplot_titles=[
                "t-SN Embeddings",
                "Peaks in Kernel Density Estimate",
                "Gaussian Mixture Approximation",
                "Component Probabilities",
            ],
            specs=[
                [
                    {"type": "xy"},
                    {"type": "surface"},
                ],
                [
                    {"type": "xy"},
                    {"type": "bar"},
                ],
            ],
        )
        for i, sub in enumerate([points_plot, dens_3d, component_plot, bar]):
            row = i // 2
            col = i % 2
            for trace in sub.data:
                fig.add_trace(trace, row=row + 1, col=col + 1)
            for shape in sub.layout.shapes:
                fig.add_shape(shape, row=row + 1, col=col + 1)
        fig = fig.update_layout(
            template="plotly_white",
            font=dict(family="Merriweather", size=14, color="black"),
            width=1200,
            height=800,
            autosize=False,
            margin=dict(r=0, l=0, t=40, b=0),
        )
        fig = fig.update_scenes(
            annotations=[
                update_annotation(annotation)
                for annotation in dens_3d.layout.scene.annotations
            ],
            col=2,
            row=1,
        )
        fig = fig.for_each_annotation(lambda a: a.update(yshift=0))
        fig = fig.update_yaxes(visible=False, row=2, col=2)
        fig = fig.update_xaxes(
            title=dict(text="$P(z)$", font=dict(size=16)), row=2, col=2
        )
        return fig
