from functools import partial
from typing import Literal, Optional, Union

import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

from turftopic.base import Encoder
from turftopic.encoders.multimodal import MultimodalEncoder
from turftopic.feature_importance import (
    ctf_idf,
    fighting_words,
    npmi,
    soft_ctf_idf,
)
from turftopic.models.gmm import GMM

FEATURE_IMPORTANCE_METHODS = {
    "soft-c-tf-idf": soft_ctf_idf,
    "c-tf-idf": ctf_idf,
    "fighting-words": fighting_words,
    "npmi": partial(npmi, smoothing=2),
}
LexicalWordImportance = Literal[
    "soft-c-tf-idf",
    "c-tf-idf",
    "npmi",
    "fighting-words",
]


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


class Peax(ClusterMixin, BaseEstimator):
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
        self.gmm_ = GaussianMixture(
            peak_pos.shape[0],
            means_init=peak_pos,
            weights_init=weights,
            random_state=self.random_state,
        )
        self.labels_ = self.gmm_.fit_predict(X)
        self.classes_ = np.sort(np.unique(self.labels_))
        self.means_ = self.gmm_.means_
        return self.labels_

    def predict_proba(self, X):
        return self.gmm_.predict_proba(X)

    def score_samples(self, X):
        return self.density.logpdf(X.T)

    def score(self, X):
        return np.mean(self.score_samples(X))


class Topeax(GMM):
    def __init__(
        self,
        encoder: Union[
            Encoder, str, MultimodalEncoder
        ] = "sentence-transformers/all-MiniLM-L6-v2",
        vectorizer: Optional[CountVectorizer] = None,
        dimensionality_reduction: Optional[TransformerMixin] = None,
        feature_importance: LexicalWordImportance = "soft-c-tf-idf",
        random_state: Optional[int] = None,
    ):
        if dimensionality_reduction is None:
            dimensionality_reduction = TSNE(2, metric="cosine", perplexity=100)
        super().__init__(
            n_components=0,
            encoder=encoder,
            vectorizer=vectorizer,
            dimensionality_reduction=dimensionality_reduction,
            feature_importance=feature_importance,
            random_state=random_state,
        )

    def _init_model(self, n_components: int):
        mixture = Peax()
        return mixture
