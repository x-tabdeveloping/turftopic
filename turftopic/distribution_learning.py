import itertools
import numpy as np
from conjugate.distributions import NormalInverseGamma
from conjugate.models import normal


class GaussianDistributionLearner:
    """Learns posterior distribution of the mean and variance of a bunch of
    (Non-multivariate) Gaussian distributions using Bayesian updating.

    This is very useful for when you cannot keep a dataset in memory and
    want to learn the importance of topics in the dataset from batches with uncertainty.
    """

    def __init__(self, mu=0, alpha=1, beta=1, nu=1):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.posteriors = []

    def init_prior(self):
        return NormalInverseGamma(
            mu=self.mu, alpha=self.alpha, beta=self.beta, nu=self.nu
        )

    def update(self, batch_doc_topic):
        """Updates posterior distributions based on the incoming batch."""
        for i_topic, dt in enumerate(batch_doc_topic.T):
            if i_topic >= len(self.posteriors):
                self.posteriors.append(self.init_prior())
            prior = self.posteriors[i_topic]
            posterior = normal(
                x_total=np.sum(dt),
                x2_total=np.sum(np.square(dt)),
                n=dt.shape[0],
                prior=prior,
            )
            self.posteriors[i_topic] = posterior

    def sample_means(self, n_datapoints=100, random_state=None):
        """Samples means from each of the learned posteriors.

        Parameters
        ----------
        n_datapoints: int, default 100
            Number of datapoints to sample from the posterior.
        random_state: int or None, default None
            Random seed to use for sampling.

        Returns
        -------
        ndarray of shape (n_topics, n_datapoints)
            Posterior samples for each topic.
        """
        out = []
        for posterior in self.posteriors:
            out.append(posterior.sample_mean(size=n_datapoints))
        return np.stack(out)

    def plot_topic_distribution(
        self, topic_names: list[str] | None = None, sort_topics=True
    ):
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        fig = go.Figure()
        if topic_names is None:
            topic_names = [f"Topic {i}" for i in range(len(self.posteriors))]
        if len(topic_names) != len(self.posteriors):
            raise ValueError(
                "The number of posteriors learned by the distribution learner is not the same as the number of topic names given."
            )
        mus = self.sample_means()
        y = mus.mean(axis=1)
        se = np.std(mus, axis=1)
        topic_colors = list(
            itertools.islice(
                itertools.cycle(px.colors.qualitative.Dark24),
                len(self.posteriors),
            )
        )
        if sort_topics:
            order = np.argsort(y)
        else:
            order = np.arange(len(self.posteriors))
        for i_topic in order:
            fig.add_bar(
                y0=topic_names[i_topic],
                x=[y[i_topic]],
                error_x=dict(
                    type="data",
                    array=[se[i_topic]],
                    visible=True,
                ),
                showlegend=False,
                name=topic_names[i_topic],
                marker=dict(
                    line=dict(color=topic_colors[i_topic], width=2),
                    color="white",
                ),
            )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Roboto Mono", color="black", size=10),
        )
        return fig
