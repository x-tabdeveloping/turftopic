from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import numpy as np
from rich.console import Console
from rich.table import Table

from turftopic.utils import export_table


def bin_timestamps(
    timestamps: list[datetime],
    bins: Union[int, list[datetime]] = 10,
) -> tuple[np.ndarray, list[datetime]]:
    if not len(timestamps) and not isinstance(timestamps[0], datetime):
        raise TypeError("Timestamps have to be `datetime` objects.")
    unix_timestamps = [timestamp.timestamp() for timestamp in timestamps]
    if isinstance(bins, list):
        if min(timestamps) < min(bins):
            raise ValueError(
                f"Earliest timestamp ({min(timestamps)}) is not later or the same as first bin edge ({min(bins)})."
            )
        if max(timestamps) >= max(bins):
            raise ValueError(
                f"Latest timestamp ({max(timestamps)}) is not earlier than last bin edge ({max(bins)})."
            )
        unix_bins = [bin.timestamp() for bin in bins]
        # Have to substract one, else it starts from one
        return np.digitize(unix_timestamps, unix_bins) - 1, bins
    else:
        # Adding one day, so that the maximum value is still included.
        max_timestamp = max(timestamps) + timedelta(days=1)
        unix_bins = np.histogram_bin_edges(unix_timestamps, bins=bins)
        unix_bins[-1] = max_timestamp.timestamp()
        bins = [datetime.fromtimestamp(ts) for ts in unix_bins]
        # Have to substract one, else it starts from one
        return np.digitize(unix_timestamps, unix_bins) - 1, bins


class DynamicTopicModel(ABC):
    @staticmethod
    def bin_timestamps(
        timestamps: list[datetime], bins: Union[int, list[datetime]] = 10
    ) -> tuple[np.ndarray, list[datetime]]:
        """Bins timestamps based on given bins.

        Parameters
        ----------
        timestamps: list[datetime]
            List of timestamps for documents.
        bins: int or list[datetime], default 10
            Time bins to use.
            If the bins are an int (N), N equally sized bins are used.
            Otherwise they should be bin edges, including the last and first edge.
            Bins are inclusive at the lower end and exclusive at the upper (lower <= timestamp < upper).

        Returns
        -------
        time_labels: ndarray of int
            Labels for time slice in each document.
        bin_edges: list[datetime]
            List of edges for time bins.
        """
        return bin_timestamps(timestamps, bins)

    @abstractmethod
    def fit_transform_dynamic(
        self,
        raw_documents,
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
    ) -> np.ndarray:
        """Fits a dynamic topic model on the corpus and returns document-topic-importances.

        Parameters
        ----------
        raw_documents
            Documents to fit the model on.
        timestamps: list[datetime]
            Timestamp for each document in `datetime` format.
        embeddings: np.ndarray, default None
            Document embeddings produced by an embedding model.
        bins: int or list[datetime], default 10
            Specifies how to bin timestamps in to time slices.
            When an `int`, the corpus will be divided into N equal time slices.
            When a list, it describes the edges of each time slice including the starting
            and final edges of the slices.

        Returns
        -------
        ndarray of shape (n_documents, n_topics)
            Document-topic importance matrix.
        """
        pass

    def fit_dynamic(
        self,
        raw_documents,
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
    ):
        """Fits a dynamic topic model on the corpus and returns document-topic-importances.

        Parameters
        ----------
        raw_documents
            Documents to fit the model on.
        timestamps: list[datetime]
            Timestamp for each document in `datetime` format.
        embeddings: np.ndarray, default None
            Document embeddings produced by an embedding model.
        bins: int or list[datetime], default 10
            Specifies how to bin timestamps in to time slices.
            When an `int`, the corpus will be divided into N equal time slices.
            When a list, it describes the edges of each time slice including the starting
            and final edges of the slices.

            Note: The final edge is not included. You might want to add one day to
            the last bin edge if it equals the last timestamp.
        """
        self.fit_transform_dynamic(raw_documents, timestamps, embeddings, bins)
        return self

    def get_time_slices(self) -> list[tuple[datetime, datetime]]:
        """Returns starting and ending datetime of
        each timeslice in the model."""
        bins = self.time_bin_edges
        res = []
        for i_bin, slice_end in enumerate(bins[1:]):
            res.append((bins[i_bin], slice_end))
        return res

    def get_topics_over_time(
        self, top_k: int = 10
    ) -> list[list[tuple[Any, list[tuple[str, float]]]]]:
        """Returns high-level topic representations in form of the top K words
        in each topic.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.

        Returns
        -------
        list[list[tuple]]
            List of topics over each time slice in the dynamic model.
            Each time slice is a list of topics.
            Each topic is a tuple of topic ID and the top k words.
            Top k words are a list of (word, word_importance) pairs.
        """
        n_topics = self.temporal_components_.shape[1]
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(n_topics))
        res = []
        for components in self.temporal_components_:
            highest = np.argpartition(-components, top_k)[:, :top_k]
            vocab = self.get_vocab()
            top = []
            score = []
            for component, high in zip(components, highest):
                importance = component[high]
                high = high[np.argsort(-importance)]
                score.append(component[high])
                top.append(vocab[high])
            topics = []
            for topic, words, scores in zip(classes, top, score):
                topic_data = (topic, list(zip(words, scores)))
                topics.append(topic_data)
            res.append(topics)
        return res

    def _topics_over_time(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        date_format: str = "%Y %m %d",
    ) -> list[list[str]]:
        temporal_components = self.temporal_components_
        slices = self.get_time_slices()
        slice_names = []
        for start_dt, end_dt in slices:
            start_str = start_dt.strftime(date_format)
            end_str = end_dt.strftime(date_format)
            slice_names.append(f"{start_str} - {end_str}")
        n_topics = self.temporal_components_.shape[1]
        try:
            topic_names = self.topic_names
        except AttributeError:
            topic_names = [f"Topic {i}" for i in range(n_topics)]
        columns = []
        rows = []
        columns.append("Time Slice")
        for topic in topic_names:
            columns.append(topic)
        for slice_name, components in zip(slice_names, temporal_components):
            fields = []
            fields.append(slice_name)
            highest = np.argpartition(-components, top_k)[:, :top_k]
            vocab = self.get_vocab()
            for component, high in zip(components, highest):
                if np.all(component == 0) or np.all(np.isnan(component)):
                    fields.append("Topic not present.")
                    continue
                importance = component[high]
                high = high[np.argsort(-importance)]
                high = high[importance != 0]
                scores = component[high]
                words = vocab[high]
                if show_scores:
                    concat_words = ", ".join(
                        [
                            f"{word}({importance:.2f})"
                            for word, importance in zip(words, scores)
                        ]
                    )
                else:
                    concat_words = ", ".join([word for word in words])
                fields.append(concat_words)
            rows.append(fields)
        return [columns, *rows]

    def print_topics_over_time(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        date_format: str = "%Y %m %d",
    ):
        """Pretty prints topics in the model in a table.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        """
        columns, *rows = self._topics_over_time(
            top_k, show_scores, date_format
        )
        table = Table(show_lines=True)
        for column in columns:
            table.add_column(column)
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)

    def export_topics_over_time(
        self,
        top_k: int = 5,
        show_scores: bool = False,
        date_format: str = "%Y %m %d",
        format="csv",
    ) -> str:
        """Pretty prints topics in the model in a table.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        format: 'csv', 'latex' or 'markdown'
            Specifies which format should be used.
            'csv', 'latex' and 'markdown' are supported.
        """
        table = self._topics_over_time(top_k, show_scores, date_format)
        return export_table(table, format=format)

    def plot_topics_over_time(self, top_k: int = 6):
        """Displays topics over time in the fitted dynamic model on a dynamic HTML figure.

        > You will need to `pip install plotly` to use this method.

        Parameters
        ----------
        top_k: int, default 6
            Number of top words per topic to display on the figure.

        Returns
        -------
        go.Figure
            Plotly graph objects Figure, that can be displayed or exported as
            HTML or static image.
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        fig = go.Figure()
        vocab = self.get_vocab()
        n_topics = self.temporal_components_.shape[1]
        try:
            topic_names = self.topic_names
        except AttributeError:
            topic_names = [f"Topic {i}" for i in range(n_topics)]
        for i_topic, topic_imp_t in enumerate(self.temporal_importance_.T):
            component_over_time = self.temporal_components_[:, i_topic, :]
            name_over_time = []
            for component in component_over_time:
                high = np.argpartition(-component, top_k)[:top_k]
                values = component[high]
                if np.all(values == 0) or np.all(np.isnan(values)):
                    name_over_time.append("<not present>")
                    continue
                high = high[np.argsort(-values)]
                name_over_time.append(", ".join(vocab[high]))
            times = self.time_bin_edges[:-1]
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=topic_imp_t,
                    mode="markers+lines",
                    text=name_over_time,
                    name=topic_names[i_topic],
                    hovertemplate="<b>%{text}</b>",
                    marker=dict(
                        line=dict(width=2, color="black"),
                        size=14,
                    ),
                    line=dict(width=3),
                )
            )
        fig.update_layout(
            template="plotly_white",
            hoverlabel=dict(font_size=16, bgcolor="white"),
            hovermode="x",
        )
        fig.update_xaxes(title="Time Slice Start")
        fig.update_yaxes(title="Topic Importance")
        return fig
