from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Union

import numpy as np
from rich.console import Console
from rich.table import Table


def bin_timestamps(
    timestamps: list[datetime],
    bins: Union[int, list[datetime]] = 10,
) -> tuple[np.ndarray, list[datetime]]:
    if not len(timestamps) and not isinstance(timestamps[0], datetime):
        raise TypeError("Timestamps have to be `datetime` objects.")
    unix_timestamps = [timestamp.timestamp() for timestamp in timestamps]
    if isinstance(bins, list):
        unix_bins = [bin.timestamp() for bin in bins]
        return np.digitize(unix_timestamps, unix_bins), bins
    else:
        unix_bins = np.histogram_bin_edges(unix_timestamps, bins=bins)
        bins = [datetime.fromtimestamp(ts) for ts in unix_bins]
        return np.digitize(unix_timestamps, unix_bins), bins


class DynamicTopicModel(ABC):
    @abstractmethod
    def fit_transform_dynamic(
        self,
        raw_documents,
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
    ) -> np.ndarray:
        pass

    def fit_dynamic(
        self,
        raw_documents,
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
    ):
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
        temporal_components = self.temporal_components_
        slices = self.get_time_slices()
        slice_names = []
        for start_dt, end_dt in slices:
            start_str = start_dt.strftime(date_format)
            end_str = end_dt.strftime(date_format)
            slice_names.append(f"{start_str} - {end_str}")
        table = Table(show_lines=True)
        table.add_column("Time Slice")
        for i_topic in range(self.temporal_components_.shape[1]):
            table.add_column(f"Topic {i_topic}")
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
            table.add_row(*fields)
        console = Console()
        console.print(table)
