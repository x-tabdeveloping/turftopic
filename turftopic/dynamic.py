import itertools
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional, Union

import numpy as np
from sklearn.exceptions import NotFittedError

from turftopic.data import TopicData


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

    def prepare_dynamic_topic_data(
        self,
        corpus: list[str],
        timestamps: list[datetime],
        embeddings: Optional[np.ndarray] = None,
        bins: Union[int, list[datetime]] = 10,
    ):
        """Produces topic inference data for a given corpus, that can be then used and reused.
        Exists to allow visualizations out of the box with topicwizard.

        Parameters
        ----------
        corpus: list of str
            Documents to infer topical content for.
        timestamps: list[datetime]
            Timestamp for each document in `datetime` format.
        embeddings: ndarray of shape (n_documents, n_dimensions)
            Embeddings of documents.
        bins: int or list[datetime], default 10
            Specifies how to bin timestamps in to time slices.
            When an `int`, the corpus will be divided into N equal time slices.
            When a list, it describes the edges of each time slice including the starting
            and final edges of the slices.

            Note: The final edge is not included. You might want to add one day to
            the last bin edge if it equals the last timestamp.

        Returns
        -------
        TopicData
            Information about topical inference in a dictionary.
        """
        if embeddings is None:
            embeddings = self.encode_documents(corpus)
        if getattr(self, "temporal_components_", None) is not None:
            try:
                document_topic_matrix = self.transform(
                    corpus, embeddings=embeddings
                )
            except (AttributeError, NotFittedError):
                document_topic_matrix = self.fit_transform_dynamic(
                    corpus,
                    timestamps=timestamps,
                    embeddings=embeddings,
                    bins=bins,
                )
        else:
            document_topic_matrix = self.fit_transform_dynamic(
                corpus, timestamps=timestamps, embeddings=embeddings, bins=bins
            )
        dtm = self.vectorizer.transform(corpus)  # type: ignore
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(self.components_.shape[0]))
        res = TopicData(
            corpus=corpus,
            document_term_matrix=dtm,
            vocab=self.get_vocab(),
            document_topic_matrix=document_topic_matrix,
            document_representation=embeddings,
            topic_term_matrix=self.components_,  # type: ignore
            transform=getattr(self, "transform", None),
            topic_names=self.topic_names,
            classes=classes,
            temporal_components=self.temporal_components_,
            temporal_importance=self.temporal_importance_,
            time_bin_edges=self.time_bin_edges,
        )
        return res
