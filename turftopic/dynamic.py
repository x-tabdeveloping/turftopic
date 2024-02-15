from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Union

import numpy as np


def bin_timestamps(
    timestamps: list[datetime],
    bins: Union[int, list[datetime]] = 10,
) -> tuple[np.ndarray, list[datetime]]:
    if not timestamps and not isinstance(timestamps[0], datetime):
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
