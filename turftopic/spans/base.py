from abc import ABC, abstractmethod
from typing import Iterable

Offsets = list[tuple[int, int]]


class SpanExtractor(ABC):
    @abstractmethod
    def extract_spans(self, document: str) -> Offsets:
        """Extracts span offsets in a document.

        Parameters
        ----------
        document: str
            Text document.

        Returns
        -------
        list[tuple[int, int]]
            Tuples of start and end characters for each extracted span.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def batch_extract_spans(self, documents: list[str]) -> list[Offsets]:
        """Extracts span offsets in each document.

        Parameters
        ----------
        document: list[str]
            List of documents.

        Returns
        -------
        list[list[tuple[int, int]]]
            Tuples of start and end characters for each extracted span for each document.
        """
        res = []
        for doc in documents:
            res.append(self.extract_spans(doc))
        return res

    def stream_spans(self, documents: list[str]) -> Iterable[str]:
        """Streams text spans from a list of documents.
        This can be useful for instance for labelling topics with highest ranking sentences.
        """
        for doc in documents:
            for span_start, span_end in self.extract_spans(doc):
                yield doc[span_start:span_end]
