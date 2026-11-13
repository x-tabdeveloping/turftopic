from abc import ABC, abstractmethod
from typing import Iterable

Offsets = list[tuple[int, int]]


class SpanExtractor(ABC):
    @abstractmethod
    def extract_spans(
        self, document: str, token_offsets: Offsets | None = None
    ) -> Offsets:
        """Extracts span offsets in a document.

        Parameters
        ----------
        document: str
            Text document.
        token_offsets: list[Offsets], default None
            Offsets for the tokens in the document produced by a tokenizer (for instance from the encoder).
            Not all SpanExtractors make use of this information.

        Returns
        -------
        list[tuple[int, int]]
            Tuples of start and end characters for each extracted span.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def batch_extract_spans(
        self, documents: list[str], token_offsets: list[Offsets] | None = None
    ) -> list[Offsets]:
        """Extracts span offsets in each document.

        Parameters
        ----------
        document: list[str]
            List of documents.
        token_offsets: list[Offsets], default None
            Offsets for the tokens in each document produced by a tokenizer (for instance from the encoder).
            Not all SpanExtractors make use of this information.

        Returns
        -------
        list[list[tuple[int, int]]]
            Tuples of start and end characters for each extracted span for each document.
        """
        if token_offsets is None:
            token_offsets = [None] * len(documents)
        res = []
        for doc, offs in zip(documents, token_offsets):
            res.append(self.extract_spans(doc, token_offsets=offs))
        return res

    def stream_spans(
        self, documents: list[str], token_offsets: list[Offsets] | None = None
    ) -> Iterable[str]:
        """Streams text spans from a list of documents.
        This can be useful for instance for labelling topics with highest ranking sentences.
        """
        if token_offsets is None:
            token_offsets = [None] * len(documents)
        for doc, offs in zip(documents, token_offsets):
            for span_start, span_end in self.extract_spans(
                doc, token_offsets=offs
            ):
                yield doc[span_start:span_end]
