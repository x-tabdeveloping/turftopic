from typing import Union

from turftopic.spans.base import Offsets, SpanExtractor

try:
    import spacy
    from spacy.language import Language
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "SpaCy is not installed on your machine, please run `pip install turftopic[spacy]` before using SpaCy tokenizers."
    ) from e


class SentenceExtractor(SpanExtractor):
    def __init__(self, nlp: Union[str, Language], load_parser: bool = False):
        """Extracts sentence spans from text.

        !!! warning
            At a minimum you have to have a SpaCy language installed to be able to use this class.

        Parameters
        ----------
        nlp: spacy.Language or str
            Either a pre-loaded SpaCy pipeline, OR
            a SpaCy model name (the model HAS TO BE installed on your machine), OR
            a language code (only when load_parser=False).
        load_parser: bool, default False
            Indicates whether the parser component should be used for sentence separation.
            When False, turftopic will try to load the SpaCy pipeline with the Sentencizer component only,
            which is rule-based. This is way faster, but less accurate than using a parser, especially for non-European languages.
        """
        self.nlp = nlp
        if isinstance(nlp, str):
            if load_parser:
                self._nlp = spacy.load(nlp)
            else:
                self._nlp = spacy.blank(nlp.split("_")[0])
                self._nlp.add_pipe("sentencizer")
        else:
            self._nlp = nlp

    def extract_spans(self, document: str, token_offsets=None) -> Offsets:
        """Extracts sentence offsets in a document.

        Parameters
        ----------
        document: str
            Text document.
        token_offsets: Offsets
            Ignored.

        Returns
        -------
        list[tuple[int, int]]
            Tuples of start and end characters for each extracted span.
        """
        doc = self._nlp(document)
        sentences = doc.sents
        offsets = []
        for sent in sentences:
            start_idx = doc[sent.start].idx
            end_idx = doc[sent.end].idx
            offsets.append((start_idx, end_idx))
        return offsets
