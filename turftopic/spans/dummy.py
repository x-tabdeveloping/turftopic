"""Inspired by SpaCy https://github.com/explosion/spaCy/blob/7ff14d4da84817f84c32b980882178ddacc066a1/spacy/pipeline/sentencizer.pyx#L19"""

from turftopic.spans.base import Offsets, SpanExtractor

DEFAULT_PUNCT_CHARS = [
    "!",
    ".",
    "?",
    "։",
    "؟",
    "۔",
    "܀",
    "܁",
    "܂",
    "߹",
    "।",
    "॥",
    "၊",
    "။",
    "።",
    "፧",
    "፨",
    "᙮",
    "᜵",
    "᜶",
    "᠃",
    "᠉",
    "᥄",
    "᥅",
    "᪨",
    "᪩",
    "᪪",
    "᪫",
    "᭚",
    "᭛",
    "᭞",
    "᭟",
    "᰻",
    "᰼",
    "᱾",
    "᱿",
    "‼",
    "‽",
    "⁇",
    "⁈",
    "⁉",
    "⸮",
    "⸼",
    "꓿",
    "꘎",
    "꘏",
    "꛳",
    "꛷",
    "꡶",
    "꡷",
    "꣎",
    "꣏",
    "꤯",
    "꧈",
    "꧉",
    "꩝",
    "꩞",
    "꩟",
    "꫰",
    "꫱",
    "꯫",
    "﹒",
    "﹖",
    "﹗",
    "！",
    "．",
    "？",
    "𐩖",
    "𐩗",
    "𑁇",
    "𑁈",
    "𑂾",
    "𑂿",
    "𑃀",
    "𑃁",
    "𑅁",
    "𑅂",
    "𑅃",
    "𑇅",
    "𑇆",
    "𑇍",
    "𑇞",
    "𑇟",
    "𑈸",
    "𑈹",
    "𑈻",
    "𑈼",
    "𑊩",
    "𑑋",
    "𑑌",
    "𑗂",
    "𑗃",
    "𑗉",
    "𑗊",
    "𑗋",
    "𑗌",
    "𑗍",
    "𑗎",
    "𑗏",
    "𑗐",
    "𑗑",
    "𑗒",
    "𑗓",
    "𑗔",
    "𑗕",
    "𑗖",
    "𑗗",
    "𑙁",
    "𑙂",
    "𑜼",
    "𑜽",
    "𑜾",
    "𑩂",
    "𑩃",
    "𑪛",
    "𑪜",
    "𑱁",
    "𑱂",
    "𖩮",
    "𖩯",
    "𖫵",
    "𖬷",
    "𖬸",
    "𖭄",
    "𛲟",
    "𝪈",
    "｡",
    "。",
]


class DummySentenceExtractor(SpanExtractor):
    """A very simple, rule-based sentencizer."""

    def extract_spans(
        self, document: str, token_offsets: Offsets | None = None
    ) -> Offsets:
        """Extracts sentence offsets in a document.

        Parameters
        ----------
        document: str
            Text document.
        token_offsets: Offsets or None
            If token offsets are given, they will be used for sentence separation,
            otherwise, it will be done on a character level.

        Returns
        -------
        list[tuple[int, int]]
            Tuples of start and end characters for each extracted span.
        """
        if token_offsets is None:
            # If no offsets are provided, we treat each character as a token.
            token_offsets = [(i, i + 1) for i in range(len(document))]
        sentence_offsets = []
        sentence_start = 0
        for token_start, token_end in token_offsets:
            token = document[token_start:token_end]
            if token in DEFAULT_PUNCT_CHARS:
                sentence = document[sentence_start:token_end]
                # We make sure that no sentences are less than 3 characters, because that is ridiculous
                if len(sentence) < 3:
                    continue
                if all((c in DEFAULT_PUNCT_CHARS) for c in sentence):
                    # We skip "sentences" that only contain punctuation.
                    sentence_start = token_end
                    continue
                sentence_offsets.append((sentence_start, token_end))
                sentence_start = token_end
        if sentence_start != token_end:
            sentence_offsets.append((sentence_start, token_end))
        return sentence_offsets
