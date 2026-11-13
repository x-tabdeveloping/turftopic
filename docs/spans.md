# Span Extraction

The main level of analysis in late-interaction models is the Span.
Spans can take many different forms and this page is dedicated to exploring how spans work.

The default Span in late interaction models, except when specified otherwise (e.g. `CTop2Vec` uses windows; `vSLDA` uses sentences) is the **token**.
This means that `token_embeddings, token_offsets` will be the same as `span_embeddings, span_offsets`.

You can alter this behaviour either by setting `window_size` and `step_size` or by specifying a `span_extractor`.

## Sliding Window over Tokens

Angelov & Inkpen (2024) suggested to use sliding windows of tokens to be used as spans in their late model ([C-Top2Vec](c_top2vec.md)).
Token embeddings in these windows are then mean-pooled, and these embeddings will be used to represent spans.

There are two parameters to specify, when it comes to sliding windows:

| Parameter | Description |
|-|-|
| `window_size` | The size of the sliding window over tokens (how many tokens should be in a span). |
| `step_size` | By how much the window should be stepped forward each time. |

Note that if `step_size` is `None`, the `window_size` will be used as the `step_size`, meaning no overlap.
If the `step_size` is smaller than the `window_size`, then windows will overlap, and if it is larger, there will be a gap between windows.

!!! note
    Windows of tokens might not end up with natural linguistic boundaries. Sometimes words are comprised of multiple tokens for instance.

```python
from turftopic import vSLDA

# vSLDA, by default, is a sentence-level topic model
model = vSLDA(
    10,
    # this is 'sentence' by default, we want to supress that behaviour
    span_extractor=None,
    window_size=50,
    step_size=40,
)
doc_topic, window_offsets = model.fit_transform(corpus)
```

## Sentence Extraction

One natural boundary for spans would be sentences.
Sentence separation is a non-trivial task, and depending on how accurate you want the results to be, there are multiple options available in Turftopic.

### Heuristic-based

Turftopic includes a very fast heuristic-based sentence extractor.
This model divides sentences based on a pre-defined list of sentence-separator tokens, and tries to filter out non-sentences based on deterministic rules.
We found that this performs reasonably well for most use cases, but in certain instances, for example with non-European languages it might be necessary to use a more sofisticated sentence separator.

```python
from turftopic.spans.dummy import DummySentenceExtractor

model = vSLDA(10, span_extractor=DummySentenceExtractor())
# OR
model = vSLDA(10, span_extractor="sentence")
```

### Parser-based

If you want more accurate sentence boundary detection, you will need a natural language parser.
In Turftopic, you can make use of SpaCy's NLP pipelines to detect sentence boundaries.

!!! info
    You will need to install SpaCy (`pip install spacy`) and a SpaCy model for the language you want to use (e.g. `python -m spacy download en_core_web_sm`).
    For more info, check out [SpaCy's documentation](https://spacy.io/models/). 

```python
from turftopic.spans.spact import SentenceExtractor

model = vSLDA(10, span_extractor=SentenceExtractor("en_core_web_sm"))
```

## Implement a Custom Span Extractor

You might want to implement your own span extractor, if you need custom spans for your topic model (e.g. named entities, paragraphs, etc.).
In order to do this, you :
 - Have to inherit from the `SpanExtractor` class
 - Have to implement the `extract_spans()` method, which takes a document, then returns span boundaries in the form of (start, end) character index pairs.

In this example I will implement a paragraph separator, where paragraphs are separated by at least two `"\n"` characters:

```python
from turftopic.spans.base import Offsets, SpanExtractor

class ParagraphExtractor(SpanExtractor):
    def extract_spans(
        self, document: str, token_offsets: Offsets | None = None
    ) -> Offsets:
        # We will ignore token offsets here, since we just need to check characters.
        span_offsets = []
        span_start = 0
        n_linebreaks = 0
        # We go through all characters
        for i_ch, ch in enumerate(document):
            # If we already have two line breaks and the next character
            # is not a line break, we save the current span's start and end character index
            if (n_linebreaks >= 2) and ch != "\n":
                span_offset = (span_start, i_ch)
                span_offsets.append(span_offset)
                span_start = i_ch
                n_linebreaks = 0
                continue
            if ch == "\n":
                n_linebreaks += 1
            else:
                n_linebreaks = 0
        # We add the last one
        if i_ch != span_start:
            span_offsets.append((span_start, i_ch + 1))
        return span_offsets


# A silly text just to try whether it works
text = "first\n\n second \n\n\n third\nparagraph\n\n\n\n last"

extractor = ParagraphExtractor()
offsets = extractor.extract_spans(text)
print(offsets)
# prints [(0, 7), (7, 18), (18, 38), (38, 42)]

print([text[start:end].strip() for start, end in offsets])
# prints ['first', 'second', 'third\nparagraph', 'last']
```

You can then use this paragraph extractor in your topic model:

```python
model = vSLDA(10, span_extractor=ParagraphExtractor())
```

## API Reference

::: turftopic.spans.base.SpanExtractor
