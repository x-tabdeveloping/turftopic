# Span Extraction

The main level of analysis in late-interaction models is the Span.
Spans can take many different forms and this page is dedicated to exploring how spans work.

The default Span in late interaction models, except when specified otherwise (e.g. `CTop2Vec` uses windows; `vSLDA` uses sentences) is the **token**.
This means that `token_embeddings, token_offsets` will be the same as `span_embeddings, span_offsets`.

You can alter this behaviour either by setting `window_size` and `step_size` or by specifying a `span_extractor`.

## Sliding Window over Tokens

***TODO***

## Heuristic Sentence Extraction


## Parser-based Sentence Extraction

## Implement a Custom Span Extractor

## API Reference

::: turftopic.spans.base.SpanExtractor
