# `TopicData`

While Turftopic provides a fully sklearn-compatible interface for training and using topic models, this is not always optimal, especially when you have to visualize models, or save more information about inference then would be practical to have in a `model` object.
We have thus added an abstraction borrowed from [topicwizard](https://github.com/x-tabdeveloping/topicwizard) called `TopicData`.

## Producing `TopicData`
Every model has methods, with which you can produce this object:

!!! quote "Prepare `TopicData` objects"
    === "`prepare_topic_data(corpus, embeddings=None)`"
        ```python
        topic_data = model.prepare_topic_data(corpus)
        # print to see what attributes are available
        print(topic_data)
        ```
        ```
        TopicData
        ├── corpus (1000)
        ├── vocab (1746,)
        ├── document_term_matrix (1000, 1746)
        ├── topic_term_matrix (10, 1746)
        ├── document_topic_matrix (1000, 10)
        ├── document_representation (1000, 384)
        ├── transform
        ├── topic_names (10)
        ├── has_negative_side
        └── hierarchy
        ```
    === "`prepare_dynamic_topic_data(corpus, timestamps, embeddings=None, bins=10)`"
        Models that support dynamic topic modeling have this method too, which includes dynamic topics in the resulting `TopicData` object.
        ```python
        import datetime

        timestamps: list[datetime.datetime] = [...] 
        topic_data = model.prepare_dynamic_topic_data(corpus, timestamps=timestamps)
        ```

## Using `TopicData`
`TopicData` is a dict-like object, and for all intents and purposes can be used as a Python dictionary, but for convenience you can also access its attributes with the dot syntax:

```python
# They are the same
assert topic_data["document_term_matrix"].shape == topic_data.document_term_matrix.shape
```

Much like models, you can pretty-print information about topic models based on the `TopicData` object, but, since it contains more information on inference then the model object itself, you sometimes have to pass less parameters than if you called the same method on the model:

```python
model.print_representative_documents(0, corpus, document_topic_matrix)
# This is simpler with TopicData, since you only have to pass the topic ID
topic_data.print_representative_documents(0)
```

When producing figures, `TopicData` also gives you shorthands for accessing the topicwizard web app and Figures API:

```python
topic_data.figures.topic_map()
```

<center>
<iframe src="https://x-tabdeveloping.github.io/topicwizard/_static/plots/topic_map.html" width="1000px" height="450px" frameborder=0></iframe>
</center>

See our guide on [Model Interpretation](model_interpretation.md) for more info.

## API Reference

::: turftopic.data.TopicData
