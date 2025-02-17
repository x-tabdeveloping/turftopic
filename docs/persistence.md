# Saving and loading

## Model persistence
All models in Turftopic can be serialized and saved to disk, or published to the HuggingFace Hub.

### Saving locally

Turftopic models can now be saved to disk using the `to_disk()` method of models:

```python
from turftopic import SemanticSignalSeparation

model = SemanticSignalSeparation(10).fit(corpus)
model.to_disk("./local_directory/")
```

### Publishing models

Models can also be pushed to HuggingFace repositories.
This way, others can also easily access and modify topic models you've trained.

```python
# The repository name is, of course, arbitrary but descriptive
model.push_to_hub("your_user/s3_20-newsgroups_10-topics")
```

### Loading models

You can load models from either the Hub or disk using the `load_model()` function:

```python
from turftopic import load_model

model = load_model("./local_directory/")
# or from hub
model = load_model("your_user/s3_20-newsgroups_10-topics")
```

## `TopicData` persistence

You can also save and load `TopicData` objects with Turftopic.
These are saved using `joblib` and therefore we recommend that you give a `.joblib` file extension to all `TopicData` files:

!!! note "Note on compatibility"
    For backwards compatibility, `TopicData` objects are saved using `joblib` as simple `dict` objects.
    If you simply load a saved `TopicData` object with joblib without using `from_disk()`, it will load as a `dict`.

=== "Save"
    ```python
    topic_data = model.prepare_topic_data(corpus)
    topic_data.to_disk("topic_data.joblib")
    ```

=== "Load"
    ```python
    from turftopic.data import TopicData
    
    topic_data = TopicData.from_disk("topic_data.joblib")
    ```

