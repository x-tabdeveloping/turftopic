# Saving and loading models

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
