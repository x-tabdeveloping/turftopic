# Model Persistence

Similarly to scikit-learn we primarily recommend that you use the `joblib` package for efficiently serializing Turftopic models.

```bash
pip install joblib
```

Make sure at all times that the Python environment producing the models has the same versions or similar of the essential packages as the recieving end.
Otherwise problems might occur at deserialization.

> It is also recommended that you do not embed custom code in models you intend to persist (custom vectorizers/encoder models).


```python
from turftopic import GMM

model = GMM(10).fit(corpus)

# Saving a model
joblib.dump(model, "gmm_10.joblib")

# Loading a model
model = joblib.load("gmm_10.joblib")
```
