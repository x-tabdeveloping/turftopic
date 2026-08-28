# Topic Distribution Learning

While in most scenarios you can store an entire document-topic matrix in memory, this is not always the case, especially with extremely large datasets.
Distribution learners in Turftopic are exactly developed for this reason.

With a distribution learner, you can pass document-topic matrices per batch, and update its parameters, while slowly learning the true distribution of topics in the dataset with uncertainty.

To use distribution learners you should install `conjugate-models`:

```bash
pip install turftopic[conjugate]
```

## Example

```python
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from turftopic import SensTopic
from turftopic.distribution_learning import GaussianDistributionLearner

ds = fetch_20newsgroups(remove=("headers", "footers", "quotes"), subset="all")
corpus = ds.data

batch_size = 2000
model = SensTopic(random_state=42)
# Initializing the distribution learner
distribution_learner = GaussianDistributionLearner()
# batch fitting over the dataset
for batch_start in range(0, len(corpus), batch_size):
    batch_end = batch_start + batch_size
    # Calculating doc_topic_matrix for current batch
    batch_doc_topic_matrix = model.partial_fit_transform(
        corpus[batch_start:batch_end],
        merge_method="asymmetric_mean",
    )
    # Updating the posteriors
    distribution_learner.update(batch_doc_topic_matrix)

# `pip install plotly` if you want to plot
distribution_learner.plot_topic_distribution(model.topic_names)
```

<figure>
  <iframe src="../images/distribution_learner.html", title="Learned topic distribution", style="height:420px;width:900px;padding:0px;border:none;"></iframe>
  <figcaption> Topic distribution learned by the GaussianDistributionLearner. </figcaption>
</figure>

## API Reference

::: turftopic.distribution_learning.GaussianDistributionLearner
