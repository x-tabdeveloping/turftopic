# Dynamic Topic Modeling

If you want to examine the evolution of topics over time, you will need a dynamic topic model.

> Note that regular static models can also be used to study the evolution of topics and information dynamics, but they can't capture changes in the topics themselves.

## Theory

A number of different conceptualizations can be used to study evolving topics in corpora, for instance:

1. One can imagine topic representations to be governed by a Brownian Markov Process (random walk), in such a case the evolution is part of the model itself.
 In layman's terms you describe the evolution of topics directly in your generative model by expecting the topic representations to be sampled from Gaussian noise around the last time step.
 Sometimes researchers will also refer to such models as _state-space_ approaches.
 This is the approach that the original [DTM paper](https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf) utilizes.
 Along with [this paper](https://arxiv.org/pdf/1709.00025.pdf) on Dynamic NMF.
2. You can fit one underlying statistical model over the entire corpus, and then do post-hoc term importance estimation per time slice.
 This is [what BERTopic does](https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html).
3. You can fit one model per time slice, and then use some aggregation procedure to merge the models.
 This approach is used in the Dynamic NMF in [this paper](https://www.cambridge.org/core/journals/political-analysis/article/exploring-the-political-agenda-of-the-european-parliament-using-a-dynamic-topic-modeling-approach/BBC7751778E4542C7C6C69E6BF954E4B).

Developing such approaches takes a lot of time and effort, and we have plans to add dynamic modeling capabilities to all models in Turftopic.
For now only models of the second kind are on our list of things to do, and dynamic topic modeling has been implemented for GMM, and will soon be implemented for Clustering Topic Models.
For more theoretical background, see the page on [GMM](GMM.md).

## Usage

Dynamic topic models in Turftopic have a unified interface.
To fit a dynamic topic model you will need a corpus, that has been annotated with timestamps.
The timestamps need to be Python `datetime` objects, but pandas `Timestamp` object are also supported.

Models that have dynamic modeling capabilities have a `fit_transform_dynamic()` method, that fits the model on the corpus over time.

```python
from datetime import datetime

from turftopic import GMM

corpus: list[str] = [...]
timestamps: list[datetime] = [...]

model = GMM(5)
document_topic_matrix = model.fit_transform_dynamic(corpus, timestamps=timestamps)
```

To display results in a table, you can use the `print_topics_over_time()` method.

```python
model.print_topics_over_time(top_k=10)
```

<center>

| Time Slice | Topic 0                           | Topic 1                            | Topic 2                      |
|------------|-----------------------------------|------------------------------------|-----------------------------|
| 1950-1970  | vinyl, record, player, album       | hard bop, beatles, elvis, doors    |                             |
| 1970-1990  | cassettes, walkman, vinyl, recording | steely, genesis, jackson, queen   | ...                         |
| 1990-2010  | cd, mp3, digital, dvd               | muse, dilla, radiohead, ...       |                             |
| 2010-2020  | ...                               |                                    |                             |

</center>

You can also display the topics over time on an interactive HTML figure.
The most important words for topics get revealed by hovering over them.

> You will need to install Plotly for this to work.

```bash
pip install plotly
```

```python
model.plot_topics_over_time(top_k=5)
```

<figure>
  <img src="../images/topics_over_time.png" width="60%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Topics over time on a Figure</figcaption>
</figure>

## Interface

All dynamic topic models have a `temporal_components_` attribute, which contains the topic-term matrices for each time slice, along with a `temporal_importance_` attribute, which contains the importance of each topic in each time slice.

::: turftopic.dynamic.DynamicTopicModel
