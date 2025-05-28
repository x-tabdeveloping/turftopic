# Discovering a Data-driven Political Compass

The Political Compass is a dimensional theory of political ideologies and views.
This model posits that political ideology is distributed along a Left-Right and Libertarian-Authoritarian axis.

<figure>
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Political_Compass_purple_LibRight.svg/1920px-Political_Compass_purple_LibRight.svg.png" width="400px" style="margin-left: auto;margin-right: auto;">
  <figcaption>The Political Compass <br>(figure from Political Compass website)</figcaption>
</figure>

While this model enjoys wide public recognition, one potential issue with it is that it is a **top-down** model,
meaning that these dimensions were not discovered from some underlying data, but is based on experts' intuitions.
Dimensional analysis of views is also typically conducted using surveys.

In this tutorial we are going to look into how one could discover a **bottom-up**, data-driven Political Compass using the power of topic modelling, we will look at:

 - How to build and train a [Semantic Signal Separation ($S^3$)](../s3.md) model on our corpus
 - How to interpret the semantic axes discovered by our model
 - How to investigate the distribution of political parties along the discovered axes


## Installation

We will install Turftopic with Plotly to be able to plot our results, and the `datasets` library, for fetching data from HF Hub.

```python
pip install datasets plotly pandas turftopic
```

## Data Preparation

For this demonstration, I will be using a synthetic dataset, in which a large language model was tasked with expressing political opinions in free-form text.

```python
from datasets import load_dataset

ds = load_dataset("JyotiNayak/political_ideologies", split="train")
texts = ds["statement"]
```

We will be using the `paraphrase-MiniLM-L12-v2` for embedding our dataset and pre-computing embeddings.

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("paraphrase-MiniLM-L12-v2")
embeddings = encoder.encode(texts, show_progress_bar=True)
```

## Model Training

We will use the $S^3$ topic model for our investigations, as it conceptualized topics as independent axes in semantic space,
meaning it is built for establishing dimensional theories similar to the Political Compass.
For more details, read the documentation page on [Semantic Signal Separation](../s3.md).

Instead of a 2-dimensional model, similar to the Political Compass, we will opt to discover 3 dimensions.

!!! note
    You can easily expand this to more dimensions, the only reason we're not doing it here is because it would take more time to interpret them, and the tutorial is more accessible this way.

```python
from turftopic import SemanticSignalSeparation

model = SemanticSignalSeparation(3, encoder=encoder, random_state=42)
doc_topic_matrix = model.fit_transform(texts, embeddings=embeddings)
```

## Model Interpretation

First, let us examine the highest and lowest ranking terms on each axis in order to gain an intuition for what the dimensions could be about.

!!! tip
    For a more detailed discussion, see the [Model Interpretation](../model_interpretation.md) page in the documentation.

```python
model.print_topics(top_k=10)
```

| Topic ID | Highest Ranking | Lowest Ranking |
| - | - | - |
| 0 | religion, religious, faith, church, religions, faiths, doctrines, freedom, freedoms, beliefs | households, labor, household, poverty, socioeconomic, hardworking, income, wage, pay, welfare |
| 1 | investments, investment, spending, fiscal, invest, funding, policy, pollution, economic, budget | racism, racial, ethnicity, diverse, discrimination, distinct, genders , ethnic, families, adoption |
| 2 | warming, carbon, environment, environmental, planet, change, solar, greenhouse, fossil, biodiversity | wealth, taxation, prosperity, wealthiest, tax, profit, entrepreneurship, taxes, fiscal, government |

While this overview already gives us some idea as to what the axes represent, we might lose a lot of information by just looking at the top N words.
Luckily, Turftopic comes with utilities for displaying a more complete compass of concepts along two axes at a time.


!!! quote "Interpret Political Axes on the Concept Compass"
    === "Axis 0 and 1"

        ```python
        model.plot_concept_compass(0, 1)
        ```

        <center>
          <iframe src="../images/pol_comp_01.html", title="Axis 0 and 1", style="height:820px;width:820px;padding:0px;border:none;"></iframe>
        </center>

    === "Axis 1 and 2"

        ```python
        model.plot_concept_compass(1, 2)
        ```

        <center>
          <iframe src="../images/pol_comp_12.html", title="Axis 1 and 2", style="height:820px;width:820px;padding:0px;border:none;"></iframe>
        </center>

    === "Axis 2 and 0"

        ```python
        model.plot_concept_compass(2, 0)
        ```

        <center>
          <iframe src="../images/pol_comp_20.html", title="Axis 2 and 0", style="height:820px;width:820px;padding:0px;border:none;"></iframe>
        </center>


!!! note
    Note that these axes seem to differ quite a bit from those proposed by the Political Compass.
    Survey-based methods usually focus more on differences in views on selected issues,
    while it seems that we have discovered more of a distribution of issue-importance.
    Surveys have long been criticized for neglected the salience of issues for individuals,
    so while this method might not replace them, it could be a very useful for the augmentation of survey results.

These plots give us a deeper insight into how concepts are distributed along the discovered axes.
A potential interpretation of these could be the following:

```python
model.rename_topics({
    0: "Religiosity",
    1: "Economic vs Social",
    2: "Environmentalism",
})
```

As a sanity check we can also try predicting these axes for a new statement that we write:

```python
model.print_topic_distribution("I am a socialist and I am concerned with the growing inequality in our societies. I'd like to see governments do more to prevent the exploitation of workers.")
```

| Topic name | Score |
| - | - |
| Economic vs Social | 1.01 |
| Religiosity | -0.78 |
| Environmentalism | -1.10 |

This makes sense, as the statement above is mostly concerned with an economic issue, is not based in religion or beliefs, and is not about the environment.


## Relating Axes to Party Affiliation

In this synthetic dataset we also have access to party affiliation labels.
As such we can investigate the relation between (hypothetical) political parties and the discovered ideological dimensions.

We will do this by organizing all information into a dataframe, then plotting it on a scatterplot matrix.

```python
import pandas as pd
import plotly.express as px

df = pd.DataFrame(doc_topic_matrix, columns=model.topic_names)
df["party"] = ["Liberal" if label == 1 else "Conservative" for label in ds["label"]]

fig = px.scatter_matrix(df, dimensions=model.topic_names, color="party", template="plotly_white")
fig = fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(opacity=0.6))
fig.show()
```

<center>
  <iframe src="../images/pol_comp_parties.html", title="Parties' distribution on the axes", style="height:820px;width:820px;padding:0px;border:none;"></iframe>
</center>

While there doesn't seem to be a clear divide on these issues between these hypothetical liberals and convservatives, some differences can already be seen.
For instance environmental issues are discussed more by liberals, while belief-based and religious issues are more prevalent in conservative texts.


