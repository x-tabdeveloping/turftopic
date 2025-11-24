# SensTopic (BETA)

SensTopic is a version of Semantic Signal Separation, that only discovers positive signals, while allowing components to be unbounded.
This is achieved with an algorithm called Semi-nonnegative Matrix Factorization or SNMF.

> :warning: This model is still in an experimental phase. More documentation and a paper are on their way. :warning:

SensTopic uses a very efficient implementation of the SNMF algorithm, that is implemented in raw NumPy, but also in JAX.
If you want to enable hardware acceleration and JIT compilation, make sure to install JAX before running the model.

```bash
pip install jax
```

Here's an example of running SensTopic on the 20 Newsgroups dataset:

```python
from sklearn.datasets import fetch_20newsgroups
from turftopic import SensTopic

corpus = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
).data

model = SensTopic(25)
model.fit(corpus)

model.print_topics()
```


| Topic ID | Highest Ranking |
| - | - |
| | ... |
| 8 | gospels, mormon, catholics, protestant, mormons, synagogues, seminary, catholic, liturgy, churches |
| 9 | encryption, encrypt, encrypting, crypt, cryptosystem, cryptography, cryptosystems, decryption, encrypted, spying |
| 10 | palestinians, israelis, palestinian, israeli, gaza, israel, gazans, palestine, zionist, aviv |
| 11 | nasa, spacecraft, spaceflight, satellites, interplanetary, astronomy, astronauts, astronomical, orbiting, astronomers |
| 12 | imagewriter, colormaps, bitmap, bitmaps, pkzip, imagemagick, colormap, formats, adobe, ghostscript |
| | ... |

## Sparsity

SensTopic has a sparsity hyper-parameter, that roughly dictates how many documents will be assigned to a single document, where many topics per document get penalized.
This means that the model is both a matrix factorization model, but can also function as a soft clustering model, depending on this parameter.
Unlike clustering models, however, it may assign multiple topics to documents that have them, and won't force every document to contain only one topic.

Higher values will make your model more like a clustering model, while lower values will make it more like a decomposition model:

??? info "Click to see code"
    ```python
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    from turftopic import SensTopic

    ds = load_dataset("gopalkalpande/bbc-news-summary", split="train")
    corpus = list(ds["Summaries"])

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = encoder.encode(corpus, show_progress_bar=True)

    models = []
    doc_topic_ms = []
    sparsities = np.array(
        [
            0.05,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            10.0,
        ]
    )
    for i, sparsity in enumerate(sparsities):
        model = SensTopic(
            n_components=3, random_state=42, sparsity=sparsity, encoder=encoder
        )
        doc_topic = model.fit_transform(corpus, embeddings=embeddings)
        doc_topic = (doc_topic.T / doc_topic.sum(axis=1)).T
        models.append(model)
        doc_topic_ms.append(doc_topic)
    a_name, b_name, c_name = models[0].topic_names
    records = []
    for i, doc_topic in enumerate(doc_topic_ms):
        for dt in doc_topic:
            a, b, c, *_ = dt
            records.append(
                {
                    "sparsity": sparsities[i],
                    a_name: a,
                    b_name: b,
                    c_name: c,
                    "topic": models[0].topic_names[np.argmax(dt)],
                }
            )
    df = pd.DataFrame.from_records(records)
    fig = px.scatter_ternary(
        df, a=a_name, b=b_name, c=c_name, animation_frame="sparsity", color="topic"
    )
    fig.show()
    ```

<figure>
  <iframe src="../images/ternary_sparsity.html", title="Ternary plot of topics in documents.", style="height:800px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Ternary plot of topic distribution in a 3 topic SensTopic model varying with sparsity. </figcaption>
</figure>

You can see that as the sparsity increases, topics get clustered much more clearly, and more weight gets allocated to the edges of the graph.

To see how many topics there are in your document you can use the `plot_topic_decay()` method, that shows you how topic weights get assigned to documents.

```python
model.plot_topic_decay()
```

<figure>
  <iframe src="../images/topic_decay.html", title="Topic Decay in SensTopic model", style="height:520px;width:1050px;padding:0px;border:none;"></iframe>
  <figcaption> Topic Decay in a SensTopic Model with sparsity=1. </figcaption>
</figure>

## Automatic number of topics

SensTopic can learn the number of topics in a given dataset.
In order to determine this quantity, we use a version of the Bayesian Information Criterion modified for NMF.
This does not work equally well for all corpora, but it can be a powerful tool when the number of topics is not known a-priori.

In this example the model finds 6 topics in the BBC News dataset:

```python
# pip install datasets
from datasets import load_dataset

ds = load_dataset("gopalkalpande/bbc-news-summary", split="train")
corpus = list(ds["Summaries"])

model = SensTopic("auto")
model.fit(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | liverpool, mourinho, chelsea, premiership, arsenal, striker, madrid, midfield, uefa, manchester |
| 1 | oscar, bafta, oscars, cast, cinema, hollywood, actor, screenplay, actors, films |
| 2 | mobile, mobiles, broadband, devices, digital, internet, computers, microsoft, phones, telecoms |
| 3 | tory, blair, minister, ministers, parliamentary, mps, parliament, politicians, constituency, ukip |
| 4 | tennis, competing, federer, wimbledon, iaaf, olympic, tournament, athlete, rugby, olympics |
| 5 | gdp, stock, economy, earnings, investments, investment, invest, exports, finance, economies |


## API Reference

::: turftopic.models.senstopic.SensTopic
