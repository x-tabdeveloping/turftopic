# Concept Vector Projection

Concept Vector Projection is an embedding-based method for extracting continuous sentiment (or other) scores from free-text documents.

<figure>
  <img src="../images/cvp.png", title="", style="width:1050px;padding:0px;border:none;"></img>
  <figcaption> Figure 1: Schematic Overview of Concept Vector Projection.<br> <i>Figure from Lyngbæk et al. (2025)</i> </figcaption>
</figure>

The method rests on the idea that one can construct a _concept vector_ by encoding positive and negative _seed phrases_ with a transformer, then taking the difference of these mean vectors.
We can then project other documents' embeddings onto these concept vectors by taking the dot product with the concept vector, thereby giving continuous scores on how related documents are to a given concept.

## Usage

### Single Concept

When projecting onto a single concept, you should specify the seeds as a tuple of positive and negative phrases.

```python
from turftopic import ConceptVectorProjection

positive = [
    "I love this product",
    "This is absolutely lovely",
    "My daughter is going to adore this"
]
negative = [
    "This product is not at all as advertised, I'm very displeased",
    "I hate this",
    "What a horrible way to deal with people"
]
cvp = ConceptVectorProjection(seeds=(positive, negative))

test_documents = ["My cute little doggy", "Few this is digusting"]
doc_concept_matrix = cvp.transform(test_documents)
print(doc_concept_matrix)
```

```python
[[0.24265897]
 [0.01709663]]
```

### Multiple Concepts

When projecting documents to multiple concepts at once, you will need to specify seeds for each concept, as well as its name.
Internally this is handled with an `OrderedDict`, which you can either specify yourself, or Turftopic can do it for you:

```python
import pandas as pd
from collections import OrderedDict

cuteness_seeds = (["Absolutely adorable", "I love how he dances with his little feet"], ["What a big slob of an abomination", "A suspicious old man sat next to me on the bus today"])
bullish_seeds = (["We are going to the moon", "This stock will prove an incredible investment"], ["I will short the hell out of them", "Uber stocks drop 7% in value after down-time."])

# Either specify it like this:
seeds = [("cuteness", cuteness_seeds), ("bullish", bullish_seeds)]
# or as an OrderedDict:
seeds = OrderedDict([("cuteness", cuteness_seeds), ("bullish", bullish_seeds)])
cvp = ConceptVectorProjection(seeds=seeds)

test_documents = ["What an awesome investment", "Tiny beautiful kitty-cat"]
doc_concept_matrix = cvp.transform(test_documents)
concept_df = pd.DataFrame(doc_concept_matrix, columns=cvp.get_feature_names_out())
print(concept_df)
```

```python
   cuteness   bullish
0  0.085957  0.288779
1  0.269454  0.009495
```

## Sentiment Arcs

Sometimes you might want to get a more granular understanding of how concepts evolve in a document.
`ConceptVectorProjection` can be used with [late-interaction/multi-vector functionality](late_interaction.md) in Turftopic, and thus you can easily generate sentiment arcs within documents that either span individual tokens or contextualized rolling windows.

!!! tip
    To get a more in-depth understanding of late-interaction/multi-vector models, read our [User Guide](late_interaction.md).

```python
from turftopic.late import LateWrapper, LateSentenceTransformer
# For plotting:
import plotly.express as px
import plotly.graph_objects as go

seeds = [("cuteness", cuteness_seeds), ("bullish", bullish_seeds)]

cvp = LateWrapper(
    ConceptVectorProjection(seeds=seeds, encoder=LateSentenceTransformer("all-MiniLM-L6-v2"))
)
test_documents = ["What an awesome investment", "Tiny beautiful kitty-cat"]
doc_concept_matrix, offsets = cvp.transform(test_documents)

# We will plot document 0's' sentiment arcs
fig = go.Figure()
# We extract the tokens
tokens = [test_documents[0][start:end] for start, end in offsets[0]]
# First token is [CLS]
tokens[0] = "[CLS]"
fig = fig.add_scatter(x=tokens, y=doc_concept_matrix[0][:, 0], name="Cuteness")
fig = fig.add_scatter(x=tokens, y=doc_concept_matrix[0][:, 1], name="Bullish")
fig.show()

```

<figure>
    <iframe src="../images/sentiment_arcs.html", title="Concepts evolving over tokens in the first document", style="height:500px;width:1000px;padding:0px;border:none;"></iframe>
    <figcaption> Figure 2: Concepts evolving over tokens in the first document. </figcaption>
</figure>


## Citation

Please cite Lyngbæk et al. (2025) and Turftopic when using Concept Vector Projection in publications:

```bibtex
@article{
  Kardos2025,
  title = {Turftopic: Topic Modelling with Contextual Representations from Sentence Transformers},
  doi = {10.21105/joss.08183},
  url = {https://doi.org/10.21105/joss.08183},
  year = {2025},
  publisher = {The Open Journal},
  volume = {10},
  number = {111},
  pages = {8183},
  author = {Kardos, Márton and Enevoldsen, Kenneth C. and Kostkan, Jan and Kristensen-McLachlan, Ross Deans and Rocca, Roberta},
  journal = {Journal of Open Source Software} 
}

@incollection{Lyngbaek2025,
  title = {Continuous Sentiment Scores for Literary and Multilingual
Contexts},
  author = {Laurits Lyngbaek and Pascale Feldkamp and Yuri Bizzoni and Kristoffer L. Nielbo and Kenneth Enevoldsen},
  year = {2025},
  booktitle = {Computational Humanities Research 2025},
  publisher = {Anthology of Computers and the Humanities},
  pages = {480--497},
  editor = {Taylor Arnold and Margherita Fantoli and Ruben Ros},
  doi = {10.63744/nVu1Zq5gRkuD}
}
```


## API Reference


::: turftopic.models.cvp.ConceptVectorProjection


