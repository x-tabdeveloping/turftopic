# Analyzing Discourse on Religion and Morality

Topic models are an effective tool for discourse analysis, and widely applied in computational humanities research.
One potential research question could be to investigate how groups online discuss the connection between morality and religion.

For this task, we are going to utilize [KeyNMF](../KeyNMF.md), which is a powerful topic model based on keyword/keyphrase extraction using contextual representations, we will look at:

 - How to construct and train a Seeded KeyNMF model on our corpus
 - How to interpret the topic model's output
 - How topics are distributed across groups

## Installation

```bash
pip install turftopic[topic-wizard]
```

## Data Preparation

We are going to use a subset of the 20 Newsgroups dataset, which includes three newsgroups oriented at discussing religion and atheism.
Luckily this dataset is directly available from scikit-learn's repositories.

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups

ds = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
    categories=[
        "alt.atheism",
        "talk.religion.misc",
        "soc.religion.christian",
    ],
)
corpus = ds.data
group_labels = np.array(ds.target_names)[ds.target]
```

Turftopic uses contextual embeddings of documents in order to understand what is in the corpus.
We will use the small and fast `all-MiniLM-L6-v2` sentence transformer to produce embeddings of our data.

!!! tip 
    Sometimes you might want to reuse embeddings in different topic models, or save them to disk.
    It is thus recommended, but not necessary to precompute them before running a topic model.

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(corpus, show_progress_bar=True)
```

## Model Training

Since we want to investigate the discourse from the perspective of morality, we would like to include this information in our model.
KeyNMF is capable of investigating a corpus from a certain angle based on a **seed phrase**, this will allow us to focus the topics in the model on morality.

!!! tip
    For a more detailed discussion, check out our documentation page on [Seeded Topic Modelling](../seeded.md)

```python
from turftopic import KeyNMF

model = KeyNMF(
    n_components=15,
    random_state=42,
    encoder=encoder,
    seed_phrase="Religion and Morality"
)
topic_data = model.prepare_topic_data(corpus, embeddings=embeddings)
```

## Model Interpretation


Let us first print the top 10 words for each topic discovered by the model.

!!! tip
    For a more detailed discussion, see the [Model Interpretation](../model_interpretation.md) page in the documentation.

```python
topic_data.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | atheist, atheists, theists, religious, abortion, agnostic, theist, just, mythology, argument |
| 1 | immoral, morals, morality, moral, morally, behavior, society, religious, meat, societies |
| 2 | bible, scripture, biblical, scriptures, faith, gospel, testament, commandments, revelation, interpretation |
| 3 | religion, religious, religions, beliefs, cult, science, spiritual, don, scientific, secular |
| 4 | god, believe, satan, existence, gods, evil, exist, just, genocide, creator |
| 5 | homosexuality, homosexual, homosexuals, heterosexual, sexual, gay, fornication, immoral, sex, sodom |
| 6 | morality, objective, subjective, absolute, natural, relativism, societal, objectively, science, correct |
| 7 | belief, faith, beliefs, believe, believing, evidence, philosophy, religions, believed, believes |
| 8 | atheism, theism, religious, alt, belief, stalin, agnosticism, theists, newsgroup, read |
| 9 | christians, religions, jesus, fundamentalist, beliefs, commandments, hell, worship, jewish, christian |
| 10 | christianity, christian, christ, gospel, evangelical, cult, book, life, faith, buddhist |
| 11 | moral, animals, species, kill, murder, killing, values, ethics, morally, society |
| 12 | church, catholic, christian, churches, catholics, religious, orthodox, theology, protestant, marriage |
| 13 | sin, sins, punishment, jesus, christ, salvation, sinner, repentance, repent, sinful |
| 14 | islam, muslim, islamic, muslims, qur, secular, religions, rushdie, law, khomeini |

We can already see multiple topics that are highly related to our research question.
Topic 2 for instance is concerned with dogmatic morality, topic 4 is focused on the subject of homosexuality,
which some religions/religious groups deem immoral, and Topic 11 is seemingly about murder and killing, but also includes *"animals"* and *"species"*, so it might be related to veganism.

To see whether our intuition is correct, we can investigate the top 10 documents on this topic:

```python
topic_data.print_representative_documents(11)
```

| Document | Score |
| - | - |
| See, there you go again, saying that a moral act is only significant if it is "voluntary." Why do you think this? And anyway, humans have the ability to disregard some of their instincts. You are attaching too many things to the term "moral," I think. Let's try this: is it "good" that animals of the... | 0.20 |
| For example, if it were instinctive not to murder... So, only intelligent beings can be moral, even if the bahavior of other beings mimics theirs? And, how much emphasis do you place on intelligence? Animals of the same species could kill each other arbitarily, but they don't. Are you trying to say ... | 0.20 |
| If you force me to do something, am I morally responsible for it? Well, make up your mind. Is it to be "instinctive not to murder" or not? It's not even correct. Animals of the same species do kill one another. Sigh. I wonder how many times we have been round this loop. I think that instinctive baha... | 0.18 |
| But now you are contradicting yourself in a pretty massive way, and I don't think you've even noticed. In another part of this thread, you've been telling us that the "goal" of a natural morality is what animals do to survive. But suppose that your omniscient being told you that the long term surviv... | 0.18 |
| Well I agree with you in the sense that they have no "moral" right to inflict these rules, but there is one thing I might add: at the very least, almost everybody wants to avoid pain, and if that means sacrific ing some stuff for a herd morality, then so be it. Right, and since they grew up and learn... | 0.17 |

By investigating the documents, we may discover, that this topic is actually about the differences between man and animal,
and whether animals can be held morally responsible.

### Investigating the Word Landscape

KeyNMF can also be thought of as a soft word clustering method, and we can display a word-map to investigate how and which words are related to each other.

```python
fig = topic_data.figures.word_map()
fig.show()
```
!!! tip
    You can zoom in on the graph to investigate groups closer.
    Hover over a word to display it.
    You can click on the legend to hide topics.

<center>
  <iframe src="../images/religion_word_map.html", title="Word Map of the Topic model", style="height:750px;width:1120px;padding:0px;border:none;"></iframe>
</center>


## Topical Differences in Groups

One interesting aspect to look at when investigating data from social media, is understanding, which topics get mostly discussed by which group.
Since we have group labels for each text in 20 Newsgroups, we can investigate how topic distributions are different in the three newsgroups we are analysing.

We can do this by summing up all document-topic distributions in the given groups, and then plotting them on a heatmap.

```python
import plotly.express as px
import pandas as pd

groups = [
    "alt.atheism",
    "talk.religion.misc",
    "soc.religion.christian",
]
group_topic_matrix = []
for label in groups:
    group_topic_matrix.append(
        topic_data.document_topic_matrix[group_labels==label].sum(axis=0)
    )
group_topic_matrix = np.stack(group_topic_matrix)
# We turn this into a dataframe, so that Plotly knows what to write on the x and y axes.
group_topic_matrix = pd.DataFrame(group_topic_matrix, columns=model.topic_names, index=groups)
fig = px.imshow(group_topic_matrix)
fig.show()
```

<center>
  <iframe src="../images/religion_topic_group_heatmap.html", title="Heatmap of topic-group distributions", style="height:350px;width:1120px;padding:0px;border:none;"></iframe>
</center>

Here we can see that the distinction between theism and atheism, as well as subjectivity and objectivity are central themes for atheists.
Christians, meanwhile, understandably, talk more about the bible, the gospel, differences between denominations and sins.
Interestingly, Islam is mostly discussed in the atheism group, but not in the religion ones.
