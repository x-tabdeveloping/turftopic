# Hierarchical Topic Modeling

You might expect some topics in your corpus to belong to a hierarchy of topics.
Some models in Turftopic allow you to investigate hierarchical relations and build a taxonomy of topics in a corpus.

Models in Turftopic that can model hierarchical relations will have a `hierarchy` property, that you can manipulate and print/visualize:

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel(n_reduce_to=10).fit(corpus)
# We cut at level 3 for plotting, since the hierarchy is very deep
model.hierarchy.cut(3).plot_tree()
```

_Drag and click to zoom, hover to see word importance_

<iframe src="../images/tree_plot.html", title="Topic hierarchy in a clustering model", style="height:800px;width:100%;padding:0px;border:none;"></iframe>


## 1. Divisive/Top-down Hierarchical Modeling

In divisive modeling, you start from larger structures, higher up in the hierarchy, and  divide topics into smaller sub-topics on-demand.
This is how hierarchical modeling works in [KeyNMF](keynmf.md), which, by default does not discover a topic hierarchy, but you can divide topics to as many subtopics as you see fit.

As a demonstration, let's load a corpus, that we know to have hierarchical themes.

```python
from sklearn.datasets import fetch_20newsgroups

corpus = fetch_20newsgroups(
    subset="all",
    remove=("headers", "footers", "quotes"),
    categories=[
        "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware",
        "talk.religion.misc",
        "alt.atheism",
    ],
).data
```

In this case, we have two base themes, which are **computers**, and **religion**.
Let us fit a KeyNMF model with two topics to see if the model finds these.

```python
from turftopic import KeyNMF

model = KeyNMF(2, top_n=15, random_state=42).fit(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | windows, dos, os, disk, card, drivers, file, pc, files, microsoft |
| 1 | atheism, atheist, atheists, religion, christians, religious, belief, christian, god, beliefs |

The results conform our intuition. Topic 0 seems to revolve around IT, while Topic 1 around atheism and religion.
We can already suspect, however that more granular topics could be discovered in this corpus.
For instance Topic 0 contains terms related to operating systems, like *windows* and *dos*, but also components, like *disk* and *card*.

We can access the hierarchy of topics in the model at the current stage, with the model's `hierarchy` property.

```python
print(model.hierarchy)
```

<div style="background-color: #F5F5F5; padding: 10px; padding-left: 20px; padding-right: 20px;">
<tt style="font-size: 11pt">
<b>Root </b><br>
├── <b style="color: blue">0</b>: windows, dos, os, disk, card, drivers, file, pc, files, microsoft <br>
└── <b style="color: blue">1</b>: atheism, atheist, atheists, religion, christians, religious, belief, christian, god, beliefs <br>
</tt>
</div>

There isn't much to see yet, the model contains a flat hierarchy of the two topics we discovered and we are at root level.
We can dissect these topics, by adding a level to the hierarchy.

Let us add 3 subtopics to each topic on the root level.

```python
model.hierarchy.divide_children(n_subtopics=3)
```

<div style="background-color: #F5F5F5; padding: 10px; padding-left: 20px; padding-right: 20px;">
<tt style="font-size: 11pt">
<b>Root </b><br>
├── <b style="color: blue">0</b>: windows, dos, os, disk, card, drivers, file, pc, files, microsoft <br>
│   ├── <b style="color: magenta">0.0</b>: dos, file, disk, files, program, windows, disks, shareware, norton, memory <br>
│   ├── <b style="color: magenta">0.1</b>: os, unix, windows, microsoft, apps, nt, ibm, ms, os2, platform <br>
│   └── <b style="color: magenta">0.2</b>: card, drivers, monitor, driver, vga, ram, motherboard, cards, graphics, ati <br>
...
</tt>
</div>

As you can see, the model managed to identify meaningful subtopics of the two larger topics we found earlier.
Topic 0 got divided into a topic mostly concerned with dos and windows, a topic on operating systems in general, and one about hardware.

You can also divide individual topics to a number of subtopics, by using the `divide()` method.
Let us divide Topic 0.0 to 5 subtopics.

```python
model.hierarchy[0][0].divide(5)
model.hierarchy
```

<div style="background-color: #F5F5F5; padding: 10px; padding-left: 20px; padding-right: 20px;">
<tt style="font-size: 11pt">
<b>Root </b><br>
├── <b style="color: blue">0</b>: windows, dos, os, disk, card, drivers, file, pc, files, microsoft <br>
│   ├── <b style="color: magenta">0.0</b>: dos, file, disk, files, program, windows, disks, shareware, norton, memory <br>
│   │   ├── <b style="color: green">0.0.1</b>: file, files, ftp, bmp, program, windows, shareware, directory, bitmap, zip <br>
│   │   ├── <b style="color: green">0.0.2</b>: os, windows, unix, microsoft, crash, apps, crashes, nt, pc, operating <br>
...
</tt>
</div>

## 2. Agglomerative/Bottom-up Hierarchical Modeling

In other models, hierarchies arise from starting from smaller, more specific topics, and then merging them together based on their similarity until a desired number of top-level topics are obtained.

This is how it is done in [clustering topic models](clustering.md) like BERTopic and Top2Vec.
Clustering models typically find a lot of topics, and it can help with interpretation to merge topics until you gain 10-20 top-level topics.

You can either do this by default on a clustering model by setting `n_reduce_to` on initialization or you can do it manually with `reduce_topics()`.
For more details, check our guide on [Clustering models](clustering.md).

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel(
    n_reduce_to=10,
    feature_importance="centroid",
    reduction_method="smallest",
    reduction_topic_representation="centroid",
    reduction_distance_metric="cosine",
)
model.fit(corpus)

print(model.hierarchy)
```

<div style="background-color: #F5F5F5; padding: 10px; padding-left: 20px; padding-right: 20px;">
<tt style="font-size: 11pt">
<b>Root</b>: <br>
├── <b style="color:blue">-1</b>: documented, obsolete, et4000, concerns, dubious, embedded, hardware, xfree86, alternative, seeking<br>
├── <b style="color:blue">20</b>: hitter, pitching, batting, hitters, pitchers, fielder, shortstop, inning, baseman, pitcher<br>
├── <b style="color:blue">284</b>: nhl, goaltenders, canucks, sabres, hockey, bruins, puck, oilers, canadiens, flyers<br>
│   ├── <b style="color:magenta">242</b>: sportschannel, espn, nbc, nhl, broadcasts, broadcasting, broadcast, mlb, cbs, cbc<br>
│   │   ├── <b style="color:green">171</b>: stadium, tickets, mlb, ticket, sportschannel, mets, inning, nationals, schedule, cubs<br>
│   │   │   └── ...<br>
│   │   └── <b style="color:green">21</b>: sportschannel, nbc, espn, nhl, broadcasting, broadcasts, broadcast, hockey, cbc, cbs<br>
│   └── <b style="color:magenta">236</b>: nhl, goaltenders, canucks, sabres, puck, oilers, andreychuk, bruins, goaltender, leafs<br>
...
</tt>
</div>


## API reference

::: turftopic.hierarchical.TopicNode

::: turftopic.hierarchical.DivisibleTopicNode

::: turftopic.models._hierarchical_clusters.ClusterNode



