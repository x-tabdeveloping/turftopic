# Modifying and finetuning models

Some models in Turftopic can be flexibly modified after being fitted.
This allows users to fit pretrained topic models to their specific use cases.

## Naming/renaming topics

Topics can be freely renamed in all topic models.
This can be beneficial when interpreting models, as it allows you to assign labels to the topics you've already looked at. 

```python
from turftopic import SemanticSignalSeparation

model = SemanticSignalSeparation(10).fit(corpus)

# you can specify a dict mapping IDs to names
model.rename_topics({0: "New name for topic 0", 5: "New name for topic 5"})
# or a list of topic names
model.rename_topics([f"Topic {i}" for i in range(10)])
```

## Changing the number of topics

Multiple models allow you to change the number of topics in a model after fitting them.

### Refitting $S^3$ with different number of topics

$S^3$ models store all information that is needed to refit them using a different number of topics, iterations or random seed.
This process is incredibly fast and allows you to explore semantics in a corpora on multiple levels of detail.
Moreover, any model you load from a third party can be refitted at will.

```python
from turftopic import load_model

model = load_model("hf_user/some_s3_model")

print(type(model))
# turftopic.models.decomp.SemanticSignalSeparation

print(len(model.topic_names))
# 10

model.refit(n_components=20, random_seed=42)
print(len(model.topic_names))
# 20
```

### Merging topics in clustering models

Clustering models are very flexible in this regard, as they allow you to merge clusters after the model has been fitted.

#### Manual topic merging

You can merge topics manually in a clustering model by using the `join_topics()` method:

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel().fit(corpus)

# This will join topic 0, 5 and 4 into topic 0
model.join_topics([0,5,4])
```

#### Hierarchical merging

You can also merge clusters automatically into a desired number of topics.
This can be done with the `reduce_topics()` method:

!!! info
    For more info on topic merging methods, check out [this page](clustering.md)

```python
model = ClusteringTopicModel().fit(corpus)
model.reduce_topics(n_reduce_to=20, reduction_method="smallest")
```

## Finetuning models on a new corpus.

Currently, you can only finetune KeyNMF to a new corpus.
You can do this by using the `partial_fit()` method on texts the model hasn't seen before:

```python
from turftopic import load_model

model = load_model("pretrained_keynmf_model")

print(type(model))
# turftopic.models.keynmf.KeyNMF

new_corpus: list[str] = [...]
# Finetune the model to the new corpus
model.partial_fit(new_corpus)

model.to_disk("finetuned_model/")
```


## Re-estimating word importance

Both $S^3$ and Clustering models come with multiple ways of estimating the importance of words for topics.
Since both of these models use post-hoc measures, these scores can be calculated without fitting a new model or refitting an old one.
This allows you to play around with different types of feature importance estimation measures for the same model (same underlying clusters or axes).

Here's an example with $S^3$:
```python
from turftopic import SemanticSignalSeparation

model = SemanticSignalSeparation(5, feature_importance="combined").fit(corpus)
model.print_topics()
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Topic ID ┃ Highest Ranking                                              ┃ Lowest Ranking                                        ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        0 │ hypocrisy, hypocritical, fallacy, debated, skeptics          │ xfree86, emulator, codes, 9600, cd300                 │
├──────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│        1 │ spectrometer, dblspace, statistically, nutritional, makefile │ uh, um, yeah, hm, oh                                  │
├──────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│        2 │ bullpen, goaltenders, pitchers, goaltender, pitching         │ intel, nsa, spying, encrypt, terrorism                │
├──────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│        3 │ espionage, wiretapping, cia, fbi, wiretaps                   │ agnosticism, agnostic, upgrading, affordable, cheaper │
├──────────┼──────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────┤
│        4 │ affordable, dealers, warrants, handguns, dealership          │ semitic, theologians, judaism, persecuted, pagan      │
└──────────┴──────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────┘


model.estimate_components("angular")
model.print_topics()
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Topic ID ┃ Highest Ranking                                              ┃ Lowest Ranking                                           ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        0 │ hypocritical, debated, hypotheses, misconceptions, fallacy   │ diagnostics, win31, modems, cd300, gd3004                │
├──────────┼──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│        1 │ spectrometer, dblspace, statistically, makefile, nutritional │ ye, sub, naked, experiences, uh                          │
├──────────┼──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│        2 │ bullpen, puckett, hitters, clemens, jenks                    │ encryption, encrypt, intel, cryptosystem, cryptosystems  │
├──────────┼──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│        3 │ journalists, cdc, chlorine, npr, briefing                    │ values, ratios, upgrading, calculations, inherit         │
├──────────┼──────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
│        4 │ handguns, warrants, warranty, reliability, handgun           │ nutritional, metabolism, deuteronomy, pathology, hormone │
└──────────┴──────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘

```

And one with clustering models:

!!! info
    Remember, these are the same underlying clusters, just described in two different ways. For further details, check out [this page](clustering.md)

```python
from turftopic import ClusteringTopicModel

model = ClusteringTopicModel(n_reduce_to=5, feature_importance="soft-c-tf-idf").fit(corpus)
model.print_topics()
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Topic ID ┃ Highest Ranking                                                            ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       -1 │ like, just, don, use, does, know, time, good, people, edu                  │
├──────────┼────────────────────────────────────────────────────────────────────────────┤
│        0 │ people, said, god, president, mr, think, going, say, did, myers            │
├──────────┼────────────────────────────────────────────────────────────────────────────┤
│        1 │ max, g9v, b8f, a86, pl, 00, 145, 1d9, dos, 34u                             │
├──────────┼────────────────────────────────────────────────────────────────────────────┤
│        2 │ msg, cancer, food, battery, water, candida, medical, vitamin, yeast, diet  │
├──────────┼────────────────────────────────────────────────────────────────────────────┤
│        3 │ 25, 55, pit, det, pts, la, bos, 03, 10, 11                                 │
├──────────┼────────────────────────────────────────────────────────────────────────────┤
│        4 │ insurance, car, dog, radar, health, bike, helmet, private, detector, speed │
└──────────┴────────────────────────────────────────────────────────────────────────────┘


model.estimate_components(feature_importance="centroid")
model.print_topics()

┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Topic ID ┃ Highest Ranking                                                                                      ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       -1 │ documented, concerns, dubious, obsolete, concern, alternative, et4000, complaints, cx, discussed     │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│        0 │ persecutions, persecution, condemning, condemnation, fundamentalists, persecuted, fundamentalism,    │
│          │ theology, advocating, fundamentalist                                                                 │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│        1 │ xfree86, pcx, emulation, microsoft, hardware, emulator, x11r5, netware, workstations, chipset        │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│        2 │ contamination, fungal, precautions, harmful, poisoning, chemicals, treatments, toxicity, dangers,    │
│          │ prevention                                                                                           │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│        3 │ nhl, bullpen, goaltenders, standings, sabres, canucks, braves, mlb, flyers, playoffs                 │
├──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤
│        4 │ automotive, vehicle, vehicles, speeding, automobile, automobiles, driving, motorcycling,             │
│          │ motorcycles, highways                                                                                │
└──────────┴───────────────────────────────────────────────────────────────────────────#───────────────────────────┘

```


