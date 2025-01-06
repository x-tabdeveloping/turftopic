# Interpreting and Visualizing Models

Interpreting topic models can be challenging. Luckily Turftopic comes loaded with a bunch of utilities you can use for interpreting your topic models.


## Relevant Terms and Documents

Quite often the most relevant words and documents for a topic can reveal a lot about its content.
We provide a bunch of pretty-printing tools for accessing this information in a readable way.

#### Relevant Words

To see the highest the most important words for each topic, use the `print_topics()` method.

```python
model.print_topics()
```

<center>

| Topic ID | Top 10 Words                                                                                    |
| -------- | ----------------------------------------------------------------------------------------------- |
|        0 | armenians, armenian, armenia, turks, turkish, genocide, azerbaijan, soviet, turkey, azerbaijani |
|        1 | sale, price, shipping, offer, sell, prices, interested, 00, games, selling                      |
|        2 | christians, christian, bible, christianity, church, god, scripture, faith, jesus, sin           |
|        3 | encryption, chip, clipper, nsa, security, secure, privacy, encrypted, crypto, cryptography      |
|         | ....                                |


</center>

#### Relevant Documents

You can also print the highest ranking documents for each topic if you have saved the document-topic matrix.

```python
# Print highest ranking documents for topic 0
model.print_representative_documents(0, corpus, document_topic_matrix)
```

<center>

| Document                                                                                             | Score |
| -----------------------------------------------------------------------------------------------------| ----- |
| Poor 'Poly'. I see you're preparing the groundwork for yet another retreat from your...              |  0.40 |
| Then you must be living in an alternate universe. Where were they? An Appeal to Mankind During the... |  0.40 |
| It is 'Serdar', 'kocaoglan'. Just love it. Well, it could be your head wasn't screwed on just right... |  0.39 |

</center>

#### Topic Distributions

You can also print a topic distribution for a piece of text, using the `print_topic_distribution()` method:
```python
model.print_topic_distribution(
    "I think guns should definitely banned from all public institutions, such as schools."
)
```

<center>

| Topic name                                | Score |
| ----------------------------------------- | ----- |
| 7_gun_guns_firearms_weapons               |  0.05 |
| 17_mail_address_email_send                |  0.00 |
| 3_encryption_chip_clipper_nsa             |  0.00 |
| 19_baseball_pitching_pitcher_hitter       |  0.00 |
| 11_graphics_software_program_3d           |  0.00 |

</center>

### Exporting your Results

If you want to share these results, you can also export all tables, by using the `export_<something>` method instead of `print_<something>`.

```python
csv_table: str = model.export_topic_distribution("something something", format="csv")

latex_table: str = model.export_topics(format="latex")

md_table: str = model.export_representative_documents(0, corpus, document_topic_matrix, format="markdown")
```

## Topic Naming

Topics in Turftopic by default are named based on the highest ranking keywords for a given topic.
You might however want to get more fitting names for your topics either automatically or assigning them manually.

### Manual topic naming

You can manually name topics in Turftopic models after having interpreted them.
If you find a more fitting name for a topic, feel free to rename it in your model.

```python
from turftopic import SemanticSignalSeparation

model = SemanticSignalSeparation(10).fit(corpus)
model.rename_topics({0: "New name for topic 0", 5: "New name for topic 5"})

```

### Automated topic naming

You can also use large language models, or other NLP techniques to assign human-readable names to topics.
Here is an example of using ChatGPT to generate topic names from the highest ranking keywords.

Read more about namer models [here](namers.md).

```python
from turftopic import KeyNMF
from turftopic.namers import OpenAITopicNamer

namer = OpenAITopicNamer("gpt-4o-mini")
model.rename_topics(namer)

model.print_topics()
```

| Topic ID | Topic Name | Highest Ranking |
| - | - | - |
| 0 | Operating Systems and Software  | windows, dos, os, ms, microsoft, unix, nt, memory, program, apps |
| 1 | Atheism and Belief Systems | atheism, atheist, atheists, belief, religion, religious, theists, beliefs, believe, faith |
| 2 | Computer Architecture and Performance | motherboard, ram, memory, cpu, bios, isa, speed, 486, bus, performance |
| 3 | Storage Technologies | disk, drive, scsi, drives, disks, floppy, ide, dos, controller, boot |
| 4 | Moral Philosophy and Ethics | morality, moral, objective, immoral, morals, subjective, morally, society, animals, species |
| 5 | Christian Faith and Beliefs | christian, bible, christians, god, christianity, religion, jesus, faith, religious, biblical |
| 6 | Serial Modem Connectivity | modem, port, serial, modems, ports, uart, pc, connect, fax, 9600 |
| 7 | Graphics Card Drivers | card, drivers, monitor, vga, driver, cards, ati, graphics, diamond, monitors |
| 8 | Windows File Management | file, files, ftp, bmp, windows, program, directory, bitmap, win3, zip |
| 9 | Printer Font Management | printer, print, fonts, printing, font, printers, hp, driver, deskjet, prints |

## Visualization

Turftopic comes with a number of **model-specific** visualization utilities, which you can check out in the [models](models.md) page.
We do provide a general overview here, as well as instructions on how to use [topicwizard](https://github.com/x-tabdeveloping/topicwizard) with Turftopic for interactive topic interpretation.

### Datamapplot *(clustering models only)*

You can interactively explore clusters using [datamapplot](https://github.com/TutteInstitute/datamapplot) directly in Turftopic!
You will first have to install `datamapplot` for this to work:

```bash
pip install turftopic[datamapplot]
```

```python
from turftopic import ClusteringTopicModel
from turftopic.namers import OpenAITopicNamer

model = ClusteringTopicModel(feature_importance="centroid").fit(corpus)

namer = OpenAITopicNamer("gpt-4o-mini")
model.rename_topics(namer)

fig = model.plot_clusters_datamapplot()
fig.save("clusters_visualization.html")
fig
```
!!! info
    If you are not running Turftopic from a Jupyter notebook, make sure to call `fig.show()`. This will open up a new browser tab with the interactive figure.

<figure>
  <iframe src="../images/cluster_datamapplot.html", title="Cluster visualization", style="height:800px;width:800px;padding:0px;border:none;"></iframe>
  <figcaption> Interactive figure to explore cluster structure in a clustering topic model. </figcaption>
</figure>


### [topicwizard](https://github.com/x-tabdeveloping/topicwizard)

topicwizard is an interactive, model-agnostic topic model visualization framework that you can use to explore your topics models and to produce beautiful plots.

topicwizard does not come preloaded with Turftopic, but the two libraries are highly compatible. You only have to install topicwizard, and it will work right out of the box.

```bash
pip install topic-wizard
```

#### topicwizard Web App

By far the easiest way to visualize your models for interpretation is to launch the topicwizard web app.
You can try out the web app on [HuggingFace Spaces](https://huggingface.co/spaces/kardosdrur/topicwizard_20newsgroups_KeyNMF).

```python
import topicwizard

topicwizard.visualize(model=model, corpus=corpus)
```

<figure>
  <img src="https://x-tabdeveloping.github.io/topicwizard/_images/screenshot_topics.png" width="100%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Screenshot of the topicwizard Web Application</figcaption>
</figure>

#### Topic Data

The easiest way to use topicwizard with Turftopic is to produce `TopicData` objects that contain all relevant information about your topic model, instead of just calling `fit()`.
All models have a `prepare_topic_data()` method, that you can use to produce this data, that also fits your topic model the same way `fit()` would do:

```python
from turftopic import KeyNMF

model = KeyNMF(10)
topic_data = model.prepare_topic_data(corpus)
```

You can then use this to launch topicwizard:

```python
import topicwizard

topicwizard.visualize(topic_data=topic_data)
```

`TopicData` can also be used for producing individual figures:

#### Figures API

You can also produce individual interactive figures using the [Figures API in topicwizard](https://x-tabdeveloping.github.io/topicwizard/figures.html).

```python
from topicwizard.figures import word_map

topic_data = model.prepare_topic_data(corpus)

fig = word_map(topic_data)
fig.show()
```

<figure>
  <img src="https://github.com/x-tabdeveloping/topicwizard/raw/main/assets/word_map.png" width="100%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Word Map produced by topicwizard</figcaption>
</figure>


```python
from topicwizard.figures import topic_wordclouds

fig = topic_wordclouds(topic_data)
fig.show()
```

<figure>
  <img src="https://github.com/x-tabdeveloping/topicwizard/raw/main/assets/topic_wordclouds.png" width="100%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Wordclouds produced by topicwizard</figcaption>
</figure>
