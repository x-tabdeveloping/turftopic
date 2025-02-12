# Interpreting and Visualizing Models

Interpreting topic models can be challenging. Luckily Turftopic comes loaded with a bunch of utilities you can use for interpreting your topic models.

```python
from turftopic import KeyNMF

model = KeyNMF(10)
topic_data = model.prepare_topic_data(corpus)
```

## Topic Tables

The easiest way you can investigate topics in your fitted model is to use the built-in pretty printing utilities, that you can call on every fitted model or `TopicData` object.

!!! quote "Interpret your models with topic tables"
    === "Relevant Words"

        ```python
        model.print_topics()
        # or
        topic_data.print_topics()
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

    === "Relevant Documents"

        ```python
        # Print highest ranking documents for topic 0
        model.print_representative_documents(0, corpus, document_topic_matrix)

        # since topic_data already stores the corpus and the doc-topic-matrix, you only need to give a topic ID
        topic_data.print_representative_documents(0)
        ```

        <center>

        | Document                                                                                             | Score |
        | -----------------------------------------------------------------------------------------------------| ----- |
        | Poor 'Poly'. I see you're preparing the groundwork for yet another retreat from your...              |  0.40 |
        | Then you must be living in an alternate universe. Where were they? An Appeal to Mankind During the... |  0.40 |
        | It is 'Serdar', 'kocaoglan'. Just love it. Well, it could be your head wasn't screwed on just right... |  0.39 |

        </center>

    === "Topic Distributions"

        ```python
        document = "I think guns should definitely banned from all public institutions, such as schools."

        model.print_topic_distribution(document)
        # or 
        topic_data.print_topic_distribution(document)
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


You can also export tables as pandas DataFrames by removing the `print_` prefix, and postfixing the method with `_df` or export tables in a given format, by using the `export_<something>` method instead of `print_<something>`.

=== "`DataFrame`"

    ```python
    model.topics_df()
    model.topic_distribution_df("something something")
    topic_data.representative_documents_df(5)
    ```

=== "Markdown"

    ```python
    model.export_topics(format="markdown")
    model.export_topic_distribution("something something", format="markdown")
    topic_data.export_representative_documents(5, format="markdown")
    ```

=== "Latex"

    ```python
    model.export_topics(format="latex")
    model.export_topic_distribution("something something", format="latex")
    topic_data.export_representative_documents(5, format="latex")
    ```

=== "CSV"

    ```python
    model.export_topics(format="csv")
    model.export_topic_distribution("something something", format="csv")
    topic_data.export_representative_documents(5, format="csv")
    ```


## Visualization with [topicwizard](https://github.com/x-tabdeveloping/topicwizard)

Turftopic comes with a number of **model-specific** visualization utilities, which you can check out on the [models](models.md) page.
We do provide a general overview here, as well as instructions on how to use [topicwizard](https://github.com/x-tabdeveloping/topicwizard) with Turftopic for interactive topic interpretation.

To use topicwizard you will first have to install it:
```python
pip install topic-wizard
```

<img src="https://x-tabdeveloping.github.io/topicwizard/_static/icon.svg" width="64px" style="float: left; margin-right: 5px">

### Web App

The easiest way to investigate any topic model interactively is to use the topicwizard web app.
You can launch the app either using a `TopicData` or a model object and a representative sample of documents.

=== "With `TopicData`"

    ```python
    topic_data.visualize_topicwizard()
    ```

=== "With `model`"
    ```python
    import topicwizard

    topicwizard.visualize(corpus=documents, model=model)
    ```

<img src="https://x-tabdeveloping.github.io/topicwizard/_images/screenshot_topics.png" width="100%" style="margin-left: auto;margin-right: auto;">


### Figures

You can also produce individual interactive figures using the [Figures API in topicwizard](https://x-tabdeveloping.github.io/topicwizard/figures.html).
Almost all figures in the Figures API can be called on the `figures` submodule of any `TopicData` object.

!!! quote "Interpret your models using interactive figures"
    === "Topic Map"
        ```python
        topic_data.figures.topic_map()
        ```

        <center>
        <iframe src="https://x-tabdeveloping.github.io/topicwizard/_static/plots/topic_map.html" width="800px" height="600px" frameborder=0></iframe>
        </center>


    === "Topic Barcharts"
        ```python
        topic_data.figures.topic_barcharts()
        ```

        <center>
        <iframe src="https://x-tabdeveloping.github.io/topicwizard/_static/plots/topic_barcharts.html" width="1000px" height="600px" frameborder=0></iframe>
        </center>

    === "Word Map"
        ```python
        topic_data.figures.word_map()
        ```

        <center>
        <iframe src="https://x-tabdeveloping.github.io/topicwizard/_static/plots/word_map.html" width="1000px" height="600px" frameborder=0></iframe>
        </center>

    === "Word Clouds"
        ```python
        topic_data.figures.topic_wordclouds()
        ```

        <center>
        <iframe src="https://x-tabdeveloping.github.io/topicwizard/_static/plots/topic_wordclouds.html" width="1200px" height="800px" frameborder=0></iframe>
        </center>

    === "Document Map"

        ```python
        topic_data.figures.document_map()
        ```

        <center>
        <iframe src="https://x-tabdeveloping.github.io/topicwizard/_static/plots/document_map.html" width="100%" height="600px" frameborder=0></iframe>
        </center>


## Datamapplot (Clustering models)

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



## Naming Topics

Topics in Turftopic by default are named based on the highest ranking keywords for a given topic.
You might however want to get more fitting names for your topics either automatically or assigning them manually.
See a our detailed guide about [Namers](../namers.md) to learn how you can use LLMs to assign names to topics.

!!! quote "Examples"

    === "Automated"

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
        | | ... | |

    === "Manual"

        ```python
        from turftopic import SemanticSignalSeparation

        model = SemanticSignalSeparation(10).fit(corpus)
        model.rename_topics({0: "New name for topic 0", 5: "New name for topic 5"})
        ```

