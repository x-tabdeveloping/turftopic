---
title: 'Turftopic: Topic Modelling with Contextual Representations from Sentence Transformers'
tags:
  - Python
  - topic modelling
  - sentence-transformers
  - embeddings
authors:
  - name: Márton Kardos
    orcid: 0000-0001-9652-4498
    affiliation: 1
  - name: Kenneth C. Enevoldsen
    orcid: 0000-0001-8733-0966
    affiliation: 1
  - name: Jan Kostkan
    orcid: 0000-0002-9707-7121
    affiliation: 1
  - name: Ross Deans Kristensen-McLachlan
    orcid: 0000-0001-8714-1911
    affiliation: "1, 3"
  - name: Roberta Rocca
    orcid: 0000-0002-0680-7097
    affiliation: 2
affiliations:
 - name: Center for Humanities Computing, Aarhus University, Denmark
   index: 1
 - name: Interacting Minds Center, Aarhus University, Denmark
   index: 2
 - name: Department of Linguistics, Cognitive Science, and Semiotics, Aarhus University, Denmark
   index: 3
date: 17 March 2025
bibliography: paper.bib
---

# Summary

Turftopic is a topic modelling library including a number of recent topic models that go beyond bag-of-words models and can understand text in context, utilizing representations from transformers.
Turftopic focuses on ease of use, providing a unified interface for a number of different modern topic models, and boasting both model-specific and model-agnostic interpretation and visualization utilities.
While the user is afforded great flexibility in model choice and customization, the library comes with reasonable defaults, so as not to needlessly overwhelm first-time users.
In addition, Turftopic allows the user to: a) model topics as they change over time, b) learn topics on-line from a stream of texts, c) find hierarchical structure in topics, d) learning topics in multilingual texts and corpora.
Users can utilize the power of large language models (LLMs) to give human-readable names to topics.
Turftopic also comes with built-in utilities for generating topic descriptions based on key-phrases or lemmas rather than individual words.

![An Overview of Turftopic's Functionality](assets/paper_banner.png)

# Statement of Need

While a number of software packages have been developed for contextual topic modelling in recent years, including BERTopic [@bertopic_paper], Top2Vec [@top2vec], CTM [@ctm], these packages include implementations of one or two topic models, and most of the utilities they provide are model-specific. This has resulted in the unfortunate situation that practitioners need to switch between different libraries and adapt to their particularities in both interface and functionality.
Some attempts have been made at creating unified packages for modern topic models, including STREAM [@stream] and TopMost [@topmost].
These packages, however, have a focus on neural models and topic model evaluation, have abstract and highly specialized interfaces, and do not include some popular topic models.
Additionally, while model interpretation is fundamental aspect of topic modelling, the interpretation utilities provided in these libraries are fairly limited, especially in comparison with model-specific packages, like BERTopic.

Turftopic unifies state-of-the-art contextual topic models under a superset of the `scikit-learn` [@scikit-learn] API, which users are likely already familiar with, and can be readily included in `scikit-learn` workflows and pipelines.
We focused on making Turftopic first and foremost an easy-to-use library that does not necessitate expert knowledge or excessive amounts of code to get started with, but gives great flexibility to power users.
Furthermore, we included an extensive suite of pretty-printing and visualization utilities that aid users in interpreting their results.
The library also includes three topic models, which to our knowledge only have implementations in Turftopic, these are: KeyNMF [@keynmf], Semantic Signal Separation (S^3^) [@s3], and GMM, a Gaussian Mixture model of document representations with a soft-c-tf-idf term weighting scheme.

# Functionality

Turftopic includes a wide array of contextual topic models from the literature, these include:
FASTopic [@fastopic], Clustering models, such as BERTopic [@bertopic_paper] and Top2Vec [@top2vec], auto-encoding topic models, like CombinedTM [@ctm] and ZeroShotTM [@zeroshot_tm], KeyNMF [@keynmf], S^3^ [@s3] and GMM.
At the time of writing, these models are representative of the state of the art in contextual topic modelling and intend to expand on them in the future.

![Components of a Topic Modelling Pipeline in Turftopic](https://x-tabdeveloping.github.io/turftopic/images/topic_modeling_pipeline.png){width="800px"}

Each model in Turftopic has an *encoder* component, which is used for producing continuous document-representations [@sentence_transformers], and a *vectorizer* component, which extracts term counts in each documents, thereby dictating which terms will be considered in topics.
The user has full control over what components should be used at different stages of the topic modelling process, thereby having fine-grained influence on the nature and quality of topics.

The library comes loaded with numerous utilities to help users interpret their results, including *pretty printing* utilities for exploring topics, *interactive visualizations* partially powered by the `topicwizard` [@topicwizard] Python package, and *automated topic naming* with LLMs.

To accommodate a variety of use cases, Turftopic can be used for *dynamic* topic modelling, where we expect topics to change over time.
Turftopic is also capable of extracting topics at multiple levels of granularity, thereby uncovering *hierarchical* topic structures.
Some models can also be fitted in an *online* fashion, where documents are accounted for as they come in batches.
Turftopic also includes *seeded* topic modelling, where a seed phrase can be used to retrieve topics relevant to the specific research question.

# Use Cases

Topic modelling is a key tool for quantitative text analysis [@quantitative_text_analysis], and can be utilized in a number of research settings, including exploratory data analysis, discourse analysis of diverse domains, such as newspapers, social media or policy documents.
Turftopic has already been utilized by @keynmf for analyzing information dynamics in Chinese diaspora media, and is currently being used in multiple ongoing research projects, including one analyzing discourse on the HPV vaccine in Denmark, and studying Danish golden-age literature.

# Target Audience

We expect that Turftopic will prove useful to a diverse user base including computational researchers in digital humanities and social sciences, and industry NLP professionals.
Turftopic is also an appropriate choice for educational purposes, providing instructors with a single, user-friendly framework for students to explore and compare alternative topic modelling approaches.

