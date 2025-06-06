# Multimodal Topic Modelling ***(BETA)***

!!! note 
    Multimodal modeling is still a BETA feature in Turftopic, and it is likely that we will add more features and change the interface in the near future.

Some corpora spread across multiple modalities.
A good example of this would be news articles with images attached.
Turftopic now supports multimodal modelling with a number of models.


## Multimodal Encoders

In order for images to be usable in Turftopic, you will need an embedding model that can both encode texts and images.
You can both use models that are supported in SentenceTransformers, or those that support the MTEB multimodal encoder interface.


!!! quote "Use a multimodal encoder model "
    === "SentenceTransformers"

        ```python
        from turftopic import KeyNMF

        multimodal_keynmf = KeyNMF(10, encoder="clip-ViT-B-32")
        ```

    === "MTEB/MIEB"
        !!! tip 
            You can find current state-of-the-art embedding models and their capabilities on the [Massive Image Embedding Benchmark leaderboard](http://mteb-leaderboard.hf.space/?benchmark_name=MIEB%28Multilingual%29).

        ```bash
        pip install "mteb<2.0.0"
        ```

        ```python
        from turftopic import KeyNMF
        import mteb

        encoder = mteb.get_model("kakaobrain/align-base")

        multimodal_keynmf = KeyNMF(10, encoder="clip-ViT-B-32")
        ```

## Corpus Structure

Currently all documents **have to have** an image attached to them, and only one image.
This is a limitation, and we will address it in the future.
Images can both be represented as file paths or `PIL.Image` objects.

```python
from PIL import Image

images: list[Image] = [Image.open("file_path/something.jpeg"), ...]
texts: list[str] = [...]

len(images) == len(texts)
```

## Basic Usage

All multimodal models have a `fit_multimodal()`/`fit_transform_multimodal()` method,
that you can use to discover topics in multimodal corpora.

!!! quote "Fit a multimodal model on a corpus"
    === "SemanticSignalSeparation"

        ```python
        from turftopic import SemanticSignalSeparation

        model = SemanticSignalSeparation(12, encoder="clip-ViT-B-32")
        model.fit_multimodal(texts, images=images)
        model.plot_multimodal_topics()
        ```

    === "KeyNMF"

        ```python
        from turftopic import KeyNMF

        model = KeyNMF(12, encoder="clip-ViT-B-32")
        model.fit_multimodal(texts, images=images)
        model.plot_multimodal_topics()
        ```


    === "Clustering Models"

        ```python
        from turftopic import ClusteringTopicModel

        # BERTopic-style
        model = ClusteringTopicModel(encoder="clip-ViT-B-32", feature_importance="c-tf-idf")
        # Top2Vec-style
        model = ClusteringTopicModel(encoder="clip-ViT-B-32", feature_importance="centroid")
        model.fit_multimodal(texts, images=images)
        model.plot_multimodal_topics()
        ```

    === "GMM"

        ```python
        from turftopic import GMM

        model = GMM(12, encoder="clip-ViT-B-32")
        model.fit_multimodal(texts, images=images)
        model.plot_multimodal_topics()
        ```

    === "AutoEncodingTopicModel"

        ```python
        from turftopic import AutoEncodingTopicModel

        # CombinedTM
        model = AutoEncodingTopicModel(12, combined=True, encoder="clip-ViT-B-32")
        # ZeroShotTM
        model = AutoEncodingTopicModel(12, combined=False, encoder="clip-ViT-B-32")
        model.fit_multimodal(texts, images=images)
        model.plot_multimodal_topics()
        ```

<iframe src="../images/multimodal.html", title="Multimodal S^3 on IKEA catalogue", style="height:800px;width:1200px;padding:0px;border:none;"></iframe>

## API reference

::: turftopic.multimodal.MultimodalModel

::: turftopic.encoders.multimodal.MultimodalEncoder



