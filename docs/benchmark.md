# Model Leaderboard

To aid you in choosing the best model for your use case, we have made a topic model benchmark and leaderboard.
The benchmark consists of all English P2P clustering tasks from the most recent version of [MTEB](https://huggingface.co/spaces/mteb/leaderboard), plus a tweet and a news dataset, as these are not present in MTEB.

Models were tested for topic quality using the methodology of [Kardos et al. 2025](https://aclanthology.org/2025.acl-long.32/),
and cluster quality using adjusted mutual information (AMI), the Fowlkes-Mallows index (FMI) and V-measure scores.
All models were run on an older, but still powerful Dell Precision laptop, with 32 GBs of RAM, and i7, which was apparently not enough,
as some models ran out of memory on some of the larger datasets.
Due to this, and the fact that the scale of the scores is different for different tasks, we present the **average percentile** scores on these metrics in the table bellow.


??? info "Click to see Benchmark code"
    ```python
    import argparse
    import json
    import time
    from itertools import chain, combinations
    from pathlib import Path
    from typing import Callable, Iterable

    import gensim.downloader as api
    import mteb
    import numpy as np
    from datasets import load_dataset
    from glovpy import GloVe
    from sentence_transformers import SentenceTransformer
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from turftopic import (GMM, AutoEncodingTopicModel, BERTopic, FASTopic, KeyNMF,
                          SemanticSignalSeparation, SensTopic, Top2Vec, Topeax)

    topic_models = {
        "Topeax(Auto)": lambda encoder, n_components: Topeax(
            encoder=encoder, random_state=42
        ),
        "BERTopic(Auto)": lambda encoder, n_components: BERTopic(
            encoder=encoder, random_state=42
        ),
        "Top2Vec(Auto)": lambda encoder, n_components: Top2Vec(
            encoder=encoder, random_state=42
        ),
        "SensTopic(Auto)": lambda encoder, n_components: SensTopic(
            n_components="auto", encoder=encoder, random_state=42
        ),
        "SensTopic": lambda encoder, n_components: SensTopic(
            n_components=n_components, encoder=encoder, random_state=42
        ),
        "KeyNMF(Auto)": lambda encoder, n_components: KeyNMF(
            n_components="auto", encoder=encoder, random_state=42
        ),
        "KeyNMF": lambda encoder, n_components: KeyNMF(
            n_components=n_components, encoder=encoder, random_state=42
        ),
        "GMM": lambda encoder, n_components: GMM(
            n_components=n_components, encoder=encoder, random_state=42
        ),
        "Top2Vec(Reduce)": lambda encoder, n_components: Top2Vec(
            n_reduce_to=n_components, encoder=encoder, random_state=42
        ),
        "BERTopic(Reduce)": lambda encoder, n_components: BERTopic(
            n_reduce_to=n_components, encoder=encoder, random_state=42
        ),
        "ZeroShotTM": lambda encoder, n_components: AutoEncodingTopicModel(
            n_components=n_components, encoder=encoder, random_state=42, combined=False
        ),
        "SemanticSignalSeparation": lambda encoder, n_components: SemanticSignalSeparation(
            n_components=n_components, encoder=encoder, random_state=42
        ),
        "FASTopic": lambda encoder, n_components: FASTopic(
            n_components=n_components, encoder=encoder, random_state=42
        ),
    }


    def load_corpora() -> Iterable[tuple[str, Callable]]:
        mteb_tasks = mteb.get_tasks(
            [
                "ArXivHierarchicalClusteringP2P",
                "BiorxivClusteringP2P.v2",
                "MedrxivClusteringP2P.v2",
                "StackExchangeClusteringP2P.v2",
                "TwentyNewsgroupsClustering.v2",
            ]
        )
        for task in mteb_tasks:

            def _load_dataset():
                task.load_data()
                ds = task.dataset["test"]
                corpus = list(ds["sentences"])
                if isinstance(ds["labels"][0], list):
                    true_labels = [label[0] for label in ds["labels"]]
                else:
                    true_labels = list(ds["labels"])
                return corpus, true_labels

            yield task.metadata.name, _load_dataset

        def _load_dataset():
            # Taken from here cardiffnlp/tweet_topic_single with "train_all"
            ds = load_dataset("kardosdrur/tweet_topic_clustering", split="train_all")
            corpus = list(ds["text"])
            labels = list(ds["label"])
            return corpus, labels

        yield "TweetTopicClustering", _load_dataset

        def _load_dataset():
            ds = load_dataset("gopalkalpande/bbc-news-summary", split="train")
            corpus = list(ds["Summaries"])
            labels = list(ds["File_path"])
            return corpus, labels

        yield "BBCNewsClustering", _load_dataset


    def diversity(keywords: list[list[str]]) -> float:
        all_words = list(chain.from_iterable(keywords))
        unique_words = set(all_words)
        total_words = len(all_words)
        return float(len(unique_words) / total_words)


    def word_embedding_coherence(keywords, wv):
        arrays = []
        for index, topic in enumerate(keywords):
            if len(topic) > 0:
                local_simi = []
                for word1, word2 in combinations(topic, 2):
                    if word1 in wv.index_to_key and word2 in wv.index_to_key:
                        local_simi.append(wv.similarity(word1, word2))
                arrays.append(np.nanmean(local_simi))
        return float(np.nanmean(arrays))


    def evaluate_clustering(true_labels, pred_labels) -> dict[str, float]:
        res = {}
        for metric in [
            metrics.fowlkes_mallows_score,
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.adjusted_mutual_info_score,
        ]:
            res[metric.__name__] = metric(true_labels, pred_labels)
        return res


    def get_keywords(model) -> list[list[str]]:
        """Get top words and ignore outlier topic."""
        n_topics = model.components_.shape[0]
        try:
            classes = model.classes_
        except AttributeError:
            classes = list(range(n_topics))
        res = []
        for topic_id, words in zip(classes, model.get_top_words()):
            if topic_id != -1:
                res.append(words)
        return res


    def evaluate_topic_quality(keywords, ex_wv, in_wv) -> dict[str, float]:
        res = {
            "diversity": diversity(keywords),
            "c_in": word_embedding_coherence(keywords, in_wv),
            "c_ex": word_embedding_coherence(keywords, ex_wv),
        }
        return res


    def load_cache(out_path):
        cache_entries = []
        with out_path.open() as cache_file:
            for line in cache_file:
                entry = json.loads(line.strip())
                cache_entry = (entry["task"], entry["model"])
                cache_entries.append(cache_entry)
        return set(cache_entries)


    def main(encoder_name: str = "all-MiniLM-L6-v2"):
        out_dir = Path("results")
        out_dir.mkdir(exist_ok=True)
        encoder_path_name = encoder_name.replace("/", "__")
        out_path = out_dir.joinpath(f"{encoder_path_name}.jsonl")
        if out_path.is_file():
            cache = load_cache(out_path)
        else:
            cache = set()
            # Create file if doesn't exist
            with out_path.open("w"):
                pass
        print("Loading external word embeddings")
        ex_wv = api.load("word2vec-google-news-300")
        print("Loading benchmark")
        tasks = load_corpora()
        for task_name, load in tasks:
            if all([(task_name, model_name) in cache for model_name in topic_models]):
                print("All models already completed, skipping.")
                continue
            print("Load corpus")
            corpus, true_labels = load()
            print("Training internal word embeddings using GloVe...")
            tokenizer = CountVectorizer().build_analyzer()
            glove = GloVe(vector_size=50)
            tokenized_corpus = [tokenizer(text) for text in corpus]
            glove.train(tokenized_corpus)
            in_wv = glove.wv
            encoder = SentenceTransformer(encoder_name, device="cpu")
            print("Encoding task corpus.")
            embeddings = encoder.encode(corpus, show_progress_bar=True)
            for model_name in topic_models:
                if (task_name, model_name) in cache:
                    print(f"{model_name} already done, skipping.")
                    continue
                print(f"Running {model_name}.")
                true_n = len(set(true_labels))
                model = topic_models[model_name](encoder=encoder, n_components=true_n)
                start_time = time.time()
                doc_topic_matrx = model.fit_transform(corpus, embeddings=embeddings)
                end_time = time.time()
                labels = getattr(model, "labels_", None)
                if labels is None:
                    labels = np.argmax(doc_topic_matrx, axis=1)
                keywords = get_keywords(model)
                print("Evaluating model.")
                clust_scores = evaluate_clustering(true_labels, labels)
                topic_scores = evaluate_topic_quality(keywords, ex_wv, in_wv)
                runtime = end_time - start_time
                res = {
                    "encoder": encoder_name,
                    "task": task_name,
                    "model": model_name,
                    "auto": "(Auto)" in model_name,
                    "runtime": runtime,
                    "dps": len(corpus) / runtime,
                    "n_components": model.components_.shape[0],
                    "true_n": len(set(true_labels)),
                    **clust_scores,
                    **topic_scores,
                }
                print("Results: ", res)
                res["keywords"] = keywords
                with out_path.open("a") as out_file:
                    out_file.write(json.dumps(res) + "\n")


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(prog="Evaluate clustering.")
        parser.add_argument("embedding_model")
        args = parser.parse_args()
        encoder = args.embedding_model
        main(encoder)
        print("DONE")
    ```


<iframe
	src="https://kardosdrur-turftopic-benchmark-table.hf.space"
	frameborder="0"
  style="padding: 0; margin: 0;"
	width="1000"
	height="620"
></iframe>

For models that are able to detect the number of topics, we ran the test with this setting, this is marked as ***(Auto)*** in our tables and plots.
For models, where users can set the number of topics, we also ran the benchmark setting the correct number of topics a-priori.

#### Topic Quality

It seems, that Auto models, and, in particular, Topeax, SensTopic, KeyNMF and GMM were best at generating high quality topics, as can be seen from interpretability scores.
Out of non-auto models, KeyNMF, GMM, ZeroShotTM, FASTopic and SensTopic did best, though ZeroShotTM and FASTopic did not run on some of the more challenging datasets due to running out of memory.

#### Cluster Quality

Clear winners in cluster quality were GMM, Topeax(also GMM-based) and SensTopic. FASTopic also did reasonably well when recovering gold clusters in the data.

<figure >
  <iframe
    src="../images/radar_chart.html"
    frameborder="0"
    style="padding: 0; margin: 0;"
    width="1000px"
    height="520px"
  ></iframe>
  <figcaption>Performance profile of all models on different metrics.
    Top 5 models on average performance are highlighted, click on legend to show the others.
  </figcaption>
</figure>

## Computational Efficiency

<figure style="text-align: center; float: right;">
  <iframe
    src="../images/performance_speed_plot.html"
    frameborder="0"
    style="padding: 0; margin: 0;"
    width="675px"
    height="420px"
  ></iframe>
</figure>

#### Speed

We recorded the amount of documents a model could process per second for each of the runs.
It seems that matrix factorization approaches were fastest ($S^3$, SensTopic, KeyNMF), while neural approaches (FASTopic, ZeroShotTM) the slowest.
While in our investigations, SensTopic seems slower than SemanticSignalSeparation, it is important to note, that SensTopic has built-in JIT compilation capabilities, once JAX is installed, and is therefore likely to be even faster than $S^3$. For more detail, see [SensTopic](SensTopic.md).
We plotted model speed versus performance on the interactive graph to the right. Model size represents the Fowlkes-Mallows Index.

<figure style="text-align: center; float: left;">
  <iframe
    src="../images/oom_error.html"
    frameborder="0"
    style="padding: 0; margin: 0;"
    width="475px"
    height="320px"
  ></iframe>
</figure>

#### Out of Memory

While we did not record memory usage, three models stood out for being unable to complete some of the more challenging tasks on the test hardware.
FASTopic failed twice, on some of the larger corpora, while Top2Vec and BERTopic had problems when trying to reduce the number of topics to a desired amount.
This is likely due to the computational and memory burden of hierarchical clustering, and thus we recommend that you do not use topic reduction if you are unsure whether your hardware will be able to handle it.
If you got your heart set on using FASTopic, we recommend that you get a lot of memory, and preferably a GPU too.
Unfortunately neural topic modelling still takes a lot of resources to run.

## Discovering the Number of Topics

A number of methods are, in theory, able to discover the number of topics in a dataset.
We have tested this, and found that this claim is rather exaggerated, especially in the case of BERTopic and Top2Vec,
which consistently overestimated the number of topics, sometimes by orders of magnitude.
This effect gets worse with larger corpora.
Topeax was the most accurate at this task, mostly when run on larger corpora, but it was still very much off most of the time.
KeyNMF and SensTopic also got reasonably close sometimes, while completely missing the mark in others.

We conclude that this area needs a lot of improvement.

| Model        | ArXivHierarchical (23) | BBCNews (5) | Biorxiv (26) | Medrxiv (51) | StackExchange (524) | TweetTopic (6) | TwentyNewsgroups (20) |
|--------------|--------------------------|-------------|---------------|---------------|-----------------------|----------------|-------------------------|
| BERTopic     | **25**                   | 42          | 602           | 1583          | 2542                  | 76             | 1861                    |
| KeyNMF       | 3                        | **5**       | 250           | 250           | **250**               | <u>2</u>            | <u>10</u>                    |
| SensTopic    | 8                        | <u>6</u>         | <u>14</u>          | <u>14</u>          | 6                     | 11             | 4                       |
| Top2Vec      | <u>18</u>                     | 18          | 405           | 1000          | 1495                  | 49             | 1612                    |
| Topeax       | 6                        | 8           | **19**        | **23**        | <u>21</u>                  | **8**          | **13**                  |

## Cite the Leaderboard

If you intend to reference the Topic Leaderboard in your research, please cite us:

```bibtex
@online{TopicLeaderboard2026,
  title     = {The Topic Model Leaderboard},
  author    = {Márton Kardos},
  year      = 2026,
  url       = {https://x-tabdeveloping.github.io/turftopic/benchmark/},
  urldate   = {2026-02-03}
}
```
