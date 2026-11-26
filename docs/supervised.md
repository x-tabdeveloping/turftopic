# Supervised Topic Modelling

Some topic models are capable of learning topics from labelled documents.
In these cases, instead of the models recovering latent topics using unsupervised learning, the topic signal is provided to the model, and the model learns topic representations from the given signals.

Currently, a handful of models are supported, each of which have a `fit_supervised()` method.
One has to pass a document-topic-matrix of shape `(n_documents, n_topics)` to this method as a `y`.
If you only have access to categorical labels, you can use `turftopic.utils.safe_binarize` to convert them into a one-hot document-topic-matrix.

Here is an example of 20Newsgroups using different topic models:

=== "SensTopic"

    ```python
    from sklearn.datasets import fetch_20newsgroups

    from turftopic import SensTopic
    from turftopic.utils import safe_binarize

    ds = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )
    corpus = ds.data
    target = ds.target
    unique_targets = np.unique(target)
    # We use safe_binarize to convert the class labels to a one-hot matrix:
    y = safe_binarize(target, unique_targets)
    
    model = SensTopic()
    model.fit_supervised(corpus, y=y)
    model.rename_topics(ds.target_names)
    model.print_topics()
    ```

=== "ClusteringTopicModel (by extension BERTopic and Top2Vec too)"

    ```python
    from sklearn.datasets import fetch_20newsgroups

    from turftopic import ClusteringTopicModel
    from turftopic.utils import safe_binarize

    ds = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )
    corpus = ds.data
    target = ds.target
    unique_targets = np.unique(target)
    # We use safe_binarize to convert the class labels to a one-hot matrix:
    y = safe_binarize(target, unique_targets)
    
    model = ClusteringTopicModel()
    model.fit_supervised(corpus, y=y)
    model.rename_topics(ds.target_names)
    model.print_topics()
    ```

=== "SemanticSignalSeparation"

    ```python
    from sklearn.datasets import fetch_20newsgroups

    from turftopic import SemanticSignalSeparation
    from turftopic.utils import safe_binarize

    ds = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )
    corpus = ds.data
    target = ds.target
    unique_targets = np.unique(target)
    # We use safe_binarize to convert the class labels to a one-hot matrix:
    y = safe_binarize(target, unique_targets)
    
    model = SemanticSignalSeparation(10)
    model.fit_supervised(corpus, y=y)
    model.rename_topics(ds.target_names)
    model.print_topics()
    ```

| Topic ID | Topic Name | Highest Ranking |
| - | - | - |
| 0 | alt.atheism | intellectually, sophistication, terminology, hypothesis, atheists, rationality, analogies, theoretical, exists, irrational |
| 1 | comp.graphics | zooming, plots, plotting, procedural, frames, animations, animation, animated, visuals, visualization |
| | ... |
| 18 | talk.politics.misc | hillary, republican, parties, clinton, libertarian, unions, reagan, politician, president, trickle |
| 19 | talk.religion.misc | beneath, ancestor, noble, born, childhood, servants, father, teach, comprised, instructed |
