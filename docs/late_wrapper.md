The `LateWrapper` class can turn any regular topic model into a late-interaction model.
This might be more or less advantageous depending on the model.

Here's how `LateWrapper` works:

  1. It encodes documents at a token or window-level based on its parameters.
  2. It flattens the span embedding array, and feeds the this into the topic model, along with the token/window text.
  3. It unflattens the output of the topic model (`doc_topic_matrix`) into a ragged array, where you get topic importance for each span.

Let's see how this works in practice, and create different types of models and late interaction:

=== "Topeax with Sliding Windows"
    ```python
    from sklearn.datasets import fetch_20newsgroups
    from turftopic import Topeax
    from turftopic.late import LateWrapper

    corpus = fetch_20newsgroups(subset="all", categories=["alt.atheism"]).data

    model = LateWrapper(
        Topeax(),
        window_size=50, # If we don't specify window size, it will use token-level embeddings
        step_size=40, # Since the step size is smaller than the window, we will get overlapping windows
    )
    doc_topic_matrix, offsets = model.fit_transform(corpus)
    model.print_topics()
    ```

=== "Sentence-level SensTopic"
    ```python
    from sklearn.datasets import fetch_20newsgroups
    from turftopic import SensTopic
    from turftopic.late import LateWrapper

    corpus = fetch_20newsgroups(subset="all", categories=["alt.atheism"]).data

    model = LateWrapper(
        SensTopic(),
        span_extractor="sentence",
    )
    doc_topic_matrix, offsets = model.fit_transform(corpus)
    model.print_topics()
    ```

=== "Token-level Clustering model"
    ```python
    from sklearn.datasets import fetch_20newsgroups
    from turftopic import ClusteringTopicModel
    from turftopic.late import LateWrapper

    corpus = fetch_20newsgroups(subset="all", categories=["alt.atheism"]).data

    model = LateWrapper(
        ClusteringTopicModel(n_reduce_to=10, feature_importance="centroid", reuction_method="smallest")),
    )
    doc_topic_matrix, offsets = model.fit_transform(corpus)
    model.print_topics()
    ```

| Topic ID | Highest Ranking |
| - | - |
| 0 | morality, moral, morals, immoral, objective, behavior, instinctive, species, inherent, animals |
| 1 | matthew, luke, bible, text, passages, mormon, texts, translations, copy, john |
| 2 | atheism, agnostics, atheist, beliefs, belief, faith, contradictory, believers, contradictions, theists |
| 3 | punishment, cruel, abortion, penalty, death, constitution, homosexuality, painless, capital, punish |
| 4 | war, arms, invaded, gulf, hussein, civilians, military, kuwait, peace, sell |
| 5 | islam, islamic, muslim, qur, muslims, imams, rushdie, quran, koran, khomeini |

The document-topic matrix, we created, is now a ragged array and contains document-topic proportions for each window in a document.
Let's see what this means in practice for the first document in our corpus:
```python
import pandas as pd

# We select document 0, then collect all information into a dataframe:
window_topic_matrix = doc_topic_matrix[0]
window_offs = offsets[0]
document = corpus[0]
# We extract the text for each window based on the offsets
window_text = [document[window_start: window_end] for window_start, window_end in window_offs]
df = pd.DataFrame(window_topic_matrix, index=window_text, columns=model.topic_names)
print(df)
```

```python
                                                    0_morality_moral_morals_immoral  1_matthew_luke_bible_text  ...  4_war_arms_invaded_gulf  5_islam_islamic_muslim_qur
From: acooper@mac.cc.macalstr.edu (Turin Turamb...                         0.334267               1.287207e-13  ...             2.626869e-26                1.459101e-04
alester College\nLines: 55\n\nIn article <C5sA2...                         0.360400               8.898302e-14  ...             3.290858e-26                1.382718e-04
u (Mike Cobb) writes:\n> I guess I'm delving in...                         0.847002               5.002921e-22  ...             4.852574e-41                3.141366e-07
this you just have a spiral.  What\nwould then ...                         0.848413               5.819050e-22  ...             8.139559e-41                3.286224e-07
, even though this would hardly seem moral.  Fo...                         0.863685               1.272204e-21  ...             2.823941e-41                2.815930e-07
whatever helps this goal is\n"moral", whatever ...                         0.864913               1.584558e-21  ...             5.780971e-41                3.003952e-07
a "hyper-morality" to apply to just the methods...                         0.865558               1.919885e-21  ...             1.251694e-40                3.231265e-07
not doing something because it is\n> a personal...                         0.868360               2.951441e-21  ...             3.085662e-40                3.494368e-07
we only consider something moral or immoral if ...                         0.872827               5.444738e-21  ...             4.708349e-40                3.580695e-07
here we have a way to discriminate\nmorals.  I ...                         0.876951               1.021014e-20  ...             3.486096e-40                3.411401e-07
enough and\nlistened to the arguments, I could ...                         0.878680               2.302363e-20  ...             5.866410e-40                3.565728e-07
.  Or, as you brought out,\n> if whatever is ri...                         0.878953               3.004052e-20  ...             5.977738e-40                3.566668e-07
> *******************************                                          0.647793               5.664651e-17  ...             1.805073e-19                4.612731e-04
```

## API Reference

::: turftopic.late.LateWrapper

