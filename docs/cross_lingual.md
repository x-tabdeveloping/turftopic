# Cross-lingual Topic Modeling

Under certain circumstances you might want to run a topic model on a multilingual corpus, where you do not want the model to capture language-differences.
In these cases we recommend that you turn to cross-lingual topic modeling.

## Natively multilingual models
Some topic models in Turftopic support cross-lingual modeling by default.
The only difference is that you will have to choose a multilingual encoder model to produce document embeddings (consult [MTEB(Multilingual)](http://mteb-leaderboard.hf.space/?benchmark_name=MTEB%28Multilingual%2C+v1%29) to find an encoder for your use case).

=== "`SemanticSignalSeparation`"

    ```python
    from turftopic import SemanticSignalSeparation

    model = SemanticSignalSeparation(10, encoder="paraphrase-multilingual-MiniLM-L12-v2")
    ```

=== "`ClusteringTopicModel`"
    ```python
    from turftopic import ClusteringTopicModel

    model = ClusteringTopicModel(encoder="paraphrase-multilingual-MiniLM-L12-v2")
    ```

=== "`AutoEncodingTopicModel(combined=False)`"

    ```python
    from turftopic import AutoEncodingTopicModel

    model = AutoEncodingTopicModel(combined=False, encoder="paraphrase-multilingual-MiniLM-L12-v2")
    ```

=== "`GMM`"

    ```python
    from turftopic import GMM

    model = GMM(encoder="paraphrase-multilingual-MiniLM-L12-v2")
    ```


## Term Matching

Other models do not support cross-lingual use out of the box, and therefore need assistance to be applicable in a multilingual context.

[KeyNMF](KeyNMF.md) can use a trick called term-matching, in which terms that are highly similar get merged into the same term, thereby allowing for one term representing the same word in multiple languages:

!!! note
    Term matching is an experimental feature in Turftopic, and might be improved or extended to more models in the future.

```python
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer

from turftopic import KeyNMF

# Loading a parallel corpus
ds = load_dataset(
    "aiana94/polynews-parallel", "deu_Latn-eng_Latn", split="train"
)
# Subsampling
ds = ds.train_test_split(test_size=1000)["test"]
corpus = ds["src"] + ds["tgt"]

model = KeyNMF(
    10,
    cross_lingual=True,
    encoder="paraphrase-multilingual-MiniLM-L12-v2",
    vectorizer=CountVectorizer()
)
model.fit(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| ... | |
| 15 | africa-afrikanisch-african, media-medien-medienwirksam, schwarzwald-negroe-schwarzer, apartheid, difficulties-complicated-problems, kontinents-continent-kontinent, äthiopien-ethiopia, investitionen-investiert-investierenden, satire-satirical, hundred-100-1001 |
| 16 | lawmaker-judges-gesetzliche, schutz-sicherheit-geschützt, an-success-eintreten, australian-australien-australischen, appeal-appealing-appeals, lawyer-lawyers-attorney, regeln-rule-rules, öffentlichkeit-öffentliche-publicly, terrorism-terroristischer-terrorismus, convicted |
| 17 | israels-israel-israeli, palästinensischen-palestinians-palestine, gay-lgbtq-gays, david, blockaden-blockades-blockade, stars-star-stelle, aviv, bombardieren-bombenexplosion-bombing, militärischer-army-military, kampfflugzeuge-warplanes |
| 18 | russischer-russlands-russischen, facebookbeitrag-facebook-facebooks, soziale-gesellschaftliche-sozialbauten, internetnutzer-internet, activism-aktivisten-activists, webseiten-web-site, isis, netzwerken-networks-netzwerk, vkontakte, media-medien-medienwirksam |
| 19 | bundesstaates-regierenden-regiert, chinesischen-chinesische-chinesisch, präsidentschaft-presidential-president, regions-region-regionen, demokratien-democratic-democracy, kapitalismus-capitalist-capitalism, staatsbürgerin-citizens-bürger, jemen-jemenitische-yemen, angolanischen-angola, media-medien-medienwirksam |

