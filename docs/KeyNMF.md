# KeyNMF

KeyNMF is a topic model that relies on contextually sensitive embeddings for keyword retrieval and term importance estimation,
while taking inspiration from classical matrix-decomposition approaches for extracting topics.

<figure>
  <img src="../images/keynmf.png" width="90%" style="margin-left: auto;margin-right: auto;">
  <figcaption>Schematic overview of KeyNMF</figcaption>
</figure>


Here's an example of how you can fit and interpret a KeyNMF model in the easiest way.

```python
from turftopic import KeyNMF

model = KeyNMF(10, encoder="paraphrase-MiniLM-L3-v2")
model.fit(corpus)

model.print_topics()
```

!!! question "Which Embedding model should I use"
    - You should probably use KeyNMF with a `paraphrase-` type embedding model. These seem to perform best in most tasks. Some examples include:
        - [paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2) - Absolutely tiny :mouse: 
        - [paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) - High performance :star2:
        - [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) - Multilingual, high-performance :earth_americas: :star2:
    - KeyNMF works remarkably well with static models, which are incredibly fast, even on your laptop:
        - [sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1) - Blazing Fast :zap: 
        - [sentence-transformers/static-similarity-mrl-multilingual-v1](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) - Multilingual, Blazing Fast :earth_americas: :zap: 

## How does KeyNMF work?

#### Keyword Extraction

KeyNMF discovers topics based on the importances of keywords for a given document.
This is done by embedding words in a document, and then extracting the cosine similarities of documents to words using a transformer-model.
Only the `top_n` keywords with positive similarity are kept.

??? info "Click to see formula"
    - For each document $d$:
        1. Let $x_d$ be the document's embedding produced with the encoder model.
        2. For each word $w$ in the document $d$:
            1. Let $v_w$ be the word's embedding produced with the encoder model.
            2. Calculate cosine similarity between word and document

            $$
            \text{sim}(d, w) = \frac{x_d \cdot v_w}{||x_d|| \cdot ||v_w||}
            $$

        3. Let $K_d$ be the set of $N$ keywords with the highest cosine similarity to document $d$.

        $$
        K_d = \text{argmax}_{K^*} \sum_{w \in K^*}\text{sim}(d,w)\text{, where }
        |K_d| = N\text{, and } \\
        w \in d
        $$

    - Arrange positive keyword similarities into a keyword matrix $M$ where the rows represent documents, and columns represent unique keywords.

        $$
        M_{dw} = 
        \begin{cases}
        \text{sim}(d,w), & \text{if } w \in K_d \text{ and } \text{sim}(d,w) > 0 \\
        0, & \text{otherwise}.
        \end{cases}
        $$

You can do this step manually if you want to precompute the keyword matrix.
Keywords are represented as dictionaries mapping words to keyword importances.

```python
model.extract_keywords(["Cars are perhaps the most important invention of the last couple of centuries. They have revolutionized transportation in many ways."])
```

```python
[{'transportation': 0.44713873,
  'invention': 0.560524,
  'cars': 0.5046208,
  'revolutionized': 0.3339205,
  'important': 0.21803442}]
```

A precomputed Keyword matrix can also be used to fit a model:

```python
keyword_matrix = model.extract_keywords(corpus)
model.fit(None, keywords=keyword_matrix)
```

#### Topic Discovery

Topics in this matrix are then discovered using Non-negative Matrix Factorization.
Essentially the model tries to discover underlying dimensions/factors along which most of the variance in term importance
can be explained.

??? info "Click to see formula"

    - Decompose $M$ with non-negative matrix factorization: $M \approx WH$, where $W$ is the document-topic matrix, and $H$ is the topic-term matrix. Non-negative Matrix Factorization is done with the coordinate-descent algorithm, minimizing square loss:

        $$
        L(W,H) = ||M - WH||^2
        $$

    You can fit KeyNMF on the raw corpus, with precomputed embeddings or with precomputed keywords.


=== "Fitting on a corpus"
    ```python
    model.fit(corpus)
    ```

=== "Pre-computed embeddings"
    ```python
    from sentence_transformers import SentenceTransformer

    trf = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = trf.encode(corpus)

    model = KeyNMF(10, encoder=trf)
    model.fit(corpus, embeddings=embeddings)
    ```
=== "Pre-computed keyword matrix"
    ```python
    keyword_matrix = model.extract_keywords(corpus)
    model.fit(None, keywords=keyword_matrix)
    ```

## Seeded Topic Modeling

When investigating a set of documents, you might already have an idea about what aspects you would like to explore.
In KeyNMF, you can describe this aspect, from which you want to investigate your corpus, using a free-text seed-phrase,
which will then be used to only extract topics, which are relevant to your research question.

??? info "How is this done?"

    KeyNMF encodes the seed phrase into a seed-embedding.
    Word importance scores in a document get weighted by their similarity to the seed-embedding.

    - Embed seed-phrase into a seed-embedding: $s$
    - When extracting keywords from a document:
        1. Let $x_d$ be the document's embedding produced with the encoder model.
        2. Let the document's relevance be $r_d = \text{sim}(d,w)$
        3. For each word $w$:
            1. Let the word's importance in the keyword matrix be: $\text{sim}(d, w) \cdot r_d$ if $r_d > 0$, otherwise $0$

```python
from turftopic import KeyNMF

model = KeyNMF(5, seed_phrase="<your seed phrase>")
model.fit(corpus)

model.print_topics()
```


=== "`'Is the death penalty moral?'`"

    | Topic ID | Highest Ranking |
    | - | - |
    | 0 | morality, moral, immoral, morals, objective, morally, animals, society, species, behavior |
    | 1 | armenian, armenians, genocide, armenia, turkish, turks, soviet, massacre, azerbaijan, kurdish |
    | 2 | murder, punishment, death, innocent, penalty, kill, crime, moral, criminals, executed |
    | 3 | gun, guns, firearms, crime, handgun, firearm, weapons, handguns, law, criminals |
    | 4 | jews, israeli, israel, god, jewish, christians, sin, christian, palestinians, christianity |

=== "`'Evidence for the existence of god'`"

    | Topic ID | Highest Ranking |
    | - | - |
    | 0 | atheist, atheists, religion, religious, theists, beliefs, christianity, christian, religions, agnostic |
    | 1 | bible, christians, christian, christianity, church, scripture, religion, jesus, faith, biblical |
    | 2 | god, existence, exist, exists, universe, creation, argument, creator, believe, life |
    | 3 | believe, faith, belief, evidence, blindly, believing, gods, believed, beliefs, convince |
    | 4 | atheism, atheists, agnosticism, belief, arguments, believe, existence, alt, believing, argument |

=== "`'Operating system kernels'`"

    | Topic ID | Highest Ranking |
    | - | - |
    | 0 | windows, dos, os, microsoft, ms, apps, pc, nt, file, shareware |
    | 1 | ram, motherboard, card, monitor, memory, cpu, vga, mhz, bios, intel |
    | 2 | unix, os, linux, intel, systems, programming, applications, compiler, software, platform |
    | 3 | disk, scsi, disks, drive, floppy, drives, dos, controller, cd, boot |
    | 4 | software, mac, hardware, ibm, graphics, apple, computer, pc, modem, program |



## Dynamic Topic Modeling

KeyNMF is also capable of modeling topics over time.
This happens by fitting a KeyNMF model first on the entire corpus, then
fitting individual topic-term matrices using coordinate descent based on the document-topic and document-term matrices in the given time slices.

??? info "Click to see formula"

    1. Compute keyword matrix $M$ for the whole corpus.
    2. Decompose $M$ with non-negative matrix factorization: $M \approx WH$.
    3. For each time slice $t$:
        1. Let $W_t$ be the document-topic proportions for documents in time slice $t$, and $M_t$ be the keyword matrix for words in time slice $t$.
        2. Obtain the topic-term matrix for the time slice, by minimizing square loss using coordinate descent and fixing $W_t$:

        $$
        H_t = \text{argmin}_{H^{*}} ||M_t - W_t H^{*}||^2
        $$

Here's an example of using KeyNMF in a dynamic modeling setting:

```python
from datetime import datetime

from turftopic import KeyNMF

corpus: list[str] = []
timestamps: list[datetime] = []

model = KeyNMF(5, top_n=5, random_state=42)
document_topic_matrix = model.fit_transform_dynamic(
    corpus, timestamps=timestamps, bins=10
)
```

You can use the `print_topics_over_time()` method for producing a table of the topics over the generated time slices.

> This example uses CNN news data.

```python
model.print_topics_over_time()
```

<center>

| Time Slice | 0_olympics_tokyo_athletes_beijing | 1_covid_vaccine_pandemic_coronavirus | 2_olympic_athletes_ioc_athlete | 3_djokovic_novak_tennis_federer | 4_ronaldo_cristiano_messi_manchester |
| - | - | - | - | - | - |
| 2012 12 06 - 2013 11 10 | genocide, yugoslavia, karadzic, facts, cnn | cnn, russia, chechnya, prince, merkel | france, cnn, francois, hollande, bike | tennis, tournament, wimbledon, grass, courts | beckham, soccer, retired, david, learn |
| 2013 11 10 - 2014 10 14 | keith, stones, richards, musician, author | georgia, russia, conflict, 2008, cnn | civil, rights, hear, why, should | cnn, kidneys, traffickers, organ, nepal | ronaldo, cristiano, goalscorer, soccer, player |
|  |  | ... |  |  |  |
| 2020 05 07 - 2021 04 10 | olympics, beijing, xinjiang, ioc, boycott | covid, vaccine, coronavirus, pandemic, vaccination | olympic, japan, medalist, canceled, tokyo | djokovic, novak, tennis, federer, masterclass | ronaldo, cristiano, messi, juventus, barcelona |
| 2021 04 10 - 2022 03 16 | olympics, tokyo, athletes, beijing, medal | covid, pandemic, vaccine, vaccinated, coronavirus | olympic, athletes, ioc, medal, athlete | djokovic, novak, tennis, wimbledon, federer | ronaldo, cristiano, messi, manchester, scored |

</center>

You can also display the topics over time on an interactive HTML figure.
The most important words for topics get revealed by hovering over them.

> You will need to install Plotly for this to work.

```bash
pip install plotly
```

```python
model.plot_topics_over_time()
```

<figure>
  <iframe src="../images/dynamic_keynmf.html", title="Topics over time", style="height:800px;width:1000px;padding:0px;border:none;"></iframe>
  <figcaption> Topics over time in a Dynamic KeyNMF model. </figcaption>
</figure>

## Hierarchical Topic Modeling

When you suspect that subtopics might be present in the topics you find with the model, KeyNMF can be used to discover topics further down the hierarchy.

This is done by utilising a special case of **weighted NMF**, where documents are weighted by how high they score on the parent topic.

??? info "Click to see formula"
    1. Decompose keyword matrix $M \approx WH$
    2. To find subtopics in topic $j$, define document weights $w$ as the $j$th column of $W$.
    3. Estimate subcomponents with **wNMF** $M \approx \mathring{W} \mathring{H}$ with document weight $w$
        1. Initialise $\mathring{H}$ and  $\mathring{W}$ randomly.
        2. Perform multiplicative updates until convergence. <br>
            $\mathring{W}^T = \mathring{W}^T \odot \frac{\mathring{H} \cdot (M^T \odot w)}{\mathring{H} \cdot \mathring{H}^T \cdot (\mathring{W}^T \odot w)}$ <br>
            $\mathring{H}^T = \mathring{H}^T \odot \frac{ (M^T \odot w)\cdot \mathring{W}}{\mathring{H}^T \cdot (\mathring{W}^T \odot w) \cdot \mathring{W}}$
    4. To sufficiently differentiate the subcomponents from each other a pseudo-c-tf-idf weighting scheme is applied to $\mathring{H}$:
        1. $\mathring{H} = \mathring{H}_{ij} \odot ln(1 + \frac{A}{1+\sum_k \mathring{H}_{kj}})$, where $A$ is the average of all elements in $\mathring{H}$

To create a hierarchical model, you can use the `hierarchy` property of the model.

```python
# This divides each of the topics in the model to 3 subtopics.
model.hierarchy.divide_children(n_subtopics=3)
print(model.hierarchy)
```

<div style="background-color: #F5F5F5; padding: 10px; padding-left: 20px; padding-right: 20px;">
<tt style="font-size: 11pt">
<b>Root </b><br>
├── <b style="color: blue">0</b>: windows, dos, os, disk, card, drivers, file, pc, files, microsoft <br>
│   ├── <b style="color: magenta">0.0</b>: dos, file, disk, files, program, windows, disks, shareware, norton, memory <br>
│   ├── <b style="color: magenta">0.1</b>: os, unix, windows, microsoft, apps, nt, ibm, ms, os2, platform <br>
│   └── <b style="color: magenta">0.2</b>: card, drivers, monitor, driver, vga, ram, motherboard, cards, graphics, ati <br>
└── <b style="color: blue">1</b>: atheism, atheist, atheists, religion, christians, religious, belief, christian, god, beliefs <br>
.    ├── <b style="color: magenta">1.0</b>: atheism, alt, newsgroup, reading, faq, islam, questions, read, newsgroups, readers <br>
.    ├── <b style="color: magenta">1.1</b>: atheists, atheist, belief, theists, beliefs, religious, religion, agnostic, gods, religions <br>
.    └── <b style="color: magenta">1.2</b>: morality, bible, christian, christians, moral, christianity, biblical, immoral, god, religion <br>
</tt>
</div>

For a detailed tutorial on hierarchical modeling click [here](hierarchical.md).

## Cross-lingual KeyNMF

KeyNMF, by default, does not come with cross-lingual capabilities, since only words that appear in a document can be assigned to it as keywords.
We, however provide a term-matching scheme that allows you to match words across languages based on their cosine similarity in a multilingual embedding model.

This is done by:

1. Computing a similarity matrix over terms.
2. Checking, which terms have similarity over a given threshold (_0.9_ is the default)
3. Building a graph from these connections, and finding graph components.
4. Adding up term importances for terms that appear in the same component for all documents.

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

## Online Topic Modeling

KeyNMF can also be fitted in an online manner.
This is done by fitting NMF with batches of data instead of the whole dataset at once.

#### Use Cases:

1. You can use online fitting when you have **very large corpora** at hand, and it would be impractical to fit a model on it at once.
2. You have **new data flowing in constantly**, and need a model that can morph the topics based on the incoming data. You can also do this in a dynamic fashion.
3. You need to **finetune** an already fitted topic model to novel data.

#### Batch Fitting

We will use the batching function from the itertools recipes to produce batches.

> In newer versions of Python (>=3.12) you can just `from itertools import batched`

```python
def batched(iterable, n: int):
    "Batch data into lists of length n. The last batch may be shorter."
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
```

You can fit a KeyNMF model to a very large corpus in batches like so:

```python
from turftopic import KeyNMF

model = KeyNMF(10, top_n=5)

corpus = ["some string", "etc", ...]
for batch in batched(corpus, 200):
    batch = list(batch)
    model.partial_fit(batch)
```

#### Precomputing the Keyword Matrix

If you desire the best results, it might make sense for you to go over the corpus in multiple epochs:

```python
for epoch in range(5):
    for batch in batched(corpus, 200):
        model.partial_fit(batch)
```

This is mildly inefficient, however, as the texts need to be encoded on every epoch, and keywords need to be extracted.
In such scenarios you might want to precompute and maybe even save the extracted keywords to disk using the `extract_keywords()` method.

Keywords are represented as dictionaries mapping words to keyword importances.

```python
model.extract_keywords(["Cars are perhaps the most important invention of the last couple of centuries. They have revolutionized transportation in many ways."])
```

```python
[{'transportation': 0.44713873,
  'invention': 0.560524,
  'cars': 0.5046208,
  'revolutionized': 0.3339205,
  'important': 0.21803442}]
```

You can extract keywords in batches and save them to disk to a file format of your choice.
In this example I will use NDJSON because of its simplicity.

```python
import json
from pathlib import Path
from typing import Iterable

# Here we are saving keywords to a JSONL/NDJSON file
with Path("keywords.jsonl").open("w") as keyword_file:
    # Doing this in batches is much more efficient than individual texts because
    # of the encoding.
    for batch in batched(corpus, 200):
        batch_keywords = model.extract_keywords(batch)
        # We serialize each
        for keywords in batch_keywords:
            keyword_file.write(json.dumps(keywords) + "\n")

def stream_keywords() -> Iterable[dict[str, float]]:
    """This function streams keywords from the file."""
    with Path("keywords.jsonl").open() as keyword_file:
        for line in keyword_file:
            yield json.loads(line.strip())

for epoch in range(5):
    keyword_stream = stream_keywords()
    for keyword_batch in batched(keyword_stream, 200):
        model.partial_fit(keywords=keyword_batch)
```

### Dynamic Online Topic Modeling

KeyNMF can be online fitted in a dynamic manner as well.
This is useful when you have large corpora of text over time, or when you want to fit the model on future information flowing in
and want to analyze the topics' changes over time.

When using dynamic online topic modeling you have to predefine the time bins that you will use, as the model can't infer these from the data.

```python
from datetime import datetime

# We will bin by years in a period of 2020-2030
bins = [datetime(year=y, month=1, day=1) for y in range(2020, 2030 + 2, 1)]
```

You can then online fit a dynamic topic model with `partial_fit_dynamic()`.

```python
model = KeyNMF(5, top_n=10)

corpus: list[str] = [...]
timestamps: list[datetime] = [...]

for batch in batched(zip(corpus, timestamps)):
    text_batch, ts_batch = zip(*batch)
    model.partial_fit_dynamic(text_batch, timestamps=ts_batch, bins=bins)
```

## Asymmetric and Instruction-tuned Embedding Models

Some embedding models can be used together with prompting, or encode queries and passages differently.
This is important for KeyNMF, as it is explicitly based on keyword retrieval, and its performance can be substantially enhanced by using asymmetric or prompted embeddings.
Microsoft's E5 models are, for instance, all prompted by default, and it would be detrimental to performance not to do so yourself.

In these cases, you're better off NOT passing a string to Turftopic models, but explicitly loading the model using `sentence-transformers`.

Here's an example of using instruct models for keyword retrieval with KeyNMF.
In this case, documents will serve as the queries and words as the passages:

```python
from turftopic import KeyNMF
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer(
    "intfloat/multilingual-e5-large-instruct",
    prompts={
        "query": "Instruct: Retrieve relevant keywords from the given document. Query: "
        "passage": "Passage: "
    },
    # Make sure to set default prompt to query!
    default_prompt_name="query",
)
model = KeyNMF(10, encoder=encoder)
```

And a regular, asymmetric example:

```python
encoder = SentenceTransformer(
    "intfloat/e5-large-v2",
    prompts={
        "query": "query: "
        "passage": "passage: "
    },
    # Make sure to set default prompt to query!
    default_prompt_name="query",
)
model = KeyNMF(10, encoder=encoder)
```

Setting the default prompt to `query` is especially important, when you are precomputing embeddings, as `query` should always be your default prompt to embed documents with.

## API Reference

::: turftopic.models.keynmf.KeyNMF
