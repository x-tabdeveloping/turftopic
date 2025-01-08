# Vectorizers

One of the most important attributes of a topic model you will have to choose is the vectorizer.
A vectorizer is responsible for extracting term-features from text.
It determines for which terms word-importance scores will be calculated.

By default, Turftopic uses sklearn's [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html),
which naively counts word/n-gram occurrences in text. This usually works quite well, but your use case might require you to use a different or more sophisticated approach.
This is why we provide a `vectorizers` module, where a wide range of useful options is available to you.

!!! question "How is this different from preprocessing?"
    You might think that preprocessing the documents might result in the same effect as some of these vectorizers, but this is not entirely the case.
    When you remove stop words or lemmatize texts in preprocessing, you remove a lot of valuable information that your topic model then can't use.
    By defining a custom vectorizer you *limit the vocabulary* of your model, thereby only learning word importance scores for certain words, but **you keep your documents fully intact**.

## Phrase Vectorizers

You might want to get phrases in your topic descriptions instead of individual words.
This could prove a very reasonable choice as it's often not words in themselves but phrases made up by them that describe a topic most accurately.
Turftopic supports multiple ways of using phrases as fundamental terms.

### N-gram Features with `CountVectorizer`

`CountVectorizer` supports n-gram extraction right out of the box.
Just define a custom vectorizer with an `n_gram_range`.

!!! tip
    While this option is naive, and will likely yield the lowest quality results, it is also incredibly fast in comparison to other phrase vectorization techniques.
    It might, however be slower, if the topic model encodes its vocabulary when fitting.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(2,3), stop_words="english")

model = KeyNMF(10, vectorizer=vectorizer)
model.fit(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | bronx away sank, blew bronx away, blew bronx, bronx away, sank manhattan, stay blew bronx, manhattan sea, away sank manhattan, said queens stay, queens stay |
| 1 | faq alt atheism, alt atheism archive, atheism overview alt, alt atheism resources, atheism faq frequently, archive atheism overview, alt atheism faq, overview alt atheism, titles alt atheism, readers alt atheism |
| 2 | theism factor fanatism, theism leads fanatism, fanatism caused theism, theism correlated fanaticism, fanatism point theism, fanatism deletion theism, fanatics tend theism, fanaticism said fanatism, correlated fanaticism belief, strongly correlated fanaticism |
| 3 | alt atheism, atheism archive, alt atheism archive, archive atheism, atheism atheism, atheism faq, archive atheism introduction, atheism archive introduction, atheism introduction alt, atheism introduction |
| | ... |

### Noun phrases with `NounPhraseCountVectorizer`

Turftopic can also use noun phrases by utilizing the [SpaCy](https://spacy.io/) package.
For Noun phrase vectorization to work, you will have to install SpaCy.

```bash
pip install turftopic[spacy]
```

You will also need to install a relevant SpaCy pipeline for the language you intend to use.
The default pipeline is English, and you should install it before attempting to use `NounPhraseCountVectorizer`.

You can find a model that fits your needs [here](https://spacy.io/models).

```bash
python -m spacy download en_core_web_sm
```

Using SpaCy pipelines will substantially slow down model fitting, but the results might be more correct and higher quality than with naive n-gram extraction.
```python
from turftopic import KeyNMF
from turftopic.vectorizers.spacy import NounPhraseCountVectorizer

model = KeyNMF(
    n_components=10,
    vectorizer=NounPhraseCountVectorizer("en_core_web_sm"),
)
model.fit(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | atheists, atheism, atheist, belief, beliefs, theists, faith, gods, christians, abortion |
| 1 | alt atheism, usenet alt atheism resources, usenet alt atheism introduction, alt atheism faq, newsgroup alt atheism, atheism faq resource txt, alt atheism groups, atheism, atheism faq intro txt, atheist resources |
| 2 | religion, christianity, faith, beliefs, religions, christian, belief, science, cult, justification |
| 3 | fanaticism, theism, fanatism, all fanatism, theists, strong theism, strong atheism, fanatics, precisely some theists, all theism |
| 4 | religion foundation darwin fish bumper stickers, darwin fish, atheism, 3d plastic fish, fish symbol, atheist books, atheist organizations, negative atheism, positive atheism, atheism index |
| | ... |

### Keyphrases with [KeyphraseVectorizers](https://github.com/TimSchopf/KeyphraseVectorizers/tree/master)

You can extract candidate keyphrases from text using KeyphraseVectorizers.
KeyphraseVectorizers uses POS tag patterns to identify phrases instead of word dependency graphs, like `NounPhraseCountVectorizer`.
KeyphraseVectorizers can potentially be faster as the dependency parser component is not needed in the SpaCy pipeline.
This vectorizer is not part of the Turftopic package, but can be easily used with it out of the box.

```bash
pip install keyphrase-vectorizers
```

```python
from keyphrase_vectorizers import KeyphraseCountVectorizer

vectorizer = KeyphraseCountVectorizer()
model = KeyNMF(10, vectorizer=vectorizer).fit(corpus)
```

## Lemmatizing and Stemming Vectorizers

Since the same word can appear in multiple forms in a piece of text, one can sometimes obtain higher quality results by stemming or lemmatizing words in a text before processing them.

!!! warning
    You should **NEVER** lemmatize or stem texts before passing them to a topic model in Turftopic, but rather, use a vectorizer that limits the model's vocabulary to the terms you are interested in.

### Extracting lemmata with `LemmaCountVectorizer`

Similarly to `NounPhraseCountVectorizer`, `LemmaCountVectorizer` relies on a [SpaCy](spacy.io) pipeline for extracting lemmas from a piece of text.
This means you will have to install SpaCy and a SpaCy pipeline to be able to use it.

```bash
pip install turftopic[spacy]
python -m spacy download en_core_web_sm
```

```python
from turftopic import KeyNMF
from turftopic.vectorizers.spacy import LemmaCountVectorizer

model = KeyNMF(10, vectorizer=LemmaCountVectorizer("en_core_web_sm"))
model.fit(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | atheist, theist, belief, christians, agnostic, christian, mythology, asimov, abortion, read |
| 1 | morality, moral, immoral, objective, society, animal, natural, societal, murder, morally |
| 2 | religion, religious, christianity, belief, christian, faith, cult, church, secular, christians |
| 3 | atheism, belief, agnosticism, religious, faq, lack, existence, theism, atheistic, allah |
| 4 | islam, muslim, islamic, rushdie, khomeini, bank, imam, bcci, law, secular |
| | ... |

### Stemming words with `StemmingCountVectorizer`

You might find that lemmatization isn't aggressive enough for your purposes and still many forms of the same word penetrate topic descriptions.
In that case you should try stemming! Stemming is available in Turftopic via the [Snowball Stemmer](https://snowballstem.org/), so it has to be installed before using stemming vectorization.

!!! question "Should I choose stemming or lemmatization?"
    In almost all cases you should **prefer lemmatizaion** over stemming, as it provides higher quality and more correct results. You should only use a stemmer if 

     1. You need something fast (lemmatization is slower due to a more involved pipeline)
     2. You know what you want and it is definitely stemming.
```bash
pip install turftopic[snowball]
```

Then you can initialize a topic model with this vectorizer:

```python
from turftopic import KeyNMF
from turftopic.vectorizers.snowball import StemmingCountVectorizer

model = KeyNMF(10, vectorizer=StemmingCountVectorizer(language="english"))
model.fit(corpus)
model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | atheism, belief, alt, theism, agnostic, stalin, lack, sceptic, exist, faith |
| 1 | religion, belief, religi, cult, faith, theism, secular, theist, scientist, dogma |
| 2 | bronx, manhattan, sank, queen, sea, away, said, com, bob, blew |
| 3 | moral, human, instinct, murder, kill, law, behaviour, action, behavior, ethic |
| 4 | atheist, theist, belief, asimov, philosoph, mytholog, strong, faq, agnostic, weak |
| | | ... |

## Non-English Vectorization

You may find that, especially with non-Indo-European languages, `CountVectorizer` does not perform that well.
In these cases we recommend that you use a vectorizer with its own language-specific tokenization rules and stop-word list:

### Vectorizing Any Language with `TokenCountVectorizer`

The [SpaCy](spacy.io) package includes language-specific tokenization and stop-word rules for just about any language.
We provide a vectorizer that you can use with the language of your choice.

```bash
pip install turftopic[spacy]
```

!!! note
    Note that you do not have to install any SpaCy pipelines for this to work.
    No pipelines or models will be loaded with `TokenCountVectorizer` only a language-specific tokenizer.

```python
from turftopic import KeyNMF
from turftopic.vectorizers.spacy import TokenCountVectorizer

# CountVectorizer for Arabic
vectorizer = TokenCountVectorizer("ar", min_df=10)

model = KeyNMF(
    n_components=10,
    vectorizer=vectorizer,
    encoder="Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet"
)
model.fit(corpus)

```

### Extracting Chinese Tokens with `ChineseCountVectorizer`

The Chinese language does not separate tokens by whitespace, unlike most Indo-European languages.
You thus need to use special tokenization rules for Chinese.
Turftopic provides tools for Chinese tokenization via the [Jieba](https://github.com/fxsjy/jieba) package.

!!! note
    We recommend that you use Jieba over SpaCy for topic modeling with Chinese.

You will need to install the package in order to be able to use our Chinese vectorizer.

```bash
pip install turftopic[jieba]
```

You can then use the `ChineseCountVectorizer` object, which comes preloaded with the jieba tokenizer along with a Chinese stop word list.

```python
from turftopic import KeyNMF
from turftopic.vectorizers.chinese import ChineseCountVectorizer

vectorizer = ChineseCountVectorizer(min_df=10, stop_words="chinese")

model = KeyNMF(10, vectorizer=vectorizer, encoder="BAAI/bge-small-zh-v1.5")
model.fit(corpus)

model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | 消息, 时间, 科技, 媒体报道, 美国, 据, 国外, 讯, 宣布, 称 |
| 1 | 体育讯, 新浪, 球员, 球队, 赛季, 火箭, nba, 已经, 主场, 时间 |
| 2 | 记者, 本报讯, 昨日, 获悉, 新华网, 基金, 通讯员, 采访, 男子, 昨天 |
| 3 | 股, 下跌, 上涨, 震荡, 板块, 大盘, 股指, 涨幅, 沪, 反弹 |
| | ... |

## API Reference

:::turftopic.vectorizers.spacy.NounPhraseCountVectorizer

:::turftopic.vectorizers.spacy.LemmaCountVectorizer

:::turftopic.vectorizers.spacy.TokenCountVectorizer

:::turftopic.vectorizers.snowball.StemmingCountVectorizer

:::turftopic.vectorizers.chinese.ChineseCountVectorizer
