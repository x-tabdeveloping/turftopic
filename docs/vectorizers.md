# Vectorizers

One of the most important attributes of a topic model you will have to choose is the vectorizer.
It determines for which terms word-importance scores will be calculated.

By default, Turftopic uses sklearn's [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html),
which naively counts word/n-gram occurrences in text. This usually works quite well, but your use case might require you to use a different or more sophisticated approach.
This is why we provide a `vectorizers` module, where a wide range of useful options is available to you.

## Chinese text

The Chinese language does not separate tokens by whitespace, unlike most Indo-European languages.
You thus need to use special tokenization rules for Chinese.
Turftopic provides tools for Chinese tokenization via the [Jieba](https://github.com/fxsjy/jieba) package.

You will need to install the package in order to be able to use our Chinese vectorizer.

```bash
pip install turftopic[jieba]
```

You can then use the `ChineseCountVectorizer` object, which comes preloaded with the jieba tokenizer along with a Chinese stop word list.

```python
from turftopic import KeyNMF
from turftopic.vectorizers.chinese import ChineseCountVectorizer

vectorizer = ChineseCountVectorizer(min_df=10, stop_words="chinese")

model = KeyNMF(10, vectorizer=vectorizer)
```
