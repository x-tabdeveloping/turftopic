import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

try:
    import snowballstemmer
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Snowball is not installed on your machine, please run `pip install turftopic[snowball]` before using the snowball stemmer."
    ) from e


class StemmingCountVectorizer(CountVectorizer):
    """Extractes stemmed words from documents using Snowball."""

    def __init__(
        self,
        language="english",
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
    ):
        self.language = language
        self._stemmer = snowballstemmer.stemmer(self.language)
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            analyzer=analyzer,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

    def build_tokenizer(self):
        super_tokenizer = super().build_tokenizer()

        def tokenizer(text):
            return self._stemmer.stemWords(super_tokenizer(text))

        return tokenizer
