import re
from typing import Union

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

try:
    import spacy
    from spacy.language import Language
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "SpaCy is not installed on your machine, please run `pip install turftopic[spacy]` before using SpaCy tokenizers."
    ) from e


class NounPhraseCountVectorizer(CountVectorizer):
    """Extracts Noun phrases from text using SpaCy.

    Parameters
    ----------
    nlp: spacy.Language or str, default "en_core_web_sm"
        A Spacy pipeline or its name.
    """

    def __init__(
        self,
        nlp: Union[Language, str] = "en_core_web_sm",
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
        self.nlp = nlp
        if isinstance(nlp, str):
            self._nlp = spacy.load(nlp)
        else:
            self._nlp = nlp
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

    def nounphrase_tokenize(self, text: str) -> list[str]:
        doc = self._nlp(text)
        tokens = []
        for chunk in doc.noun_chunks:
            if chunk[0].is_stop:
                chunk = chunk[1:]
            phrase = chunk.text
            phrase = re.sub(r"[^\w\s]", " ", phrase)
            phrase = " ".join(phrase.split()).strip()
            if phrase:
                tokens.append(phrase)
        return tokens

    def build_tokenizer(self):
        return self.nounphrase_tokenize


class LemmaCountVectorizer(CountVectorizer):
    """Extracts lemmata from text using SpaCy.

    Parameters
    ----------
    nlp: spacy.Language or str, default "en_core_web_sm"
        A Spacy pipeline or its name.
    """

    def __init__(
        self,
        nlp: Union[Language, str] = "en_core_web_sm",
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
        self.nlp = nlp
        if isinstance(nlp, str):
            self._nlp = spacy.load(nlp)
        else:
            self._nlp = nlp
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

    def lemma_tokenize(self, text: str) -> list[str]:
        doc = self._nlp(text)
        tokens = []
        for token in doc:
            if token.is_stop or not token.is_alpha:
                continue
            tokens.append(token.lemma_.strip())
        return tokens

    def build_tokenizer(self):
        return self.lemma_tokenize


class TokenCountVectorizer(CountVectorizer):
    """Tokenizes text with SpaCy using its language-specific tokenization rules and stop-word lists

    Parameters
    ----------
    language_code: str, default "en"
        Language code for the language you intend to use.
    remove_stop_words: bool, default True
        Indicates whether stop words should be removed.
    remove_nonalpha: bool, default True
        Indicates whether only tokens containing alphabetical characters should be kept.
    """

    def __init__(
        self,
        language_code: str = "en",
        remove_stop_words: bool = True,
        remove_nonalpha: bool = True,
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
        self.language_code = language_code
        self.remove_stop_words = remove_stop_words
        self.remove_nonalpha = remove_nonalpha
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
        nlp = spacy.blank(self.language_code)

        def tokenize(text: str) -> list[str]:
            doc = nlp(text)
            result = []
            for tok in doc:
                if self.remove_stop_words and tok.is_stop:
                    continue
                if self.remove_nonalpha and not tok.is_alpha:
                    continue
                result.append(tok.orth_)
            return result

        return tokenize
