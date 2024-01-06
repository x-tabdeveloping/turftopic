from sklearn.feature_extraction.text import CountVectorizer


def default_vectorizer() -> CountVectorizer:
    return CountVectorizer(min_df=10, stop_words="english")
