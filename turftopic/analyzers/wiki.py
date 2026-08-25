import numpy as np
import re
from itertools import zip_longest
from turftopic.analyzers.base import Analyzer, AnalysisResults
from sklearn.metrics.pairwise import cosine_similarity
from turftopic.serialization import get_package_versions
from rich.progress import track

URL = "https://{language_code}.wikipedia.org/w/api.php"
VERSIONS = get_package_versions()
VERSION = VERSIONS["turftopic"]

CLEANR = re.compile("<.*?>")

HEADERS = {
    "User-Agent": f"TurftopicBot/0.1 (martonkardos@cas.au.dk) turftopic/{VERSION}"
}


def remove_html(text: str):
    cleantext = re.sub(CLEANR, "", text)
    return cleantext


def remove_parens(s):
    return re.sub(r"\([^)]*\)", "", s)


class WikiAnalyzer(Analyzer):
    """Analyze topic model with a page titles and summaries from Wikipedia's API.
    The analyzer searches wikipedia with the highest rankning N keywords from a topic
    and then ranks pages based on their semantic proximity to example keywords and documents
    from the topic using the topic model's encoder.

    Parameters
    ----------
    topic_model: ContextualModel
        Topic model to use for embedding keywords and documents.
    language_code: str, default "en"
        Wikipedia language code for the language of the documents.
    n_keywords: int, default 5
        Number of search words to use when searching Wikipedia.
    similarity_threshold: float = 0.7
        Cosine similarity threshold between page titles and topic representations
        to consider the page a match.
    limit: int, default 10
        Maximum number of pages to return in each search.
    prune_summaries, default True,
        Indicates whether only the first sentence should be used from the page summaries.
    """

    use_summaries = False

    def __init__(
        self,
        topic_model,
        language_code: str = "en",
        n_keywords: int = 5,
        similarity_threshold: float = 0.5,
        limit: int = 10,
        prune_summaries=True,
        penalize_length=True,
    ):
        import requests

        self.session = requests.Session()
        self.topic_model = topic_model
        self.n_keywords = n_keywords
        self.similarity_threshold = similarity_threshold
        self.limit = limit
        self.prune_summaries = prune_summaries
        self.language_code = language_code
        self.penalize_length = penalize_length

    def summarize_document(self, document: str) -> str:
        raise NotImplementedError

    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError

    def _search_page(self, keywords: list[str]):
        query = " ".join(keywords[: self.n_keywords])
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": self.limit,
        }
        results = self.session.get(
            url=URL.format(language_code=self.language_code),
            params=params,
            headers=HEADERS,
        )
        try:
            data = results.json()
            return data["query"]["search"]
        except Exception:
            return []

    def _get_summary(self, pageid):
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "wiki",
            "exintro": 1,
            "pageids": pageid,
        }
        results = self.session.get(
            url=URL.format(language_code=self.language_code),
            params=params,
            headers=HEADERS,
        )
        try:
            data = results.json()
            pages = data["query"]["pages"]
            summary = pages[str(pageid)]["extract"]
            if self.prune_summaries:
                summary = summary.split(".")[0] + "."
            return summary
        except Exception:
            return None

    def _get_topic_embedding(
        self, keywords: list[str], documents: list[str] = None
    ):
        repr_str = list(keywords)
        if documents is not None:
            repr_str.extend(documents)
        embeddings = self.topic_model.encode_documents(repr_str)
        return np.mean(embeddings, axis=0)

    def _get_best_match(
        self, keywords: list[str], documents: list[str] = None
    ):
        search_results = self._search_page(keywords)
        search_results = [
            entry
            for entry in search_results
            if len(remove_parens(entry["title"]).split()) < 5
        ]
        if not search_results:
            return None
        titles = [entry["title"] for entry in search_results]
        snippets = [remove_html(entry["snippet"]) for entry in search_results]
        repr_str = titles
        topic_embedding = self._get_topic_embedding(keywords, documents)
        page_embeddings = self.topic_model.encode_documents(repr_str)
        sim = cosine_similarity([topic_embedding], page_embeddings)[0]
        threshold = self.similarity_threshold
        if self.penalize_length:
            lengths = np.array([len(title.split()) for title in titles])
            sim = sim / lengths
            threshold = threshold / np.max(lengths)
        i_best_page = np.argmax(sim)
        if sim[i_best_page] < self.similarity_threshold:
            return None
        return dict(
            name=remove_parens(titles[i_best_page]),
            snippet=snippets[i_best_page],
            pageid=search_results[i_best_page]["pageid"],
        )

    def describe_topic(
        self,
        keywords: list[str],
        documents=None,
    ):
        """Gives abstract summarization of topic content."""
        best_match = self._get_best_match(keywords)
        if best_match is None:
            return None
        return self._get_summary(best_match["pageid"])

    def name_topic(
        self,
        keywords: list[str],
        documents=None,
    ) -> str:
        """Names one topic based on top descriptive aspects."""
        best_match = self._get_best_match(keywords, documents)
        if best_match is None:
            return None
        return best_match["name"]

    def analyze_topics(
        self,
        keywords: list[list[str]],
        documents: list[list[str]] = None,
        use_summaries=None,
    ) -> AnalysisResults:
        """
        Parameters
        ----------
        keywords: list[list[str]]
            Keywords for each topic.
        documents: list[list[str]], default None
            Top documents for each topic.
        use_summaries: None
            Ignored.

        Returns
        -------
        dict
            Dictionary containing `topic_names`, `topic_descriptions` and `document_summaries` if relevant.
        """
        output = {"topic_names": [], "topic_descriptions": []}
        if documents is None:
            for keys in track(keywords, description="Analyzing topics..."):
                best_match = self._get_best_match(keys)
                if best_match is None:
                    output["topic_names"].append(None)
                    output["topic_descriptions"].append(None)
                    continue
                output["topic_names"].append(best_match["name"])
                summary = self._get_summary(best_match["pageid"])
                output["topic_descriptions"].append(summary)
        else:
            for keys, docs in track(
                zip_longest(keywords, documents),
                description="Analyzing topics...",
                total=len(keywords),
            ):
                best_match = self._get_best_match(keys, docs)
                if best_match is None:
                    output["topic_names"].append(None)
                    output["topic_descriptions"].append(None)
                    continue
                output["topic_names"].append(best_match["name"])
                summary = self._get_summary(best_match["pageid"])
                output["topic_descriptions"].append(summary)
        return AnalysisResults(**output)
