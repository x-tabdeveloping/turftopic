from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.progress import track

NAMER_PROMPT = """
You will be tasked with naming a topic.
Based on the provided information, create a short label that best summarizes the topics.
Only respond with a short, human readable topic name and nothing else.

The topic is described by the following set of keywords: {keywords}.
"""

DESCRIPTION_PROMPT = """
Based on the provided information, summarize the content of the topic in a couple of sentences.
Do NOT name the topic, just describe its contents.

The topic is described by the following set of keywords: {keywords}.
"""

DOCUMENT_TEMPLATE = """
In addition the topic is characterized by the following documents:
{documents}
"""

CONTEXT_TEMPLATE = """
Additional context for the task provided by the user: {context}
"""

SUMMARY_PROMPT = """
Summarize the following document: {document}
"""

DEFAULT_SYSTEM_PROMPT = """
You are a topic analyzer.
When the user asks you to name a topic, you respond with a name for the topic they describe.
When they ask you to describe the topic briefly, you respond with a couple of sentences explaining what the topic is about.
When they ask you to summarize a document, you respond with a brief summary.
"""


@dataclass
class AnalysisResults:
    """Container class for results of topic analysis.

    Attributes
    ----------
    topic_names: list[str]
        Generated topic names.
    topic_descriptions: list[str]
        Genreated topic descriptions.
    document_summaries: list[list[str]], default None
        Summaries of top 10 documents for each topic, when use_summaries is enabled.
    """

    topic_names: list[str]
    topic_descriptions: list[str]
    document_summaries: Optional[list[list[str]]] = None

    def to_dict(self) -> dict:
        """Returns the analysis result as a dictionary"""
        res = dict(
            topic_names=self.topic_names,
            topic_descriptions=self.topic_descriptions,
        )
        if self.document_summaries is not None:
            res["document_summaries"] = self.document_summaries
        return res

    def to_df(self):
        """Turns analysis result object into a dataframe"""
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "You need to pip install pandas to be able to use dataframes."
            )
        return pd.DataFrame(self.to_dict())


class Analyzer(ABC):
    system_prompt = DEFAULT_SYSTEM_PROMPT
    summary_prompt = SUMMARY_PROMPT
    namer_prompt = NAMER_PROMPT
    description_prompt = DESCRIPTION_PROMPT
    context = None
    use_summaries = False

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generates response to a given prompt."""
        pass

    def summarize_document(self, document: str) -> str:
        """Summarizes document so that analysis becomes easier."""
        prompt = self.summary_prompt.format(document=document)
        return self.generate_text(prompt)

    def describe_topic(
        self,
        keywords: list[str],
        documents: Optional[list] = None,
    ):
        """Gives abstract summarization of topic content."""
        _keys = ", ".join(keywords)
        prompt = self.description_prompt.format(keywords=_keys)
        if documents:
            prompt += self.template_documents(documents)
        if self.context:
            prompt += CONTEXT_TEMPLATE.format(context=self.context)
        return self.generate_text(prompt)

    def name_topic(
        self,
        keywords: list[str],
        documents: Optional[list] = None,
    ) -> str:
        """Names one topic based on top descriptive aspects."""
        _keys = ", ".join(keywords)
        prompt = self.namer_prompt.format(keywords=_keys)
        if documents:
            prompt += self.template_documents(documents)
        if self.context:
            prompt += CONTEXT_TEMPLATE.format(context=self.context)
        return self.generate_text(prompt)

    def name_topics(
        self,
        keywords: list[list[str]],
        documents: list[list[str]] = None,
    ) -> list[str]:
        """Names all topics based on top descriptive terms.

        Parameters
        ----------
        keywords: list[list[str]]
            Top K highest ranking terms on the topics.
        documents: list[list[str]], optional
            Top K relevant documents to each topic.

        Returns
        -------
        list[str]
            Topic names returned by the namer.
        """
        names = []
        if documents is not None:
            key_doc = list(zip(keywords, documents))
            for keys, docs in track(key_doc, description="Naming topics..."):
                names.append(self.name_topic(keys, documents=docs))
        else:
            for keys in track(keywords, description="Naming topics..."):
                names.append(self.name_topic(keys))
        return names

    def template_documents(self, documents: list[str]) -> str:
        doc_list = "\n".join([f" - {doc}" for doc in documents])
        return """
        In addition the topic is characterized by the following documents:
        {documents}
        """.format(
            documents=doc_list
        )

    def analyze_topics(
        self,
        keywords: list[list[str]],
        documents: Optional[list[list[str]]] = None,
        use_summaries: Optional[bool] = None,
    ) -> AnalysisResults:
        """Analyzes topic model with a language model.
        Generates topic names, descriptions and document summaries (optional).

        Parameters
        ----------
        keywords: list[list[str]]
            Keywords for each topic.
        documents: list[list[str]], optional
            Top documents for each topic.
        use_summaries: bool, optional
            Indicates whether the analyzer should summarize documents
            prior to analyzing the topic.

        Returns
        -------
        dict
            Dictionary containing `topic_names`, `topic_descriptions` and `document_summaries` if relevant.
        """
        console = Console()
        output = {"topic_names": [], "topic_descriptions": []}
        use_summaries = (
            use_summaries if use_summaries is not None else self.use_summaries
        )
        if documents is not None:
            if use_summaries:
                output["document_summaries"] = []
                for docs in track(
                    documents, description="Summarizing documents"
                ):
                    _sums = []
                    for doc in docs:
                        _sums.append(self.summarize_document(doc))
                    output["document_summaries"].append(_sums)
                console.log("Documents summarized.")
                # Updating parameter so summaries are used down-stream
                documents = output["document_summaries"]
            # Organizing into a list so we can iterate and know the length at the same time.
            key_doc_pairs = list(zip(keywords, documents))
            for keys, docs in track(
                key_doc_pairs, description="Generating topic names"
            ):
                output["topic_names"].append(
                    self.name_topic(keys, documents=docs)
                )
            console.log("Topic names generated.")
            for keys, docs in track(
                key_doc_pairs, description="Generating topic descriptions."
            ):
                output["topic_descriptions"].append(
                    self.describe_topic(keys, documents=docs)
                )
            console.log("Topic descriptions generated.")
        else:
            for keys in track(keywords, description="Naming"):
                output["topic_names"].append(
                    self.name_topic(keys, documents=None)
                )
            console.log("Topic names generated.")
            for keys in track(keywords, description="Describing topics."):
                output["topic_descriptions"].append(
                    self.describe_topic(keys, documents=None)
                )
            console.log("Topic descriptions generated.")
        return AnalysisResults(**output)
