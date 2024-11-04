from abc import ABC, abstractmethod

from rich.progress import track

DEFAULT_PROMPT = """
You will be tasked with naming a topic.
Based on the keywords, create a short label that best summarizes the topics.
Only respond with a short, human readable topic name and nothing else.

The topic is described by the following set of keywords: {keywords}.
"""


DEFAULT_SYSTEM_PROMPT = """
You are a topic namer. When the user gives you a set of keywords, you respond with a name for the topic they describe.
You only repond briefly with the name of the topic, and nothing else.
"""


class TopicNamer(ABC):
    @abstractmethod
    def name_topic(
        self,
        keywords: list[str],
    ) -> str:
        """Names one topics based on top descriptive terms.

        Parameters
        ----------
        keywords: list[str]
            Top K highest ranking terms on the topic.

        Returns
        -------
        str
            Topic name returned by the namer.
        """
        pass

    def name_topics(
        self,
        keywords: list[list[str]],
    ) -> list[str]:
        """Names all topics based on top descriptive terms.

        Parameters
        ----------
        keywords: list[list[str]]
            Top K highest ranking terms on the topics.

        Returns
        -------
        list[str]
            Topic names returned by the namer.
        """
        names = []
        for keys in track(keywords, description="Naming topics..."):
            names.append(self.name_topic(keys))
        return names
