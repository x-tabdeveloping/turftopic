from abc import ABC, abstractmethod
from typing import Optional

DEFAULT_POSITIVE_PROMPT = """
You will be tasked with naming a topic.
The topic is described by the following set of keywords: {positive}.

Based on the keywords, create a short label (3 words maximum) that best summarizes the topics.
Only respond with the topic name and nothing else.
"""

DEFAULT_NEGATIVE_PROMPT = """
You will be tasked with naming a topic.
The topic is described with most relevant positive and negative terms.
Make sure to consider the negative terms as well when naming the topic.
An example of a topic name like this would be: Oriental vs. European Cuisine

Positive terms: {positive}

Negative terms: {negative}

Based on the keywords, create a short label (5 words maximum) that best summarizes the topics.
Only respond with the topic name and nothing else.
"""

DEFAULT_SYSTEM_PROMPT = """
You are a topic namer. When the user gives you a set of keywords, you respond with a name for the topic they describe.
When negative terms are also specified, you take them into account.
You only repond briefly with the name of the topic, and don't ramble, nor reason about your choice.
"""


class TopicNamer(ABC):
    @abstractmethod
    def name_topic(
        self,
        positive: list[str],
        negative: Optional[list[str]] = None,
    ) -> str:
        """Names one topics based on top descriptive terms.

        Parameters
        ----------
        positive: list[str]
            Top K highest ranking terms on the topic.
        negative: list[str], default None
            Top K lowest ranking terms on the topic.
            (this is only relevant in the context of $S^3$)

        Returns
        -------
        str
            Topic name returned by the namer.
        """
        pass

    def name_topics(
        self,
        positive: list[list[str]],
        negative: Optional[list[list[str]]] = None,
    ) -> list[str]:
        """Names all topics based on top descriptive terms.

        Parameters
        ----------
        positive: list[list[str]]
            Top K highest ranking terms on the topics.
        negative: list[list[str]], default None
            Top K lowest ranking terms on the topics
            (this is only relevant in the context of $S^3$)

        Returns
        -------
        list[str]
            Topic names returned by the namer.
        """
        if negative is not None:
            return [
                self.name_topic(pos, neg)
                for pos, neg in zip(positive, negative)
            ]
        return [self.name_topics(pos) for pos in positive]
