from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from turftopic.base import ContextualModel
from turftopic.utils import export_table


def _build_tree(h: TopicHierarchy, tree: Tree, top_k: int, level=0):
    names = h._topic_desc(top_k, level=level)
    if h.subtopics is None:
        for name in names:
            tree.add(name)
        return
    for name, sub in zip(names, h.subtopics):
        branch = tree.add(name)
        _build_tree(sub, branch, top_k, level=level + 1)


COLOR_PER_LEVEL = [
    "bright_blue",
    "bright_magenta",
    "bright_cyan",
    "bright_green",
    "bright_red",
    "bright_yellow",
    "cyan",
    "magenta",
    "blue",
    "white",
]


@dataclass
class TopicHierarchy:
    model: ContextualModel
    components_: np.ndarray
    document_topic_matrix: np.ndarray
    subtopics: Optional[list[TopicHierarchy]] = None
    path: tuple[int] = ()
    desc: str = "Root"

    def __getitem__(self, index: int):
        return self.subtopics[index]

    @property
    def level(self) -> int:
        return len(self.path)

    @level.setter
    def level(self, value):
        self._level = value

    def get_topics(
        self, top_k: int = 10
    ) -> list[tuple[Any, list[tuple[str, float]]]]:
        """Returns high-level topic representations in form of the top K words
        in each topic.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.

        Returns
        -------
        list[tuple]
            List of topics. Each topic is a tuple of
            topic ID and the top k words.
            Top k words are a list of (word, word_importance) pairs.
        """
        n_topics = self.components_.shape[0]
        try:
            classes = self.model.classes_
        except AttributeError:
            classes = list(range(n_topics))
        highest = np.argpartition(-self.components_, top_k)[:, :top_k]
        vocab = self.model.get_vocab()
        top = []
        score = []
        for component, high in zip(self.components_, highest):
            importance = component[high]
            high = high[np.argsort(-importance)]
            score.append(component[high])
            top.append(vocab[high])
        topics = []
        for topic, words, scores in zip(classes, top, score):
            topic_data = (topic, list(zip(words, scores)))
            topics.append(topic_data)
        return topics

    @property
    def topic_names(self) -> list[str]:
        """Names of the topics based on the highest scoring 4 terms."""
        topic_desc = self.get_topics(top_k=4)
        names = []
        for topic_id, terms in topic_desc:
            concat_words = "_".join([word for word, importance in terms])
            names.append(f"{topic_id}_{concat_words}")
        return names

    def _topic_desc(self, top_k: int = 10, level=0) -> str:
        topic_desc = self.get_topics(top_k=top_k)
        names = []
        color = COLOR_PER_LEVEL[min(level, len(COLOR_PER_LEVEL) - 1)]
        for topic_id, terms in topic_desc:
            concat_words = ", ".join([word for word, importance in terms])
            topic_id = ".".join(str(elem) for elem in [*self.path, topic_id])
            names.append(
                f"[{color} bold]{topic_id}[/]: [italic]{concat_words}[/]"
            )
        return names

    def __str__(self):
        tree = Tree(f"[bold]{self.desc}[/]")
        _build_tree(self, tree, top_k=10, level=self.level)
        console = Console()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def __repr__(self):
        return str(self)

    def _topics_table(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: bool = False,
    ) -> list[list[str]]:
        columns = ["Topic ID", "Highest Ranking"]
        if show_negative:
            columns.append("Lowest Ranking")
        rows = []
        try:
            classes = self.model.classes_
        except AttributeError:
            classes = list(range(self.components_.shape[0]))
        vocab = self.model.get_vocab()
        for topic_id, component in zip(classes, self.components_):
            highest = np.argpartition(-component, top_k)[:top_k]
            highest = highest[np.argsort(-component[highest])]
            lowest = np.argpartition(component, top_k)[:top_k]
            lowest = lowest[np.argsort(component[lowest])]
            if show_scores:
                concat_positive = ", ".join(
                    [
                        f"{word}({importance:.2f})"
                        for word, importance in zip(
                            vocab[highest], component[highest]
                        )
                    ]
                )
                concat_negative = ", ".join(
                    [
                        f"{word}({importance:.2f})"
                        for word, importance in zip(
                            vocab[lowest], component[lowest]
                        )
                    ]
                )
            else:
                concat_positive = ", ".join([word for word in vocab[highest]])
                concat_negative = ", ".join([word for word in vocab[lowest]])
            row = [f"{topic_id}", f"{concat_positive}"]
            if show_negative:
                row.append(concat_negative)
            rows.append(row)
        return [columns, *rows]

    def print_topics(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: bool = False,
    ):
        """Pretty prints topics at the current level of the hierarchy.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        show_negative: bool, default False
            Indicates whether the most negative terms should also be displayed.
        """
        columns, *rows = self._topics_table(top_k, show_scores, show_negative)
        table = Table(show_lines=True)
        table.add_column("Topic ID", style="blue", justify="right")
        table.add_column(
            "Highest Ranking",
            justify="left",
            style="magenta",
            max_width=100,
        )
        if show_negative:
            table.add_column(
                "Lowest Ranking",
                justify="left",
                style="red",
                max_width=100,
            )
        for row in rows:
            table.add_row(*row)
        console = Console()
        console.print(table)

    def export_topics(
        self,
        top_k: int = 10,
        show_scores: bool = False,
        show_negative: bool = False,
        format: str = "csv",
    ) -> str:
        """Exports top K words from topics in a table in a given format.
        Returns table as a pure string.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return for each topic.
        show_scores: bool, default False
            Indicates whether to show importance scores for each word.
        show_negative: bool, default False
            Indicates whether the most negative terms should also be displayed.
        format: 'csv', 'latex' or 'markdown'
            Specifies which format should be used.
            'csv', 'latex' and 'markdown' are supported.
        """
        table = self._topics_table(
            top_k, show_scores, show_negative=show_negative
        )
        return export_table(table, format=format)

    def add_level(self, n_subtopics: int, **kwargs):
        """Adds level to hierarchy based on the topics on this level.

        Parameters
        ----------
        n_subtopics: int
            Number of subtopics to add for each topic on this level.
        """
        self.subtopics = self.model.calculate_subtopics(
            hierarchy=self, n_subtopics=n_subtopics, **kwargs
        )
        for i, (desc, subtopic) in enumerate(
            zip(self._topic_desc(level=self.level), self.subtopics)
        ):
            subtopic.path = (*self.path, i)
            subtopic.desc = desc
        return self
