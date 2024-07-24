from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.tree import Tree

from turftopic.base import ContextualModel

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
class TopicNode:
    """Node for a topic in a topic hierarchy.

    Parameters
    ----------
    model: ContextualModel
        Underlying topic model, which the hierarchy is based on.
    path: tuple[int], default ()
        Path that leads to this node from the root of the tree.
    word_importance: ndarray of shape (n_vocab), default None
        Importance of each word in the vocabulary for given topic.
    document_topic_vector: ndarray of shape (n_documents), default None
        Importance of the topic in all documents in the corpus.
    children: list[TopicNode], default None
        List of subtopics within this topic.
    """

    model: ContextualModel
    path: tuple[int] = ()
    word_importance: Optional[np.ndarray] = None
    document_topic_vector: Optional[np.ndarray] = None
    children: Optional[list[TopicNode]] = None

    @classmethod
    def create_root(
        cls,
        model: ContextualModel,
        components: np.ndarray,
        document_topic_matrix: np.ndarray,
    ) -> TopicNode:
        """Creates root node from a topic models' components and topic importances in documents."""
        children = []
        n_components = components.shape[0]
        for i, comp, doc_top in zip(
            range(n_components), components, document_topic_matrix.T
        ):
            children.append(
                cls(
                    model,
                    path=(i,),
                    word_importance=comp,
                    document_topic_vector=doc_top,
                    children=None,
                )
            )
        return TopicNode(
            model,
            path=(),
            word_importance=None,
            document_topic_vector=None,
            children=children,
        )

    @property
    def level(self) -> int:
        """Indicates how deep down the hierarchy the topic is."""
        return len(self.path)

    def get_words(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Returns top words and words importances for the topic.

        Parameters
        ----------
        top_k: int, default 10
            Number of top words to return.

        Returns
        -------
        list[tuple[str, float]]
            List of word, importance pairs.
        """
        if (self.word_importance is None) or (
            self.document_topic_vector
        ) is None:
            return []
        idx = np.argpartition(-self.word_importance, top_k)[:top_k]
        order = np.argsort(self.word_importance[idx])
        idx = idx[order]
        imp = self.word_importance[idx]
        words = self.model.get_vocab()[idx]
        return list(zip(words, imp))

    @property
    def description(self) -> str:
        """Returns a high level description of the topic with its path in the tree
        and top words."""
        if not len(self.path):
            path = "Root"
        else:
            path = ".".join([str(idx) for idx in self.path])
        words = []
        for word, imp in self.get_words(top_k=10):
            words.append(word)
        concat_words = ", ".join(words)
        color = COLOR_PER_LEVEL[min(self.level, len(COLOR_PER_LEVEL) - 1)]
        stylized = f"[{color} bold]{path}[/]: [italic]{concat_words}[/]"
        console = Console()
        with console.capture() as capture:
            console.print(stylized, end="")
        return capture.get()

    def _build_tree(self, tree: Tree = None, top_k: int = 10) -> Tree:
        if tree is None:
            tree = Tree(self.description)
        else:
            tree = tree.add(self.description)
        if self.children is not None:
            for child in self.children:
                child._build_tree(tree)
        return tree

    def __str__(self):
        tree = self._build_tree(top_k=10)
        console = Console()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def __repr__(self):
        return str(self)

    def clear(self):
        """Deletes children of the given node."""
        self.children = None
        return self

    def __getitem__(self, index: int):
        if self.children is None:
            raise IndexError("Current node is a leaf and has not children.")
        return self.children[index]

    def divide(self, n_subtopics: int, **kwargs):
        """Divides current node into smaller subtopics.
        Only works when the underlying model is a divisive hierarchical model.

        Parameters
        ----------
        n_subtopics: int
            Number of topics to divide the topic into.
        """
        try:
            self.children = self.model.divide_topic(
                node=self, n_subtopics=n_subtopics, **kwargs
            )
        except AttributeError as e:
            raise AttributeError(
                "Looks like your model is not a divisive hierarchical model."
            ) from e
        return self

    def divide_children(self, n_subtopics: int, **kwargs):
        """Divides all children of the current node to smaller topics.
        Only works when the underlying model is a divisive hierarchical model.

        Parameters
        ----------
        n_subtopics: int
            Number of topics to divide the topics into.
        """
        if self.children is None:
            raise ValueError(
                "Current Node is a leaf, children can't be subdivided."
            )
        for child in self.children:
            child.divide(n_subtopics, **kwargs)
        return self
