from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


def _tree_plot(hierarchy: TopicNode):
    """Plots hierarchy with Plotly as a Tree"""
    try:
        import igraph as ig
        import plotly.express as px
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "You will need to install plotly and igraph to use hierarchical plotting functionality."
        ) from e

    def word_table(node):
        entries = []
        words = node.get_words(top_k=10)
        for word, imp in words:
            entries.append(f"<b>{word}</b>: <i>{imp:.2f}</i>")
        return " <br> ".join(entries)

    def traverse(h, nodes, edges, tables, parent=None):
        nodes.append(h)
        tables.append(word_table(h))
        if parent is not None:
            edges.append([parent._path_str(), h._path_str()])
        if h.children is not None:
            for child in h.children:
                traverse(child, nodes, edges, tables, parent=h)

    nodes = []
    edges = []
    tables = []
    for child in hierarchy.children:
        traverse(child, nodes, edges, tables)
    node_names = [node._simple_desc for node in nodes]
    node_ids = [node._path_str() for node in nodes]
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edges_idx = [
        [node_to_idx[start], node_to_idx[end]] for start, end in edges
    ]
    graph = ig.Graph(len(nodes), edges=edges_idx, directed=True)
    layout = graph.layout("rt")
    layout.rotate(-90)
    x, y = np.array(layout.coords).T
    xmin, xmax = np.min(x), np.max(x)
    xpad = (xmax - xmin) * 0.35
    # Mapping root nodes to colors
    color_scheme = px.colors.qualitative.Dark24
    root_nodes = [str(node.path[0]) for node in nodes]
    color_map = {
        root_node: color_scheme[i % len(color_scheme)]
        for i, root_node in enumerate(np.unique(root_nodes))
    }
    fig = px.scatter(x=x, y=y, text=node_names, template="plotly_white")
    fig = fig.update_traces(
        customdata=np.array([[table] for table in tables]),
        hovertemplate="<b>%{text}</b> <br> <br> %{customdata[0]}",
        marker=dict(color=[color_map[root_node] for root_node in root_nodes]),
    )
    fig = fig.update_traces(marker=dict(size=20))
    fig = fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family="Roboto Mono"),
    )
    fig = fig.update_yaxes(
        showgrid=False,
        visible=False,
        zeroline=False,
    )
    fig = fig.update_xaxes(
        showgrid=False,
        visible=False,
        zeroline=False,
        range=(xmin - xpad, xmax + xpad),
    )
    for start, end in edges_idx:
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x[start],
            y0=y[start],
            x1=x[end],
            y1=y[end],
            opacity=0.2,
        )
    return fig


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

    def _path_str(self):
        return ".".join([str(level_id) for level_id in self.path])

    @property
    def classes_(self):
        if self.children is None:
            raise AttributeError("TopicNode doesn't have children.")
        return np.array([child.path[-1] for child in self.children])

    @property
    def components_(self):
        if self.children is None:
            raise AttributeError("TopicNode doesn't have children.")
        return np.stack([child.word_importance for child in self.children])

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
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = np.arange(n_components)
        for topic_id, comp, doc_top in zip(
            classes, components, document_topic_matrix.T
        ):
            children.append(
                cls(
                    model,
                    path=(topic_id,),
                    word_importance=comp,
                    document_topic_vector=doc_top,
                    children=None,
                )
            )
        return cls(
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
        if self.word_importance is None:
            return []
        vocab = self.model.get_vocab()
        most_important = np.argsort(-self.word_importance)[:top_k]
        words = vocab[most_important]
        imp = self.word_importance[most_important]
        return list(zip(words, imp))

    @property
    def description(self) -> str:
        """Returns a high level description of the topic with its path in the tree
        and top words."""
        if not len(self.path):
            path = "Root"
        else:
            path = str(
                self.path[-1]
            )  # ".".join([str(idx) for idx in self.path])
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

    @property
    def _simple_desc(self) -> str:
        if not len(self.path):
            path = "Root"
        else:
            path = str(
                self.path[-1]
            )  # ".".join([str(idx) for idx in self.path])
        words = []
        for word, imp in self.get_words(top_k=5):
            words.append(word)
        concat_words = ", ".join(words)
        return f"{path}: {concat_words}"

    def _build_tree(
        self,
        tree: Tree = None,
        top_k: int = 10,
        max_depth: Optional[int] = None,
    ) -> Tree:
        if tree is None:
            tree = Tree(self.description)
        else:
            tree = tree.add(self.description)
        out_of_depth = (max_depth is not None) and (self.level >= max_depth)
        if out_of_depth:
            if self.children is not None:
                tree.add("...")
            return tree
        if self.children is not None:
            for child in self.children:
                child._build_tree(tree, max_depth=max_depth)
        return tree

    def print_tree(
        self,
        top_k: int = 10,
        max_depth: Optional[int] = None,
    ) -> None:
        """Print hierarchy in tree form.

        Parameters
        ----------
        top_k: int, default 10
            Number of words to print for each topic.
        max_depth: int, default None
            Maximum depth at which topics should be printed in the hierarchy.
            If None, the entire hierarchy is printed.
        """
        tree = self._build_tree(top_k=top_k, max_depth=max_depth)
        console = Console()
        console.print(tree)

    def __str__(self):
        tree = self._build_tree(top_k=10, max_depth=3)
        console = Console()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def __repr__(self):
        return str(self)

    def __getitem__(self, id_or_path: int):
        if self.children is None:
            raise IndexError(
                "Current node is a leaf and does not have children."
            )
        mapping = {
            topic_class: i_topic
            for i_topic, topic_class in enumerate(self.classes_)
        }
        return self.children[mapping[id_or_path]]

    def __iter__(self):
        return iter(self.children)

    def plot_tree(self):
        """Plots hierarchy as an interactive tree in Plotly."""
        return _tree_plot(self)

    def _append_path(self, path_prefix: int):
        self.path = (path_prefix, *self.path)
        if self.children is not None:
            for child in self.children:
                child._append_path(path_prefix)

    def copy(self, deep: bool = True) -> TopicNode:
        """Creates a copy of the given node.

        Parameters
        ----------
        deep: bool, default True
            Indicates whether the copy should be deep or shallow.
            Deep copies are done recursively, while shallow copies only
            contain references to the original children.

        Returns
        -------
        Copy of original hierarchy.
        """
        if (self.children is None) or (not deep):
            return type(self)(
                model=self.model,
                path=self.path,
                children=self.children,
                word_importance=self.word_importance,
                document_topic_vector=self.document_topic_vector,
            )
        else:
            children = [child.copy(deep=True) for child in self.children]
            return type(self)(
                model=self.model,
                path=self.path,
                children=children,
                word_importance=self.word_importance,
                document_topic_vector=self.document_topic_vector,
            )

    def cut(self, max_depth: int) -> TopicNode:
        """Cuts hierarchy at a given depth, returns copy of the hierarchy with levels beyond max_depth removed.

        Parameters
        ----------
        max_depth: int
            Maximum level of nodes to keep.

        Returns
        -------
        TopicNode
            Hierarchy cut at the given level.
            Contains a deep copy of the original nodes.
        """
        if (self.level >= max_depth) or (not self.children):
            return type(self)(
                model=self.model,
                path=self.path,
                children=None,
                word_importance=self.word_importance,
                document_topic_vector=self.document_topic_vector,
            )
        else:
            children = [child.cut(max_depth) for child in self.children]
            return type(self)(
                model=self.model,
                path=self.path,
                children=children,
                word_importance=self.word_importance,
                document_topic_vector=self.document_topic_vector,
            )

    def collect_leaves(self) -> list[TopicNode]:
        def _collect_leaves(node: TopicNode, leaves: list[TopicNode]):
            if not node.children:
                leaves.append(node.copy(deep=False))
            else:
                for child in node.children:
                    _collect_leaves(child, leaves)

        leaves = []
        _collect_leaves(self, leaves)
        return leaves

    def flatten(self) -> TopicNode:
        """Returns new hierarchy with only the leaves of the tree.

        Returns
        -------
        TopicNode
            Root node containing all leaves in a hierarchy.
            Copies of the original nodes.
        """
        leaves = self.collect_leaves()
        ids = [leaf.path[-1] for leaf in leaves]
        # If the IDs are not unique, we label them from 0 to N
        if len(set(ids)) != len(ids):
            current = 0
            new_ids = []
            for node_id in ids:
                if node_id != -1:
                    new_ids.append(current)
                    current += 1
                else:
                    new_ids.append(-1)
            ids = new_ids
        for leaf_id, leaf in zip(ids, leaves):
            leaf.path = (*self.path, leaf_id)
        return type(self)(
            model=self.model,
            path=self.path,
            word_importance=self.word_importance,
            document_topic_vector=self.document_topic_vector,
            children=leaves,
        )


@dataclass
class DivisibleTopicNode(TopicNode):
    """Node for a topic in a topic hierarchy that can be subdivided."""

    def clear(self):
        """Deletes children of the given node."""
        self.children = None
        return self

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

    def __str__(self):
        tree = self._build_tree(top_k=10, max_depth=3)
        console = Console()
        with console.capture() as capture:
            console.print(tree)
        return capture.get()

    def __repr__(self):
        return str(self)
