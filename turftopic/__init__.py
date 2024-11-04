from turftopic.base import ContextualModel
from turftopic.error import NotInstalled
from turftopic.models.cluster import ClusteringTopicModel
from turftopic.models.decomp import SemanticSignalSeparation
from turftopic.models.fastopic import FASTopic
from turftopic.models.gmm import GMM
from turftopic.models.keynmf import KeyNMF
from turftopic.serialization import load_model

try:
    from turftopic.models.ctm import AutoEncodingTopicModel
except ModuleNotFoundError:
    AutoEncodingTopicModel = NotInstalled("AutoEncodingTopicModel", "pyro-ppl")


__all__ = [
    "ClusteringTopicModel",
    "SemanticSignalSeparation",
    "GMM",
    "KeyNMF",
    "AutoEncodingTopicModel",
    "ContextualModel",
    "FASTopic",
    "load_model",
]
