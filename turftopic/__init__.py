from turftopic.error import NotInstalled
from turftopic.models.cluster import ClusteringTopicModel
from turftopic.models.decomp import ComponentTopicModel
from turftopic.models.gmm import MixtureTopicModel
from turftopic.models.keynmf import KeyNMF

try:
    from turftopic.models.ctm import AutoEncodingTopicModel
except ModuleNotFoundError:
    AutoEncodingTopicModel = NotInstalled("AutoEncodingTopicModel", "pyro-ppl")

__all__ = [
    "ClusteringTopicModel",
    "ComponentTopicModel",
    "MixtureTopicModel",
    "KeyNMF",
    "AutoEncodingTopicModel",
]
