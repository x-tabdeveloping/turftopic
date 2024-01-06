from turftopic.error import NotInstalled

try:
    from turftopic.encoders.cohere import CohereEmbeddings
except ModuleNotFoundError:
    CohereEmbeddings = NotInstalled("CohereEmbeddings", "cohere")

try:
    from turftopic.encoders.openai import OpenAIEmbeddings
except ModuleNotFoundError:
    OpenAIEmbeddings = NotInstalled("OpenAIEmbeddings", "openai")

try:
    from turftopic.encoders.voyage import VoyageEmbeddings
except ModuleNotFoundError:
    VoyageEmbeddings = NotInstalled("VoyageEmbeddings", "voyageai")

__all__ = ["CohereEmbeddings", "OpenAIEmbeddings", "VoyageEmbeddings"]
