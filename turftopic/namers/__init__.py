from turftopic.error import NotInstalled
from turftopic.namers.hf_transformers import LLMTopicNamer

try:
    from turftopic.namers.openai import OpenAITopicNamer
except ModuleNotFoundError:
    OpenAITopicNamer = NotInstalled("OpenAITopicNamer", "openai")


__all__ = ["OpenAITopicNamer", "LLMTopicNamer"]
