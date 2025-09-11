import warnings

from turftopic.analyzers import LLMAnalyzer, OpenAIAnalyzer

warnings.warn(
    """
The turftopic.namers is deprecated in favor of turftopic.analyzers and will be fully phased out in 1.1.0!
For now, we provide aliases to the analyzers module, but we encourage you to use the analyzers module
directly. """,
    DeprecationWarning,
)

LLMTopicNamer = LLMAnalyzer
OpenAITopicNamer = OpenAIAnalyzer

__all__ = ["OpenAITopicNamer", "LLMTopicNamer"]
