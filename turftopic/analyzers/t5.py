from typing import Optional

from transformers import pipeline

from turftopic.analyzers.base import Analyzer

T5_NAME_PROMPT = "What is the topic? Keywords: {keywords}"
T5_DESC_PROMPT = "Summarize in one sentence. Keywords: {keywords}"


class T5Analyzer(Analyzer):
    """Analyze topic model with a text-to-text model.

    Parameters
    ----------
    model_name: str, default 'google/flan-t5-small'
        Text-to-text model to use for analyses.
    use_summaries: bool, default False
        Indicates whether the language model should summarize documents before
        analyzing the topics.
    context: str, default None
        Additional context provided to the analyzer for analysis.
        e.g. "Analyze topics from blog posts related to morality and religion"
    system_prompt: str, default None
        Ignored, exists for compatibility
    summary_prompt: str, default None
        Prompt to use for abstractive summarization.
    namer_prompt: str, default None
        Prompt template for naming topics.
    description_prompt: str, default None
        Prompt template for generating topic descriptions.
    device: str, default "cpu"
        ID of the device to run the language model on.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        context: Optional[str] = None,
        use_summaries: bool = False,
        system_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        namer_prompt: Optional[str] = T5_NAME_PROMPT,
        description_prompt: Optional[str] = T5_DESC_PROMPT,
        device: str = "cpu",
    ):
        self.device = device
        self.model_name = model_name
        self.pipeline = pipeline(
            task="text2text-generation",
            model=self.model_name,
            device=self.device,
        )
        self.summary_prompt = summary_prompt or self.summary_prompt
        self.namer_prompt = namer_prompt or self.namer_prompt
        self.description_prompt = description_prompt or self.description_prompt
        self.use_summaries = use_summaries

    def generate_text(self, prompt: str) -> str:
        return self.pipeline(prompt)
