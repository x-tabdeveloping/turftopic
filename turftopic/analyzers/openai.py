from typing import Optional

import openai

from turftopic.analyzers.base import Analyzer


class OpenAIAnalyzer(Analyzer):
    """Analyze topic model with an OpenAI LLM.

    Parameters
    ----------
    model_name: str, default 'gpt-5-nano'
        OpenAI model to use.
    use_summaries: bool, default False
        Indicates whether the language model should summarize documents before
        analyzing the topics.
    context: str, default None
        Additional context provided to the analyzer for analysis.
        e.g. "Analyze topics from blog posts related to morality and religion"
    system_prompt: str, default None
        System prompt to use for the language model.
    summary_prompt: str, default None
        Prompt to use for abstractive summarization.
    namer_prompt: str, default None
        Prompt template for naming topics.
    description_prompt: str, default None
        Prompt template for generating topic descriptions.
    """

    def __init__(
        self,
        model_name: str = "gpt-5-nano",
        context: Optional[str] = None,
        use_summaries: bool = False,
        system_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        namer_prompt: Optional[str] = None,
        description_prompt: Optional[str] = None,
    ):
        self.client = openai.OpenAI()
        self.model_name = model_name
        self.system_prompt = system_prompt or self.system_prompt
        self.summary_prompt = summary_prompt or self.summary_prompt
        self.namer_prompt = namer_prompt or self.namer_prompt
        self.description_prompt = description_prompt or self.description_prompt
        self.use_summaries = use_summaries

    def generate_text(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )
        return response.choices[0].message.content
