import openai

from turftopic.namers.base import (DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT,
                                   TopicNamer)


class OpenAITopicNamer(TopicNamer):
    """Name topics with an OpenAI model.

    Parameters
    ----------
    model_name: str, default 'gpt-4o-mini'
        OpenAI model to use.
    prompt_template: str
        Prompt template to use when no negative terms are specified.
    system_prompt: str
        System prompt to use for the language model.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        prompt_template: str = DEFAULT_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self.client = openai.OpenAI()
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

    def name_topic(
        self,
        keywords: list[list[str]],
    ) -> str:
        prompt = self.prompt_template.format(keywords=", ".join(keywords))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )
        return response.choices[0].message.content
