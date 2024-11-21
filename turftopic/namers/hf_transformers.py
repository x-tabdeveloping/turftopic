from transformers import pipeline

from turftopic.namers.base import (DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT,
                                   TopicNamer)


class LLMTopicNamer(TopicNamer):
    """Name topics with an instruction-finetuned LLM, e.g. Zephyr-7b-beta

    Parameters
    ----------
    model_name: str, default 'HuggingFaceTB/SmolLM2-1.7B-Instruct'
        Model to load from :hugs: Hub.
    prompt_template: str
        Prompt template to use when no negative terms are specified.
    system_prompt: str
        System prompt to use for the language model.
    device: str, default 'cpu'
        Device to run the model on.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        prompt_template: str = DEFAULT_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.device = device
        self.pipe = pipeline(
            "text-generation", self.model_name, device=self.device
        )

    def name_topic(
        self,
        keywords: list[list[str]],
    ) -> str:
        prompt = self.prompt_template.format(keywords=", ".join(keywords))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.pipe(messages, max_new_tokens=24)[0]["generated_text"][
            -1
        ]
        label = response["content"]
        return label
