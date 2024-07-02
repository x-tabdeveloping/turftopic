from typing import Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from turftopic.namers.base import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_POSITIVE_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    TopicNamer,
)


class Text2TextTopicNamer(TopicNamer):
    """Name topics with a Text2Text model (e.g. Google's T5).

    Parameters
    ----------
    model_name: str, default 'google/flan-t5-large'
        Model to load from :hugs: Hub.
    prompt_template: str
        Prompt template to use when no negative terms are specified.
    axis_prompt_template: str
        Prompt template to use when negative terms are also specified.
    device: str, default 'cpu'
        Device to run the model on.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        prompt_template: str = DEFAULT_POSITIVE_PROMPT,
        axis_prompt_template: str = DEFAULT_NEGATIVE_PROMPT,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.axis_prompt_template = axis_prompt_template
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            self.device
        )

    def name_topic(
        self,
        positive: list[list[str]],
        negative: Optional[list[list[str]]] = None,
    ) -> str:
        if negative is not None:
            prompt = self.axis_prompt_template.format(
                positive=", ".join(positive), negative=", ".join(negative)
            )
        else:
            prompt = self.prompt_template.format(positive=", ".join(positive))
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=24)
        label = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return label


class ChatTopicNamer(TopicNamer):
    """Name topics with a Chat model, e.g. Zephyr-7b-beta

    Parameters
    ----------
    model_name: str, default 'HuggingFaceH4/zephyr-7b-beta'
        Model to load from :hugs: Hub.
    prompt_template: str
        Prompt template to use when no negative terms are specified.
    axis_prompt_template: str
        Prompt template to use when negative terms are also specified.
    system_prompt: str
        System prompt to use for the language model.
    device: str, default 'cpu'
        Device to run the model on.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceH4/zephyr-7b-beta",
        prompt_template: str = DEFAULT_POSITIVE_PROMPT,
        axis_prompt_template: str = DEFAULT_NEGATIVE_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.axis_prompt_template = axis_prompt_template
        self.system_prompt = system_prompt
        self.device = device
        self.pipe = pipeline(
            "text-generation", self.model_name, device=self.device
        )

    def name_topic(
        self,
        positive: list[list[str]],
        negative: Optional[list[list[str]]] = None,
    ) -> str:
        if negative is not None:
            prompt = self.axis_prompt_template.format(
                positive=", ".join(positive), negative=", ".join(negative)
            )
        else:
            prompt = self.prompt_template.format(positive=", ".join(positive))
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.pipe(messages, max_new_tokens=24)[0]["generated_text"][
            -1
        ]
        label = response["content"]
        return label
