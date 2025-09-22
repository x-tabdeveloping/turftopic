import re
from typing import Optional

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

from turftopic.analyzers.base import Analyzer

GLINER_DESC_PROMPT = "Summarize the following:"
GLINER_NAME_PROMPT = "What is the common topic of these documents?"


class UTCAnalyzer(Analyzer):
    """Analyze topic model with a universal token classification model.

    Parameters
    ----------
    model_name: str, default 'knowledgator/UTC-DeBERTa-base-v2'
        GliNER model to use for analyses.
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
        model_name: str = "knowledgator/UTC-DeBERTa-base-v2",
        context: Optional[str] = None,
        use_summaries: bool = False,
        system_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        namer_prompt: Optional[str] = GLINER_NAME_PROMPT,
        description_prompt: Optional[str] = GLINER_DESC_PROMPT,
        device: str = "cpu",
        threshold: float = 0.5,
        max_tokens: int = 256,
    ):
        self.threshold = threshold
        self.device = device
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
        )
        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first",
            device=self.device,
        )
        self.summary_prompt = summary_prompt or self.summary_prompt
        self.namer_prompt = namer_prompt or self.namer_prompt
        self.description_prompt = description_prompt or self.description_prompt
        self.use_summaries = use_summaries

    def summarize_document(self, document: str) -> str:
        prompt = self.summary_prompt
        prompt = re.sub(r"{[^}]*}*", "", prompt)
        return "\n".join(self.process(prompt, document))

    def describe_topic(
        self,
        keywords: list[str],
        documents: Optional[list] = None,
    ):
        prompt = self.description_prompt
        _keys = ", ".join(keywords)
        text = """ ## Top Keywords:
        {keywords}
        """.format(
            keywords=_keys
        )
        if documents:
            text += "## Top Documents:"
            for document in documents:
                text += f"\n - {document}"
        items = self.process(prompt, text)
        items = [item.lower().split() for item in items]
        items = [s for s in items if s]
        return "•" + "\n • ".join(items)

    def name_topic(
        self,
        keywords: list[str],
        documents: Optional[list] = None,
    ) -> str:
        prompt = self.namer_prompt
        _keys = ", ".join(keywords)
        text = """ ## Top Keywords:
        {keywords}
        """.format(
            keywords=_keys
        )
        if documents:
            text += "## Top Documents:"
            for document in documents:
                text += f"\n - {document}"
        spans = self.process(prompt, text)
        spans = [s.strip().lower() for s in spans if s.strip()]
        spans = set(spans)
        return " | ".join(spans)

    def generate_text(self, prompt: str) -> str:
        return " ".join(self.process(prompt="", text=prompt))

    def process(self, prompt: str, text: str) -> str:
        # Concatenate text and prompt for full input
        input_ = f"{prompt}\n{text}"
        model_input = next(
            self.nlp.preprocess(
                input_,
                tokenizer_params={"max_length": self.max_tokens},
                is_split_into_words=False,
                delimiter=" ",
                offset_mapping=None,
            )
        )
        with torch.no_grad():
            model_output = self.nlp._forward(model_input)
        results = self.nlp.postprocess(
            [model_output], aggregation_strategy="first"
        )
        spans = []
        prompt_length = len(prompt)  # Get prompt length
        for result in results:
            # check whether score is higher than treshold
            if result["score"] < self.threshold:
                continue
            # Adjust indices by subtracting prompt length
            start = result["start"] - prompt_length
            # If indexes belongs to the prompt - continue
            if start < 0:
                continue
            end = result["end"] - prompt_length
            # Extract span from original text using adjusted indices
            span = text[start:end]
            # Remove unnecessary white space
            span = " ".join(span.split())
            spans.append(span)
        return spans
