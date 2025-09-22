from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from turftopic.analyzers.base import Analyzer


def remove_thinking_trace(response: str) -> str:
    """Removes think tag and all tokens in-between"""
    if "<think>" not in response:
        return response
    before, after = response.split("<think>")
    between, after = after.split("</think>")
    return before + after


class LLMAnalyzer(Analyzer):
    """Analyze topic model with an open LLM from HF Hub.

    Parameters
    ----------
    model_name: str, default 'HuggingFaceTB/SmolLM3-3B'
        Open LLM to use from HF Hub.
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
    max_new_tokens: int, default 32768
        Max new tokens to generate when analyzing.
    enable_thinking: bool, default False
        Indicates whether thinking mode should be enabled.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM3-3B",
        context: Optional[str] = None,
        use_summaries: bool = False,
        system_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        namer_prompt: Optional[str] = None,
        description_prompt: Optional[str] = None,
        max_new_tokens: int = 32768,
        device: str = "cpu",
        enable_thinking: bool = False,
    ):
        self.device = device
        self.model_name = model_name
        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        ).to(self.device)
        self.summary_prompt = summary_prompt or self.summary_prompt
        self.namer_prompt = namer_prompt or self.namer_prompt
        self.description_prompt = description_prompt or self.description_prompt
        self.use_summaries = use_summaries
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

    def generate_text(self, prompt: str) -> str:
        thinking = "/think" if self.enable_thinking else "/no_think"
        system_prompt = self.system_prompt + thinking
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.model.device
        )
        # Generate the output
        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=32768
        )
        # Get and decode the output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        result = remove_thinking_trace(result)
        return result
