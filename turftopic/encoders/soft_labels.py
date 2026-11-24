"""Language-model teacher for soft-label distillation (DSL)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from turftopic.encoders.base import ExternalEncoder

DEFAULT_INSTRUCTION = (
    "Generate a label that captures the overall theme of the document below."
)
DEFAULT_TEMPLATE = "{instruction}\n\nDocument: {document}\n\nTopic:"


@dataclass
class VocabularyMap:
    """How vocabulary words map onto LM tokens."""

    token_ids: np.ndarray  # (n_vocab,) id of the first token of " word"
    single_token: np.ndarray  # (n_vocab,) bool, the word is exactly one token


def map_vocabulary(tokenizer, vocabulary: Sequence[str]) -> VocabularyMap:
    """Find the token that starts each vocabulary word when it follows a space.

    Words made of one token are scored exactly. Words made of several tokens are
    scored by their first token, which is only an approximation, so `single_token`
    marks the words whose score is exact. `soft_labels` leaves the others out.
    """
    pieces = [
        tokenizer.encode(f" {word}", add_special_tokens=False)
        for word in vocabulary
    ]
    if any(len(p) == 0 for p in pieces):
        raise ValueError("A vocabulary word produced no tokens.")
    # SentencePiece tokenizers that emit a stand-alone space token before every word:
    # the distinguishing token is then the second one.
    first = {p[0] for p in pieces}
    offset = (
        1
        if (
            len(first) == 1
            and len(pieces) > 1
            and all(len(p) > 1 for p in pieces)
        )
        else 0
    )
    token_ids = np.array([p[offset] for p in pieces], dtype=np.int64)
    single = np.array(
        [
            len(p) == offset + 1 and word.isalpha()
            for p, word in zip(pieces, vocabulary)
        ],
        dtype=bool,
    )
    return VocabularyMap(token_ids=token_ids, single_token=single)


def build_prompts(
    documents: Iterable[str],
    tokenizer,
    instruction: str = DEFAULT_INSTRUCTION,
    template: str = DEFAULT_TEMPLATE,
    max_length: Optional[int] = None,
) -> list[str]:
    """Render one prompt per document; shorten the document if the prompt is too long.

    Truncating the tokenized prompt would cut off its end ("Topic:"), which is the
    position we read from, so the document text is shortened instead.
    """
    prompts = []
    for document in documents:
        prompt = template.format(
            instruction=instruction, document=document
        ).rstrip()
        if max_length is not None:
            n_tokens = len(tokenizer.encode(prompt))
            if n_tokens > max_length:
                overhead = len(
                    tokenizer.encode(
                        template.format(
                            instruction=instruction, document=""
                        ).rstrip()
                    )
                )
                budget = max(max_length - overhead - 2, 1)
                doc_ids = tokenizer.encode(document, add_special_tokens=False)[
                    :budget
                ]
                shortened = tokenizer.decode(doc_ids, skip_special_tokens=True)
                prompt = template.format(
                    instruction=instruction, document=shortened
                ).rstrip()
        prompts.append(prompt)
    return prompts


class SoftLabelEncoder(ExternalEncoder):
    """Causal language model that yields a document embedding and a soft label.

    The document is placed in a prompt that asks for its theme. The embedding is the
    last-layer hidden state at the end of the prompt, and the soft label is the
    next-token distribution at that position, restricted to a vocabulary.

    Parameters
    ----------
    model_name: str, default "baidu/ERNIE-4.5-0.3B-PT"
        Any causal language model on the Hugging Face Hub.
    instruction: str
        Instruction placed before the document.
    template: str
        Prompt template with `{instruction}` and `{document}`.
    batch_size: int, default 16
        Documents per forward pass.
    max_length: int, default 512
        Longer prompts have their document shortened (the end of the prompt is kept).
    device: str, default None
        Defaults to CUDA when available.
    """

    def __init__(
        self,
        model_name: str = "baidu/ERNIE-4.5-0.3B-PT",
        instruction: str = DEFAULT_INSTRUCTION,
        template: str = DEFAULT_TEMPLATE,
        batch_size: int = 16,
        max_length: Optional[int] = 512,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.instruction = instruction
        self.template = template
        self.batch_size = batch_size
        self.max_length = max_length
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = (
                torch.bfloat16 if self.device.type == "cuda" else torch.float32
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps the last real token at position -1 for every row.
        self.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )
            .eval()
            .to(self.device)
        )

    def encode(self, sentences: Iterable[str]) -> np.ndarray:
        """Document embeddings: the last hidden state at the end of the prompt."""
        return self.extract(list(sentences), [])[1]

    @torch.no_grad()
    def extract(
        self,
        documents: Sequence[str],
        vocabulary: Sequence[str],
        progress=None,
    ):
        """Return (logits, hidden): arrays of shape (n_docs, n_vocab) and (n_docs, hidden_size)."""
        if len(vocabulary):
            vocab_map = map_vocabulary(self.tokenizer, vocabulary)
        else:
            vocab_map = VocabularyMap(
                np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
            )
        token_ids = torch.as_tensor(vocab_map.token_ids, device=self.device)
        prompts = build_prompts(
            documents,
            self.tokenizer,
            self.instruction,
            self.template,
            self.max_length,
        )
        lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        order = np.argsort(lengths)[
            ::-1
        ]  # longest first: an out-of-memory error shows up at once
        logits = np.empty((len(prompts), len(vocabulary)), dtype=np.float32)
        hidden = None
        for start in range(0, len(order), self.batch_size):
            rows = order[start : start + self.batch_size]
            batch = self.tokenizer(
                [prompts[i] for i in rows], return_tensors="pt", padding=True
            )
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask")
            }
            out = self.model(
                **batch, use_cache=False, output_hidden_states=True
            )
            last_logits = (
                out.logits[:, -1, :]
                .index_select(-1, token_ids)
                .float()
                .cpu()
                .numpy()
            )
            last_hidden = out.hidden_states[-1][:, -1, :].float().cpu().numpy()
            if hidden is None:
                hidden = np.empty(
                    (len(prompts), last_hidden.shape[-1]), dtype=np.float32
                )
            logits[rows] = last_logits
            hidden[rows] = last_hidden
            if progress is not None:
                progress(min(start + self.batch_size, len(order)), len(order))
        return logits, hidden, vocab_map


def soft_labels(logits, temperature=3.0, mask=None):
    """softmax(logits / temperature) over the vocabulary, row-wise.

    `mask` (bool, one entry per word) selects the words that take part; the others
    get zero. Pass `VocabularyMap.single_token` to keep exactly scored words only.
    """
    z = np.asarray(logits, dtype=np.float64) / temperature
    if mask is not None and not np.all(mask):
        if not np.any(mask):
            raise ValueError(
                "No vocabulary word is a single alphabetic token for this language model."
            )
        z = np.where(np.asarray(mask)[None, :], z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)
