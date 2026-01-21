import itertools
from typing import Iterable, List

import numpy as np
import torch
from tqdm import trange


def batched(iterable, n: int) -> Iterable[List[str]]:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def encode_chunks(
    encoder,
    texts,
    batch_size=64,
    window_size=50,
    step_size=40,
):
    """
    Returns
    -------
    chunk_embeddings: list[np.ndarray]
        Embedding matrix of chunks for each document.
    chunk_positions: list[list[tuple[int, int]]]
        List of start and end character index of chunks for each document.
    """
    chunk_positions = []
    chunk_embeddings = []
    for start_index in trange(
        0,
        len(texts),
        batch_size,
        desc="Encoding batches...",
    ):
        batch = texts[start_index : start_index + batch_size]
        features = encoder.tokenize(batch)
        with torch.no_grad():
            output_features = encoder.forward(features)
        n_tokens = output_features["attention_mask"].sum(axis=1)
        # Find first nonzero elements in each document
        # The document could be padded from the left, so we have to watch out for this.
        start_token = torch.argmax(
            (output_features["attention_mask"] > 0).to(torch.long), axis=1
        )
        end_token = start_token + n_tokens
        for i_doc in range(len(batch)):
            _chunk_embeddings = []
            _chunk_positions = []
            for chunk_start in range(
                start_token[i_doc], end_token[i_doc], step_size
            ):
                chunk_end = min(chunk_start + window_size, end_token[i_doc])
                _emb = output_features["token_embeddings"][
                    i_doc, chunk_start:chunk_end, :
                ].mean(axis=0)
                _chunk_embeddings.append(_emb)
                chunk_text = (
                    encoder.tokenizer.decode(
                        features["input_ids"][i_doc, chunk_start:chunk_end],
                        skip_special_tokens=True,
                    )
                    .replace("[CLS]", "")
                    .replace("[SEP]", "")
                    .strip()
                )
                doc_text = texts[start_index + i_doc]
                start_char = doc_text.find(chunk_text)
                end_char = start_char + len(chunk_text)
                _chunk_positions.append((start_char, end_char))
            _chunk_embeddings = np.stack(_chunk_embeddings)
            chunk_embeddings.append(_chunk_embeddings)
            chunk_positions.append(_chunk_positions)
    return chunk_embeddings, chunk_positions
