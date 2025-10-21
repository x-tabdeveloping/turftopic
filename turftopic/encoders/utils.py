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
    sentences,
    batch_size=64,
    window_size=50,
    step_size=40,
    return_chunks=False,
    show_progress_bar=False,
):
    chunks = []
    chunk_embeddings = []
    for start_index in trange(
        0,
        len(sentences),
        batch_size,
        desc="Encoding batches...",
        disable=not show_progress_bar,
    ):
        batch = sentences[start_index : start_index + batch_size]
        features = encoder.tokenize(batch)
        with torch.no_grad():
            output_features = encoder.forward(features)
        n_tokens = output_features["attention_mask"].sum(axis=1)
        for i_doc in range(len(batch)):
            for chunk_start in range(0, n_tokens[i_doc], step_size):
                chunk_end = min(chunk_start + window_size, n_tokens[i_doc])
                _emb = output_features["token_embeddings"][
                    i_doc, chunk_start:chunk_end, :
                ].mean(axis=0)
                chunk_embeddings.append(_emb)
                if return_chunks:
                    chunks.append(
                        encoder.tokenizer.decode(
                            features["input_ids"][i_doc, chunk_start:chunk_end]
                        )
                        .replace("[CLS]", "")
                        .replace("[SEP]", "")
                    )
    if not return_chunks:
        chunks = None
    chunk_embeddings = np.stack(chunk_embeddings)
    return chunk_embeddings, chunks
