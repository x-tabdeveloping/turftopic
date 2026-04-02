import itertools
import warnings
from typing import Iterable, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tokenizers import Tokenizer
from tqdm import trange


def is_contextual(encoder):
    return hasattr(encoder, "encode_tokens")


Offsets = list[tuple[int, int]]
Lengths = list[int]


class LateSentenceTransformer(SentenceTransformer):
    def encode(
        self, sentences: Union[str, list[str], np.ndarray], *args, **kwargs
    ):
        warnings.warn(
            "Encoder is contextual but topic model is not using contextual embeddings. Perhaps you wanted to use another topic model."
        )
        return super().encode(sentences, *args, **kwargs)

    def _encode_tokens(
        self,
        texts,
        batch_size=32,
        show_progress_bar=True,
    ) -> tuple[list[np.ndarray], list[Offsets]]:
        """
        Returns
        -------
        token_embeddings: list[np.ndarray]
            Embedding matrix of tokens for each document.
        offsets: list[list[tuple[int, int]]]
            Start and end character of each token in each document.
        """
        token_embeddings = []
        offsets = []
        tokenizer = Tokenizer.from_pretrained(self.model_card_data.base_model)
        for start_index in trange(
            0,
            len(texts),
            batch_size,
            disable=not show_progress_bar,
            desc="Encoding tokens...",
        ):
            batch = texts[start_index : start_index + batch_size]
            features = self.tokenize(batch)
            with torch.no_grad():
                output_features = self.forward(features)
            n_tokens = output_features["attention_mask"].sum(axis=1)
            # Find first nonzero elements in each document
            # The document could be padded from the left, so we have to watch out for this.
            start_token = torch.argmax(
                (output_features["attention_mask"] > 0).to(torch.long), axis=1
            )
            end_token = start_token + n_tokens
            for i_doc in range(len(batch)):
                _token_embeddings = output_features["token_embeddings"][
                    i_doc, start_token[i_doc] : end_token[i_doc], :
                ].numpy(force=True)
                _offsets = tokenizer.encode(batch[i_doc]).offsets
                token_embeddings.append(_token_embeddings)
                offsets.append(_offsets)
        return token_embeddings, offsets

    def encode_tokens(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ):
        """Produces contextual token embeddings over all documents.

        Parameters
        ----------
        sentences: list[str]
            Documents to encode contextually.
        batch_size: int, default 32
            Size of the batch of document to encode at once.
        show_progress_bar: bool, default True
            Indicates whether a progress bar should be displayed when encoding.

        Returns
        -------
        token_embeddings: list[np.ndarray]
            Embedding matrix of tokens for each document.
        offsets: list[list[tuple[int, int]]]
            Start and end character of each token in each document.
        """
        # This is needed because the above implementation does not normalize embeddings,
        # which normally happens to document embeddings.
        token_embeddings, offsets = self._encode_tokens(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        token_embeddings = [normalize(emb) for emb in token_embeddings]
        return token_embeddings, offsets

    def encode_windows(
        self,
        sentences: list[str],
        batch_size: int = 32,
        window_size: int = 50,
        step_size: int = 40,
        show_progress_bar: bool = True,
    ):
        """Produces contextual embeddings for a sliding window of tokens similar to C-Top2Vec.

        Parameters
        ----------
        sentences: list[str]
            Documents to encode contextually.
        batch_size: int, default 32
            Size of the batch of document to encode at once.
        window_size: int, default 50
            Size of the sliding window.
        step_size: int, default 40
            Step size of the window.
            If step_size < window_size, windows will overlap.
            If step_size == window_size, then windows are separate.
            If step_size > window_size, there will be gaps between the windows.
            In this case, we throw a warning, as this is probably unintended behaviour.
        show_progress_bar: bool, default True
            Indicates whether a progress bar should be displayed when encoding.

        Returns
        -------
        window_embeddings: list[np.ndarray]
            Embedding matrix of windows for each document.
        offsets: list[list[tuple[int, int]]]
            Start and end character of each token in each document.
        """
        token_embeddings, token_offsets = self._encode_tokens(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        window_embeddings = []
        window_offsets = []
        for emb, offs in zip(token_embeddings, token_offsets):
            _offsets = []
            _embeddings = []
            for start_index in trange(0, len(emb), step_size):
                end_index = start_index + window_size
                window_emb = np.mean(emb[start_index:end_index], axis=0)
                _embeddings.append(window_emb)
                _offsets.append((offs[start_index][0], offs[end_index][1]))
            window_embeddings.append(normalize(np.stack(_embeddings)))
            window_offsets.append(_offsets)
        return window_embeddings, window_offsets
