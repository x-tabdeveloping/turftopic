from __future__ import annotations

from abc import abstractmethod
from typing import Iterable

import numpy as np
from PIL import Image

from turftopic.base import ContextualModel

UrlStr = str

ImageRepr = [Image | UrlStr]


def _load_images(images: Iterable[ImageRepr]) -> Iterable[Image]:
    for image in images:
        if isinstance(image, str):
            image = Image.open(image)
        yield image


def _naive_join_embeddings(
    text_embeddings: np.ndarray, image_embeddings: np.ndarray
) -> np.ndarray:
    """Produces document embeddings by averaging text and image embeddings"""
    return np.mean(np.stack([text_embeddings, image_embeddings]), axis=0)


class MultimodalModel(ContextualModel):
    """Base model for multimodal topic models."""

    def encode_multimodal(
        self,
        sentences: list[str],
        images: list[ImageRepr],
    ) -> dict[str, np.ndarray]:
        if len(sentences) != len(images):
            raise ValueError("Images and documents were not the same length.")
        images = _load_images(images)
        if hasattr(self.encoder_, "get_text_embeddings"):
            text_embeddings = np.array(
                self.encoder_.get_text_embeddings(sentences)
            )
        else:
            text_embeddings = self.encoder_.encode(sentences)
        if hasattr(self.encoder_, "get_image_embeddings"):
            image_embeddings = np.array(
                self.encoder_.get_image_embeddings(images)
            )
        else:
            image_embeddings = self.encoder_.encode(images)
        if hasattr(self.encoder_, "get_fused_embeddings"):
            document_embeddings = np.array(
                self.encoder_.get_fused_embeddings(
                    texts=sentences,
                    images=images,
                )
            )
        else:
            document_embeddings = _naive_join_embeddings(
                text_embeddings, image_embeddings
            )

        return {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "document_embeddings": document_embeddings,
        }

    @abstractmethod
    def fit_transform_multimodal(
        self,
        raw_documents: Iterable[str],
        images: Iterable[tuple[ImageRepr] | ImageRepr],
        y=None,
    ) -> np.ndarray:
        pass
