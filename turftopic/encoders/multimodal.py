from typing import Protocol

from PIL import Image


class MultimodalEncoder(Protocol):
    """Base class for external encoder models."""

    def get_text_embeddings(
        self,
        texts: list[str],
        *,
        batch_size: int = 8,
        **kwargs,
    ): ...

    def get_image_embeddings(
        self,
        images: list[Image.Image],
        *,
        batch_size: int = 8,
        **kwargs,
    ): ...

    def get_fused_embeddings(
        self,
        texts: list[str] = None,
        images: list[Image.Image] = None,
        batch_size: int = 8,
        **kwargs,
    ): ...
