from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Optional, TypedDict

import numpy as np
from PIL import Image

UrlStr = str

ImageRepr = [Image.Image | UrlStr]


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


class MultimodalEmbeddings(TypedDict):
    text_embeddings: np.ndarray
    image_embeddings: np.ndarray
    document_embeddings: np.ndarray


class MultimodalModel:
    """Base model for multimodal topic models."""

    def encode_multimodal(
        self,
        sentences: list[str],
        images: list[ImageRepr],
    ) -> dict[str, np.ndarray]:
        if len(sentences) != len(images):
            raise ValueError("Images and documents were not the same length.")
        images = list(_load_images(images))
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
            image_embeddings = np.stack(
                [self.encoder_.encode(image) for image in images]
            )
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
        raw_documents: list[str],
        images: list[ImageRepr],
        y=None,
        embeddings: Optional[MultimodalEmbeddings] = None,
    ) -> np.ndarray:
        pass

    def fit_multimodal(
        self,
        raw_documents: list[str],
        images: list[ImageRepr],
        y=None,
        embeddings: Optional[MultimodalEmbeddings] = None,
    ):
        self.fit_transform_multimodal(raw_documents, images, y, embeddings)
        return self

    def _top_image_grid(self, ind_topic: int, final_size=(1200, 1200)):
        grid_size = (3, 3)
        grid_img = Image.new("RGB", final_size, (255, 255, 255))
        cell_width = final_size[0] // grid_size[0]
        cell_height = final_size[1] // grid_size[1]
        for idx, img in enumerate(self.top_images[ind_topic]):
            img = img.resize(
                (cell_width, cell_height), resample=Image.Resampling.LANCZOS
            )
            x_offset = (idx % grid_size[0]) * cell_width
            y_offset = (idx // grid_size[1]) * cell_height
            grid_img.paste(img, (x_offset, y_offset))
        return grid_img

    def plot_topics_with_images(self, n_columns: int = 3):
        try:
            import plotly.graph_objects as go
        except (ImportError, ModuleNotFoundError) as e:
            raise ModuleNotFoundError(
                "Please install plotly if you intend to use plots in Turftopic."
            ) from e
        fig = go.Figure()
        width, height = 1200, 1200
        scale_factor = 0.25
        w, h = width * scale_factor, height * scale_factor
        padding = 10
        n_cols = 3
        n_components = self.components_.shape[0]
        n_rows = n_components // n_cols + int(bool(n_components % n_cols))
        figure_height = (h + padding) * n_rows
        figure_width = (w + padding) * n_cols
        fig = fig.add_trace(
            go.Scatter(
                x=[0, figure_width],
                y=[0, figure_height],
                mode="markers",
                marker_opacity=0,
            )
        )
        vocab = self.get_vocab()
        for i, component in enumerate(self.components_):
            col = i % n_cols
            row = i // n_cols
            top_7 = vocab[np.argsort(-component)[:7]]
            image = self._top_image_grid(i, (width, height))
            x0 = (w + padding) * col
            y0 = (h + padding) * (n_rows - row)
            fig = fig.add_layout_image(
                dict(
                    x=x0,
                    sizex=w,
                    y=y0,
                    sizey=h,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    layer="below",
                    sizing="stretch",
                    source=image,
                ),
            )
            fig.add_annotation(
                x=(w + padding) * col + (w / 2),
                y=(h + padding) * (n_rows - row) - (h / 2),
                text="<b> " + "<br> ".join(top_7),
                font=dict(
                    size=16,
                    family="Times New Roman",
                    color="white",
                ),
                bgcolor="rgba(0,0,0, 0.5)",
            )
        fig = fig.update_xaxes(visible=False, range=[0, figure_width])
        fig = fig.update_yaxes(
            visible=False,
            range=[0, figure_height],
            # the scaleanchor attribute ensures that the aspect ratio stays constant
            scaleanchor="x",
        )
        fig = fig.update_layout(
            width=figure_width,
            height=figure_height,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )
        return fig
