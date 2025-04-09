from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Optional, TypedDict, Union

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from turftopic.encoders.multimodal import MultimodalEncoder

UrlStr = str

ImageRepr = Union[Image.Image, UrlStr, None]


def _load_images(images: Iterable[ImageRepr]) -> Iterable[Image]:
    for image in images:
        if image is None:
            yield None
        elif isinstance(image, str):
            yield Image.open(image)
        else:
            yield image


def _naive_join_embeddings(
    text_embeddings: np.ndarray, image_embeddings: np.ndarray
) -> np.ndarray:
    """Produces document embeddings by averaging text and image embeddings"""
    return np.nanmean(np.stack([text_embeddings, image_embeddings]), axis=0)


class MultimodalEmbeddings(TypedDict):
    """Dict type for embeddings in a multimodal context.

    Exclusively used for type checking."""

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
        """Produce multimodal embeddings of the documents passed to the model.

        Parameters
        ----------
        sentences: list[str]
            Textual documents to encode.
        images: list[ImageRepr]
            Corresponding images for each document.

        Returns
        -------
        MultimodalEmbeddings
            Text, image and joint document embeddings.

        """
        if len(sentences) != len(images):
            raise ValueError("Images and documents were not the same length.")
        if hasattr(self.encoder_, "get_text_embeddings"):
            text_embeddings = np.array(
                self.encoder_.get_text_embeddings(sentences)
            )
        else:
            text_embeddings = self.encoder_.encode(sentences)
        embedding_size = text_embeddings.shape[1]
        images = _load_images(images)
        if hasattr(self.encoder_, "get_image_embeddings"):
            image_embeddings = np.array(
                self.encoder_.get_image_embeddings(list(images))
            )
        else:
            image_embeddings = []
            for image in images:
                if image is not None:
                    image_embeddings.append(self.encoder_.encode(image))
                else:
                    image_embeddings.append(np.full(embedding_size, np.nan))
            image_embeddings = np.stack(image_embeddings)
            print(image_embeddings)
        if hasattr(self.encoder_, "get_fused_embeddings"):
            document_embeddings = np.array(
                self.encoder_.get_fused_embeddings(
                    texts=sentences,
                    images=list(images),
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

    @staticmethod
    def validate_embeddings(embeddings: Optional[MultimodalEmbeddings]):
        if embeddings is None:
            return
        try:
            document_embeddings = embeddings["document_embeddings"]
            image_embeddings = embeddings["image_embeddings"]
        except KeyError as e:
            raise TypeError(
                "embeddings do not contain document and image embeddings, can't be used for multimodal modelling."
            ) from e
        if document_embeddings.shape != image_embeddings.shape:
            raise ValueError(
                f"Shape mismatch between document_embeddings {document_embeddings.shape} and image_embeddings {image_embeddings.shape}"
            )

    def validate_encoder(self):
        if not hasattr(self.encoder_, "encode"):
            if not all(
                (
                    hasattr(self.encoder_, "get_text_embeddings"),
                    hasattr(self.encoder_, "get_image_embeddings"),
                ),
            ):
                raise TypeError(
                    "An encoder must either have an encode() method or a get_text_embeddings and get_image_embeddings method (optionally get_fused_embeddings)"
                )

    @abstractmethod
    def fit_transform_multimodal(
        self,
        raw_documents: list[str],
        images: list[ImageRepr],
        y=None,
        embeddings: Optional[MultimodalEmbeddings] = None,
    ) -> np.ndarray:
        """Fits topic model in a multimodal context and returns the document-topic matrix.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        images: list[ImageRepr]
            Images corresponding to each document.
        y: None
            Ignored, exists for sklearn compatibility.
        embeddings: MultimodalEmbeddings
            Precomputed multimodal embeddings.

        Returns
        -------
        ndarray of shape (n_documents, n_topics)
            Document-topic matrix.
        """
        pass

    def fit_multimodal(
        self,
        raw_documents: list[str],
        images: list[ImageRepr],
        y=None,
        embeddings: Optional[MultimodalEmbeddings] = None,
    ):
        """Fits topic model on a multimodal corpus.

        Parameters
        ----------
        raw_documents: iterable of str
            Documents to fit the model on.
        images: list[ImageRepr]
            Images corresponding to each document.
        y: None
            Ignored, exists for sklearn compatibility.
        embeddings: MultimodalEmbeddings
            Precomputed multimodal embeddings.

        Returns
        -------
        Self
            The fitted topic model
        """
        self.fit_transform_multimodal(raw_documents, images, y, embeddings)
        return self

    @staticmethod
    def collect_top_images(
        images: list[Image.Image],
        image_topic_matrix: np.ndarray,
        n_images: int = 20,
        negative: bool = False,
    ) -> list[list[Image.Image]]:
        top_images: list[list[Image.Image]] = []
        for image_topic_vector in image_topic_matrix.T:
            if negative:
                image_topic_vector = -image_topic_vector
            top_im_ind = np.argsort(-image_topic_vector)[:20]
            top_im = [images[i] for i in top_im_ind]
            top_images.append(top_im)
        return top_images

    @staticmethod
    def _image_grid(
        images: list[Image.Image],
        final_size=(1200, 1200),
        grid_size: tuple[int, int] = (4, 4),
    ):
        grid_img = Image.new("RGB", final_size, (255, 255, 255))
        cell_width = final_size[0] // grid_size[0]
        cell_height = final_size[1] // grid_size[1]
        n_rows, n_cols = grid_size
        for idx, img in enumerate(images[: n_rows * n_cols]):
            img = img.resize(
                (cell_width, cell_height), resample=Image.Resampling.LANCZOS
            )
            x_offset = (idx % grid_size[0]) * cell_width
            y_offset = (idx // grid_size[1]) * cell_height
            grid_img.paste(img, (x_offset, y_offset))
        return grid_img

    def plot_topics_with_images(self, n_cols: int = 3, grid_size: int = 4):
        """Plots the most important images for each topic, along with keywords.

        Note that you will need to `pip install plotly` to use plots in Turftopic.

        Parameters
        ----------
        n_cols: int, default 3
            Number of columns you want to have in the grid of topics.
        grid_size: int, default 4
            The square root of the number of images you want to display for a given topic.
            For instance if grid_size==4, all topics will have 16 images displayed,
            since the joint image will have 4 columns and 4 rows.

        Returns
        -------
        go.Figure
            Plotly figure containing top images and keywords for topics.
        """
        if not hasattr(self, "top_images"):
            raise ValueError(
                "Model either has not been fit or was fit without images. top_images property missing."
            )
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
            images = self.top_images[i]
            image = self._image_grid(
                images, (width, height), grid_size=(grid_size, grid_size)
            )
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
