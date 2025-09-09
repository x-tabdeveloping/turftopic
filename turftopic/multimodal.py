from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Optional, TypedDict, Union

import numpy as np
from PIL import Image

from turftopic.data import TopicData

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


def encode_multimodal(
    encoder, sentences: list[str], images: list[ImageRepr]
) -> dict[str, np.ndarray]:
    """Produce multimodal embeddings of the documents passed to the model.

    Parameters
    ----------
    encoder
        MTEB or SentenceTransformer compatible embedding model.
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
    if hasattr(encoder, "get_text_embeddings"):
        text_embeddings = np.array(encoder.get_text_embeddings(sentences))
    else:
        text_embeddings = encoder.encode(sentences)
    embedding_size = text_embeddings.shape[1]
    images = list(_load_images(images))
    if hasattr(encoder, "get_image_embeddings"):
        image_embeddings = np.array(encoder.get_image_embeddings(images))
    else:
        image_embeddings = []
        for image in images:
            if image is not None:
                image_embeddings.append(encoder.encode(image))
            else:
                image_embeddings.append(np.full(embedding_size, np.nan))
        image_embeddings = np.stack(image_embeddings)
    if hasattr(encoder, "get_fused_embeddings"):
        document_embeddings = np.array(
            encoder.get_fused_embeddings(
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
        return encode_multimodal(self.encoder_, sentences, images)

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
            top_im = [images[int(i)] for i in top_im_ind]
            top_images.append(top_im)
        return top_images

    def prepare_multimodal_topic_data(
        self,
        corpus: list[str],
        images: list[ImageRepr],
        embeddings: Optional[MultimodalEmbeddings] = None,
    ) -> TopicData:
        """Produces multimodal topic inference data for a given corpus, that can be then used and reused.
        Exists to allow visualizations out of the box with topicwizard.

        Parameters
        ----------
        corpus: list[str]
            Documents to infer topical content for.
        images: list[ImageRepr]
            Images belonging to the documents.
        embeddings: MultimodalEmbeddings
            Embeddings of documents.

        Returns
        -------
        TopicData
            Information about topical inference in a dictionary.
        """
        if embeddings is None:
            embeddings = self.encode_multimodal(corpus, images)
        document_topic_matrix = self.fit_transform_multimodal(
            corpus, images=images, embeddings=embeddings
        )
        dtm = self.vectorizer.transform(corpus)  # type: ignore
        try:
            classes = self.classes_
        except AttributeError:
            classes = list(range(self.components_.shape[0]))
        res = TopicData(
            corpus=corpus,
            document_term_matrix=dtm,
            vocab=self.get_vocab(),
            document_topic_matrix=document_topic_matrix,
            document_representation=embeddings["document_embeddings"],
            topic_term_matrix=self.components_,  # type: ignore
            transform=getattr(self, "transform", None),
            topic_names=self.topic_names,
            classes=classes,
            has_negative_side=self.has_negative_side,
            hierarchy=getattr(self, "hierarchy", None),
            images=images,
            top_images=self.top_images,
            negative_images=getattr(self, "negative_images", None),
        )
        return res
