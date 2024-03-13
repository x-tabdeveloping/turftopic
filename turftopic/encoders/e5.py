import warnings
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class E5Encoder(SentenceTransformer):
    """Encoder model oriented at using E5 models.
    ```python
    from turftopic.encoders import E5Encoder
    from turftopic import GMM
    model = GMM(10, encoder=E5Encoder(model_name="intfloat/multilingual-e5-small", prefix="query: "))
    ```
    Parameters
    ----------
    model_name: str
        Embedding model to use.
        Either a SentenceTransformers pre-trained models or a model from HuggingFace Hub.
    prefix : Optional[str]
        A string that gets added to the start of each document (formats each document followingly: `f"{prefix}{text}"`).
        Expected by most E5 models. Consult model cards on Hugging Face to see what prefix is expected by your specific model.
    preprocessor : Optional[Callable]
        A function that formats documents as desired.
        Overwrites `prefix` and only applies if `prefix == None`.
        Both input and output must be string.
        First argument must be input text.
        By default `None`.
    Examples
    --------
    Instructional models can also be used.
    In this case, the documents should be prefixed with a one-sentence instruction that describes the task.
    See Notes for available models and instruction suggestions.
    ```python
    from turftopic.encoders import E5Encoder
    def add_instruct_prefix(document: str) -> str:
        task_description = "YOUR_INSTRUCTION"
        return f'Instruct: {task_description}\nQuery: {document}'
    encoder = E5Encoder(model_name="intfloat/multilingual-e5-large-instruct", preprocessor=add_instruct_prefix)
    model = GMM(10, encoder=encoder)
    ```
    Or the same can be done using a `prefix` argument:
    ```python
    from turftopic.encoders import E5Encoder
    from turftopic import GMM
    prefix = "Instruct: YOUR_INSTRUCTION\nQuery: "
    encoder = E5Encoder(model_name="intfloat/multilingual-e5-large-instruct", prefix=prefix)
    model = GMM(10, encoder=encoder)
    ```
    Notes
    -----
    See available E5-based sentence transformers on Hugging Face Hub:
    https://huggingface.co/models?library=sentence-transformers&sort=trending&search=e5
    Instruction templates:
    https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py#L106
    """

    def __init__(
        self,
        model_name: str,
        prefix: Optional[str] = None,
        preprocessor: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

        # check for both prefix and preprocessor being specified
        if prefix is not None and preprocessor is not None:
            warnings.warn(
                "Both `prefix` and `preprocessor` are specified. `preprocessor` will be ignored! "
                "To avoid this warning, specify only one of them.",
            )

        # pick either prefix or preprocessor to do the job
        if prefix is not None:
            self.preprocessor = lambda x: f"{prefix}{x}"
        else:
            if preprocessor is not None:
                try:
                    assert self._is_preprocessor_valid(
                        preprocessor=preprocessor
                    )
                    self.preprocessor = preprocessor
                except:
                    raise AssertionError(
                        "`preprocessor` failed vaildation. Please make sure your preprocessor returns type `str`."
                    )
            else:
                raise ValueError(
                    "Either `prefix` or `preprocessor` must be specified."
                )

    @staticmethod
    def _is_preprocessor_valid(preprocessor: Callable) -> bool:
        """Check if preprocessor returns string."""
        input_0 = None
        input_1 = "What are assertions? and why would you use them?"
        output_0 = preprocessor(input_0)
        output_1 = preprocessor(input_1)

        return all(
            [
                isinstance(output_0, str),
                isinstance(output_1, str),
            ]
        )

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        """
        Parameters
        ----------
        sentences: list[str]
            Input text.
        Notes
        -----
        See docs for `SentenceTransformer.encode` for available **kwargs
        https://www.sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
        """

        sentences = [self.preprocessor(sentence) for sentence in sentences]

        return super().encode(sentences, **kwargs)
