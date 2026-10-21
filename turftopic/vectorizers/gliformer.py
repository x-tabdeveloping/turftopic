import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

try:
    from gliformer import GLiFormer
    from pydantic import BaseModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("Install GliFormer.") from e


class Topic(BaseModel):
    topics: list[str]


class ConceptVectorizer(CountVectorizer):
    def __init__(
        self,
        model_path: str = "knowledgator/gliformer-base-v1",
        schema=None,
        max_length: int | None = None,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
    ):
        self.schema = schema
        self.model_path = model_path
        self.schema = schema
        if schema is None:
            self._schema = {"topic": Topic}
        else:
            self._schema = schema
        self.model = GLiFormer.from_pretrained(
            self.model_path,
            load_tokenizer=True,
            max_length=max_length,
        )
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )

    def extract_concepts(self, text: str) -> list[str]:
        try:
            records = self.model.structure(
                text,
                self._schema,
            )
            concepts = records["topic"][0]["topics"]
            concepts = list(
                set(
                    [
                        " ".join(marker.split()).title()
                        for marker in concepts
                        if (len(marker.split()) < 4)
                        and all(part.isalnum() for part in marker.split())
                    ]
                )
            )
            return concepts
        except (KeyError, IndexError):
            return []

    def build_tokenizer(self):
        return self.extract_concepts
