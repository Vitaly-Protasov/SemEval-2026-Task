from enum import StrEnum

from pydantic import BaseModel


class XMLRawFields(StrEnum):
    RID = "rid"
    ID = "id"
    TEXT = "text"
    OPINIONS = "Opinions"
    OPINION = "Opinion"
    REVIEW = "Review"
    REVIEWS = "Reviews"
    SENTENCES = "sentences/sentence"
    SENTENCES_ELEMENT = "sentences"
    SENTENCE = "sentence"


class XMLFieldsFinal(StrEnum):
    REVIEW_ID = "review_id"
    SENTENCE_ID = "sentence_id"
    TEXT = "text"
    TARGET = "target"
    CATEGORY = "category"
    POLARITY = "polarity"
    OPINION = "opinion"
    FROM = "from"
    FROM_ = "from_"
    TO = "to"
    INTENSITY = "intensity"


class XMLModelFinal(BaseModel):
    review_id: str | None = None
    sentence_id: str | None = None
    text: str | None = None
    target: str | None = None
    category: str | None = None
    polarity: str | None = None
    opinion: str | None = None
    from_: str | None = None
    to: str | None = None
    intensity: str | None = None


class SemEvalFormatQuadrupletFields(StrEnum):
    ASPECT = "Aspect"
    OPINION = "Opinion"
    CATEGORY = "Category"
    VA = "VA"


class SemEvalFormatQuadrupletModel(BaseModel):
    Aspect: str
    Opinion: str
    Category: str
    VA: str


class SemEvalFormatLineFields(StrEnum):
    ID = "ID"
    TEXT = "Text"
    QUADRUPLET = "Quadruplet"


class SemEvalFormatLineModel(BaseModel):
    ID: list[str]
    Text: list[str]
    Quadruplet: list[list[dict[str, str]]]
