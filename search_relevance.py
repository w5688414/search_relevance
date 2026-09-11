from __future__ import annotations

from enum import StrEnum
from typing import Callable, Protocol


class RelevanceLabel(StrEnum):
    EXACTLY_SATISFIED = "Exactly Satisfied"
    SUBSTITUTE = "Substitute"
    COMPLEMENT = "Complement"
    IRRELEVANT = "Irrelevant"


VALID_RELEVANCE_LABELS = tuple(label.value for label in RelevanceLabel)


class GenerativeModel(Protocol):
    def generate(self, prompt: str) -> str: ...


def build_relevance_prompt(query: str, candidate: str) -> str:
    return (
        "You are a search relevance classifier.\n"
        "Choose exactly one label for how the candidate relates to the query.\n"
        "Valid labels: Exactly Satisfied, Substitute, Complement, Irrelevant.\n"
        "Return only the label.\n\n"
        f"Query: {query}\n"
        f"Candidate: {candidate}\n"
        "Label:"
    )


def normalize_relevance_label(value: str) -> RelevanceLabel:
    normalized = value.strip()

    for label in RelevanceLabel:
        if normalized.casefold() == label.value.casefold():
            return label

    for label in RelevanceLabel:
        if label.value.casefold() in normalized.casefold():
            return label

    raise ValueError(
        f"Unsupported relevance label {value!r}. Expected one of: {', '.join(VALID_RELEVANCE_LABELS)}."
    )


def predict_relevance_label(
    model: GenerativeModel | Callable[[str], str], query: str, candidate: str
) -> RelevanceLabel:
    prompt = build_relevance_prompt(query=query, candidate=candidate)

    if hasattr(model, "generate"):
        response = model.generate(prompt)
    else:
        response = model(prompt)

    return normalize_relevance_label(response)
