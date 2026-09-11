from __future__ import annotations

from enum import Enum
import re
from typing import Callable, Protocol


class RelevanceLabel(str, Enum):
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

    matches = {
        label
        for label in RelevanceLabel
        if re.search(
            rf"(?<!\w){re.escape(label.value)}(?!\w)",
            normalized,
            flags=re.IGNORECASE,
        )
    }

    if len(matches) == 1:
        return matches.pop()
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous relevance label {value!r}. Expected exactly one of: {', '.join(VALID_RELEVANCE_LABELS)}."
        )

    raise ValueError(
        f"Unsupported relevance label {value!r}. Expected one of: {', '.join(VALID_RELEVANCE_LABELS)}."
    )


def predict_relevance_label(
    model: GenerativeModel | Callable[[str], str], query: str, candidate: str
) -> RelevanceLabel:
    """Predict a relevance label from a generative model response.

    If ``model`` defines a callable ``generate(prompt)`` method, that interface
    is used. Otherwise, ``model`` itself must be callable and will be invoked
    with the prompt.
    """
    prompt = build_relevance_prompt(query=query, candidate=candidate)
    generate = getattr(model, "generate", None)

    if callable(generate):
        response = generate(prompt)
    elif callable(model):
        response = model(prompt)
    else:
        raise TypeError("model must define a callable generate(prompt) method or be callable")

    return normalize_relevance_label(response)
