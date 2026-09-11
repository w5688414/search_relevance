import unittest

from search_relevance import (
    RelevanceLabel,
    build_relevance_prompt,
    normalize_relevance_label,
    predict_relevance_label,
)


class FakeModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class InvalidModel:
    generate = "not-callable"


class SearchRelevanceTests(unittest.TestCase):
    def test_prompt_includes_supported_labels(self) -> None:
        prompt = build_relevance_prompt("iphone charger", "USB-C wall charger")

        self.assertIn("Exactly Satisfied", prompt)
        self.assertIn("Substitute", prompt)
        self.assertIn("Complement", prompt)
        self.assertIn("Irrelevant", prompt)

    def test_predict_relevance_label_uses_generative_model_output(self) -> None:
        model = FakeModel("Complement")

        label = predict_relevance_label(
            model=model,
            query="gaming console",
            candidate="extra controller",
        )

        self.assertEqual(label, RelevanceLabel.COMPLEMENT)
        self.assertEqual(len(model.prompts), 1)
        self.assertIn("gaming console", model.prompts[0])
        self.assertIn("extra controller", model.prompts[0])

    def test_predict_relevance_label_accepts_callable_model(self) -> None:
        label = predict_relevance_label(
            model=lambda prompt: "Exactly Satisfied",
            query="wireless mouse",
            candidate="wireless mouse",
        )

        self.assertEqual(label, RelevanceLabel.EXACTLY_SATISFIED)

    def test_predict_relevance_label_rejects_invalid_model_object(self) -> None:
        with self.assertRaisesRegex(TypeError, "model must define a callable generate"):
            predict_relevance_label(
                model=InvalidModel(),
                query="desk lamp",
                candidate="office chair",
            )

    def test_normalize_relevance_label_extracts_label_from_sentence(self) -> None:
        label = normalize_relevance_label("The best label is Substitute.")

        self.assertEqual(label, RelevanceLabel.SUBSTITUTE)

    def test_normalize_relevance_label_rejects_ambiguous_sentence(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported relevance label"):
            normalize_relevance_label("Substitute or Complement")

    def test_predict_relevance_label_rejects_unknown_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported relevance label"):
            predict_relevance_label(
                model=FakeModel("Not sure"),
                query="desk lamp",
                candidate="office chair",
            )


if __name__ == "__main__":
    unittest.main()
