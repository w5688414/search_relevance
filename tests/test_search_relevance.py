import unittest
from unittest import mock

from search_relevance import (
    DEFAULT_MODEL_NAME,
    GemmaRelevanceClassifier,
    build_messages,
    main,
    normalize_label,
)


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeTensor:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]))

    def __getitem__(self, index):
        return self.data[index]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.prompt = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        self.tokenize = tokenize
        self.add_generation_prompt = add_generation_prompt
        return "PROMPT"

    def __call__(self, prompt, return_tensors="pt"):
        self.prompt = prompt
        return FakeInputs({"input_ids": FakeTensor([[10, 20, 30]])})

    def decode(self, _tokens, skip_special_tokens=True):
        return "The best label is Complement."


class FakeModel:
    device = "cpu"

    def generate(self, **_kwargs):
        return [[10, 20, 30, 40, 50]]


class SearchRelevanceTests(unittest.TestCase):
    def test_build_messages_uses_query_and_goods_info(self):
        messages = build_messages("iphone case", "Shockproof magnetic phone case")
        self.assertIn("Exactly Satisfied", messages[0]["content"])
        self.assertIn("Query: iphone case", messages[1]["content"])
        self.assertIn("Goods info: Shockproof magnetic phone case", messages[1]["content"])

    def test_normalize_label_accepts_embedded_response(self):
        self.assertEqual(normalize_label("Prediction: Exactly Satisfied"), "Exactly Satisfied")
        self.assertEqual(normalize_label("substitute"), "Substitute")
        self.assertEqual(normalize_label("This looks irrelevant to the query."), "Irrelevant")

    def test_predict_returns_normalized_label(self):
        classifier = GemmaRelevanceClassifier(
            model_name=DEFAULT_MODEL_NAME,
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
        )
        prediction = classifier.predict("running shoes", "Sports socks")

        self.assertEqual(prediction.label, "Complement")
        self.assertEqual(prediction.query, "running shoes")
        self.assertEqual(prediction.goods_info, "Sports socks")

    def test_constructor_requires_both_injected_dependencies(self):
        with self.assertRaisesRegex(
            ValueError,
            "Provide both model and tokenizer together, or omit both.",
        ):
            GemmaRelevanceClassifier(model=FakeModel())

    def test_main_rejects_negative_temperature(self):
        with self.assertRaises(SystemExit) as context:
            with mock.patch("sys.stderr"):
                main(
                    [
                        "--query",
                        "running shoes",
                        "--goods-info",
                        "Sports socks",
                        "--temperature",
                        "-0.1",
                    ]
                )

        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
