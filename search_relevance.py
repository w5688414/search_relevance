import argparse
import json
import re
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_MODEL_NAME = "google/gemma-3-4b-it"
LABELS = (
    "Exactly Satisfied",
    "Substitute",
    "Complement",
    "Irrelevant",
)


def _simplify_label(value: str) -> str:
    return re.sub(r"[^a-z]+", "", value.lower())


NORMALIZED_LABELS = {
    _simplify_label("Exactly Satisfied"): "Exactly Satisfied",
    _simplify_label("Exact Satisfied"): "Exactly Satisfied",
    _simplify_label("Exact Match"): "Exactly Satisfied",
    _simplify_label("Substitute"): "Substitute",
    _simplify_label("Complement"): "Complement",
    _simplify_label("Irrelevant"): "Irrelevant",
}

SYSTEM_PROMPT = (
    "You are a search relevance classifier. "
    "Given a user query and goods information, return exactly one label from: "
    "Exactly Satisfied, Substitute, Complement, Irrelevant. "
    "Do not explain your answer."
)


@dataclass
class RelevancePrediction:
    query: str
    goods_info: str
    label: str
    raw_output: str
    model_name: str


def build_messages(query: str, goods_info: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Query: {query}\n"
                f"Goods info: {goods_info}\n"
                "Return one label only."
            ),
        },
    ]


def normalize_label(raw_output: str) -> str:
    candidates = [raw_output.strip(), *[line.strip() for line in raw_output.splitlines()]]
    for candidate in candidates:
        simplified = _simplify_label(candidate)
        if simplified in NORMALIZED_LABELS:
            return NORMALIZED_LABELS[simplified]

    simplified_output = _simplify_label(raw_output)
    for simplified_label, canonical_label in NORMALIZED_LABELS.items():
        if simplified_label in simplified_output:
            return canonical_label

    raise ValueError(
        "Model output did not contain a supported label. "
        f"Expected one of {', '.join(LABELS)} but received: {raw_output!r}"
    )


class GemmaRelevanceClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if (model is None) != (tokenizer is None):
            raise ValueError("Provide both model and tokenizer together, or omit both.")

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = model
        self.tokenizer = tokenizer
        self._torch = None
        self._input_device = None

        if self.model is None or self.tokenizer is None:
            self._load_runtime()

    def _load_runtime(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        preferred_device = self._select_device(torch)
        if preferred_device == "cuda":
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        elif preferred_device in {"mps", "xpu"}:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if preferred_device in {"mps", "xpu"}:
            self._input_device = torch.device(preferred_device)
            self.model = self.model.to(self._input_device)

    @staticmethod
    def _select_device(torch: Any) -> str | None:
        if torch.cuda.is_available():
            return "cuda"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"

        xpu_runtime = getattr(torch, "xpu", None)
        if xpu_runtime is not None and xpu_runtime.is_available():
            return "xpu"

        return None

    def _build_prompt(self, query: str, goods_info: str) -> str:
        messages = build_messages(query, goods_info)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return "\n\n".join(message["content"] for message in messages) + "\nLabel:"

    def predict(self, query: str, goods_info: str) -> RelevancePrediction:
        prompt = self._build_prompt(query, goods_info)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self._input_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(self._input_device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = 0.9

        context_manager = self._torch.inference_mode() if self._torch else nullcontext()
        with context_manager:
            output = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        raw_output = self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()
        label = normalize_label(raw_output)
        return RelevancePrediction(
            query=query,
            goods_info=goods_info,
            label=label,
            raw_output=raw_output,
            model_name=self.model_name,
        )


def predict_relevance_label(
    query: str,
    goods_info: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> str:
    classifier = GemmaRelevanceClassifier(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return classifier.predict(query, goods_info).label


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict a search relevance label with Gemma.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--goods-info", required=True, help="Product or goods description.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model id for a Gemma instruction model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of generation tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic decoding.",
    )
    args = parser.parse_args(argv)
    if args.temperature < 0:
        parser.error("--temperature must be greater than or equal to 0.")

    classifier = GemmaRelevanceClassifier(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    prediction = classifier.predict(args.query, args.goods_info)
    print(json.dumps(asdict(prediction), ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
