# search_relevance

Minimal generative-model based relevance classification for four labels:

- Exactly Satisfied
- Substitute
- Complement
- Irrelevant

## Usage

```python
from search_relevance import predict_relevance_label


class MyModel:
    def generate(self, prompt: str) -> str:
        return "Exactly Satisfied"


label = predict_relevance_label(
    model=MyModel(),
    query="wireless mouse",
    candidate="wireless mouse",
)
```

If a model object exposes both `generate(prompt)` and `__call__(prompt)`,
`predict_relevance_label()` uses `generate(prompt)` first.