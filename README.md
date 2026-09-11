# search_relevance

Predict a relevance label from a search query and goods information with a Gemma instruction model.

## Labels

- Exactly Satisfied
- Substitute
- Complement
- Irrelevant

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default model id is `google/gemma-3-4b-it`. You can override it with `--model` if you have a different Gemma checkpoint available in your environment.

## Usage

```bash
python search_relevance.py \
  --query "iphone charger" \
  --goods-info "20W USB-C fast wall charger compatible with iPhone" 
```

Example output:

```json
{"query": "iphone charger", "goods_info": "20W USB-C fast wall charger compatible with iPhone", "label": "Exactly Satisfied", "raw_output": "Exactly Satisfied", "model_name": "google/gemma-3-4b-it"}
```

## Tests

```bash
python -m unittest discover -s tests
```