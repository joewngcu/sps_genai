# Assignment 1: FastAPI Word Embedding API

This project extends the Module 3 FastAPI activity with the
spaCy word embedding functionality from Module 2.

## Requirements

- Python 3.12
- uv package manager

## Setup

From the project root, install the dependencies:

```bash
uv sync
```

The dependencies include the spaCy en_core_web_lg model.

## Run

```bash
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs to test the API.

## Endpoints

### GET /

Returns a greeting.

### POST /generate

Generates text using the bigram model.

Example request:

```json
{
  "start_word": "the",
  "length": 20
}
```

Use a single starting word from the sample corpus.
Generation may stop early if no next word is available.

### POST /embedding

Returns the query word, vector dimensions, and full embedding.

Example request:

```json
{
  "word": "apple"
}
```

The response contains a 300-dimensional embedding.
Blank input and input containing multiple whitespace-separated
words are rejected with status code 400.
Words without vectors in the model may return a zero vector.

## Code Organization

- app/main.py: API endpoints and request models.
- app/bigram_model.py: Bigram text generation logic.
- app/embedding_model.py: spaCy embedding function.

## References

Adapted from the course materials:
- Module 2 Practical 2: Word Sampling
- Module 2 Practical 3: Word Embeddings
- Module 3: Simple Text Generation FastAPI Project