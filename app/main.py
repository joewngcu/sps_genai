from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding
from app.image_classifier import router as image_router
from app.digit_generator import router as digit_router

app = FastAPI()
app.include_router(image_router)
app.include_router(digit_router)

# Sample corpus from the class activity.
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned "
    "and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(ge=1, le=1000)


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word.strip()

    if not word or len(word.split()) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please enter a single word."
        )

    embedding = calculate_embedding(word)

    return {
        "word": word,
        "dimensions": len(embedding),
        "embedding": embedding.tolist()
    }
