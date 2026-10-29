# Applied Generative AI Coursework API

This project extends the Module 3 FastAPI activity with the
spaCy word embedding functionality from Module 2, a CIFAR-10 image classifier,
and the Assignment 3 MNIST GAN digit generator.

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

- app/main.py: API endpoints; registers the image classification and digit generation routers.
- app/bigram_model.py: Bigram text generation logic.
- app/embedding_model.py: spaCy embedding function.
- app/cnn_model.py: CNN architecture for CIFAR-10 classification.
- app/image_classifier.py: Loads the CNN and serves POST /classify.
- app/gan_model.py: Assignment 3 GAN Generator and Discriminator.
- app/digit_generator.py: Loads the GAN generator and serves POST /generate-digit.
- train_cnn.py, train_gan.py: Training scripts.
- tests/test_gan.py: GAN architecture and API tests.

## References

Adapted from the course materials:
- Module 2 Practical 2: Word Sampling
- Module 2 Practical 3: Word Embeddings
- Module 3: Simple Text Generation FastAPI Project

## Assignment 3: MNIST GAN

The generator and discriminator in `app/gan_model.py` follow the supplied
Assignment 3 architecture. The generator maps a 100-value Gaussian noise vector
to a 28x28 grayscale image. This is an unconditional GAN: it generates a random
digit and does not accept a requested digit label.

### Architecture

| Generator layer | Output shape for batch B |
| --- | --- |
| Noise | B x 100 |
| Linear 100 -> 6272, reshape | B x 128 x 7 x 7 |
| ConvTranspose2d 128 -> 64, kernel 4, stride 2, padding 1 | B x 64 x 14 x 14 |
| BatchNorm2d(64), ReLU | B x 64 x 14 x 14 |
| ConvTranspose2d 64 -> 1, kernel 4, stride 2, padding 1, Tanh | B x 1 x 28 x 28 |

| Discriminator layer | Output shape for batch B |
| --- | --- |
| Input image | B x 1 x 28 x 28 |
| Conv2d 1 -> 64, kernel 4, stride 2, padding 1, LeakyReLU(0.2) | B x 64 x 14 x 14 |
| Conv2d 64 -> 128, kernel 4, stride 2, padding 1 | B x 128 x 7 x 7 |
| BatchNorm2d(128), LeakyReLU(0.2), flatten | B x 6272 |
| Linear 6272 -> 1, Sigmoid | B x 1 real/fake probability |

The final Sigmoid makes the discriminator output a probability, so training
uses `BCELoss`. There is no extra activation after the generator's initial
Linear layer. The generator has 765,761 trainable parameters; the discriminator
has 138,817.

### Train

From the repository root:

```bash
uv sync --locked
uv run python train_gan.py --epochs 30 --device auto
```

The script downloads MNIST, uses all 60,000 training images, and maps pixel
values from [0, 255] to [-1, 1] to match Tanh. It alternates discriminator and
generator updates with batch size 128 and Adam (learning rate 0.0002,
betas 0.5 and 0.999). The discriminator sees real targets of 1 and detached
generated images with targets of 0. The generator is trained against targets
of 1 through a discriminator whose weights are temporarily frozen.

`--device auto` chooses CUDA, then Apple MPS, then CPU. The default seed is 42.
Fixed seeds make the data order and sampled noise repeatable; results can still
differ across hardware and library versions.

Outputs are `models/gan_mnist.pt`, `models/gan_metrics.csv`,
`models/gan_training_config.json`, and a fixed-noise 8x8 sample grid for every
epoch in `models/gan_samples/`. The checkpoint includes both models and Adam
states. To continue an interrupted run with the same configuration and device:

```bash
uv run python train_gan.py --epochs 40 --device auto --resume models/gan_mnist.pt
```

`--epochs` is the total desired epoch count, including completed epochs. A quick
pipeline check is available with `--epochs 1 --max-batches 2 --output-dir tmp/gan-smoke`;
this does not constitute a full training run.

### Generate a handwritten digit

Start the existing API:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

POST `/generate-digit` returns one actual 28x28 grayscale PNG. An optional seed
repeats the same sample from the same checkpoint:

```bash
curl -X POST http://127.0.0.1:8000/generate-digit \
  -H 'Content-Type: application/json' \
  -d '{"seed": 42}' --output digit.png
```

The request can also use `{}` or omit the body to generate a random sample.
Open `http://127.0.0.1:8000/docs` to see the route. Seeds must be integers from
0 to 9223372036854775807. A `digit` field is rejected because the required model
has no label conditioning.

The API loads the trained checkpoint lazily on CPU, uses evaluation mode and
inference mode, and converts the Tanh output back to [0, 255]. Set
`GAN_MODEL_PATH` to override the checkpoint location. If weights are absent or
incompatible, the route returns HTTP 503 with the training command. Restart
the server after replacing weights because the loaded generator is cached.
The existing text, embedding, and classification routes remain registered.

### Verify

```bash
uv run python -m unittest discover -s tests -v
```

The tests cover the architecture, image dimensions, PNG output, deterministic
seeds, request validation, checkpoint errors, and registration alongside the
existing routes. Training evidence is in `models/gan_metrics.csv`,
`models/gan_training_config.json`, and `models/gan_samples/`.

### GAN references

- Supplied Assignment 3 PDF: architecture and seven theory questions.
- [PyTorch ConvTranspose2d documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html): output-size formula.
- [PyTorch BatchNorm documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html): population variance for training-batch normalization.
