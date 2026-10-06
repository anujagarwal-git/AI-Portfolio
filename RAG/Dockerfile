# =============================================================================
# RegRAG app image - one image, two services (api + ui), see docker-compose.demo.yml
#
# Baked in: Python deps (CPU-only PyTorch), the embedder + reranker weights,
# the code, prompts, registry, source PDFs and the parent-document store.
# NOT baked in: the vectors (regrag-qdrant image) and the LLM (regrag-ollama).
# =============================================================================
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/hf \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# 1. Dependencies only - a separate layer, so a code change does not reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 2. The two models loaded on EVERY question, downloaded once at build time.
#    Names must match config.EMBED_MODEL / config.RERANK_MODEL - checked in step 4.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 3. Code and data (what is allowed in is listed in .dockerignore).
COPY . .
RUN uv sync --frozen --no-dev

# 4. Fail the BUILD, not the tester's first question:
#    - model names in step 2 still match config
#    - registry loads, i.e. every PDF it names is inside the image
RUN python -c "from regrag import config, registry; \
assert config.EMBED_MODEL == 'BAAI/bge-small-en-v1.5', 'EMBED_MODEL changed - update Dockerfile step 2'; \
assert config.RERANK_MODEL == 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'RERANK_MODEL changed - update Dockerfile step 2'; \
registry.load(); print('registry OK')"

# No network needed for models at run time.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
