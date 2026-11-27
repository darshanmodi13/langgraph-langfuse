# LangGraph + Langfuse, locally

Learning [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) (building AI agents as graphs) and [Langfuse](https://langfuse.com) (tracing and evaluating them), using only free, local tools.

The running example is a small **support ticket helper** that grows chapter by chapter.

## What you need

- [Ollama](https://ollama.com) with the model: `ollama pull qwen3.5:9b`
- [Docker](https://www.docker.com) (to run Langfuse)
- [uv](https://docs.astral.sh/uv/) (to run Python)

## Setup

1. Create a `.env` file in the project root with your Langfuse keys and the secrets for the Langfuse containers (see `docker-compose.yml`, every value marked `# from .env`).
2. Start Langfuse: `docker compose up -d`, then open http://localhost:3000, sign up and create API keys.
3. Add the keys to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=http://localhost:3000
   ```
4. Install packages: `uv sync`

## Run a chapter

```bash
uv run src/chapter3_tools.py
```

Then open Langfuse → Tracing to see what happened inside the run.

## Lint and format

```bash
uv run ruff check --fix src/
uv run ruff format src/
```

## Folder layout

| Path | What it is |
|---|---|
| `src/` | one runnable file per chapter; `common.py` holds the shared model setup |
| `prompts/` | the prompts used to generate the book chapters with NotebookLM |
| `sources-langgraph/` | LangGraph doc pages saved from [docs.langchain.com](https://docs.langchain.com), used as NotebookLM sources |
