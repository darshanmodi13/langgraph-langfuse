LANGGRAPH + LANGFUSE BOOK: NOTEBOOKLM PROMPTS
=============================================

FIRST, ONCE: upload 00-book-rules.md (this folder) to the notebook as a source.
It holds the shared rules, the chapter plan, the Langfuse UI reference and facts
tested on this setup. Every prompt tells NotebookLM to follow it, and every
chapter ticks it. (NotebookLM report prompts must be under 5,000 characters, so
the shared rules cannot live inside each prompt.)

One file per chapter. Each file has:
  STEP 1  sources to ADD to the notebook (if not added yet) and sources to TICK for that chapter
  STEP 2  the full prompt, between two scissor lines (-------- 8< --------)

How to generate a chapter:
  1. Do STEP 1 of the chapter file.
  2. Studio -> Reports -> Document (not Interactive) -> pencil icon.
  3. Delete everything in the box, paste the text between the scissor lines, Generate.
  4. One chapter per report. Never ask for several chapters at once.

Why "tick ONLY": with every source ticked, NotebookLM pulls in topics from other
chapters (that is how the first Chapter 2 ended up covering 6 chapters).

CHAPTER PLAN
  01 Setup: Ollama, uv, self-hosted Langfuse and your first trace     (main.py)
  02 Why LangGraph, and how to see inside it with Langfuse           (chapter-02.py, step2_routing.py)
  03 Tools and the agent loop                                        (step3_tools.py)
  04 Parallel work and quality loops                                 (step4_advanced.py)
  05 Logging vs tracing: sessions, users, tags and metadata
  06 Memory: checkpointers, threads and long-term stores
  07 Human in the loop: pausing for approval
  08 Streaming output
  09 Prompt management: versioning prompts in Langfuse
  10 Scores, annotation and user feedback
  11 Datasets and experiments
  12 LLM-as-a-judge with a local model
  13 Bigger graphs, cost and dashboards
  14 Capstone: the complete support ticket helper

SOURCES
  The notebook starts with 22 sources. 00-book-rules.md plus the chapters add 24 more (46 in total),
  which stays under NotebookLM's 50-source limit on the free plan.
  docs.langchain.com links cannot be imported by NotebookLM, so LangGraph pages
  are uploaded as files from ../sources-langgraph/ (01-10 already uploaded,
  11-15 are added in chapters 4, 6 and 13).

BEFORE CHAPTER 12
  Langfuse blocks localhost and host.docker.internal for LLM connections by default.
  Add this line to the .env next to docker-compose.yml, then run: docker compose up -d
    LANGFUSE_LLM_CONNECTION_WHITELISTED_HOST=host.docker.internal

After generating a chapter, check the code before sharing it:
  - every app.invoke(...) passes config={"callbacks": [CallbackHandler()]}
  - every script ends with get_client().flush()
  - uv commands only, ChatOllama with reasoning=False, START/END edges
