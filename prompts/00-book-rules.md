# Book rules: LangGraph + Langfuse with local models

These rules apply to every chapter of this book. When a prompt says "follow 00-book-rules.md", follow everything below.

## 1. Book style

- Write like a real programming book chapter, not like an AI assistant: a title, a "What you will learn" list, a "Before you start" list, numbered sections, complete runnable code, and a short recap.
- The reader is a web developer (TypeScript) who is new to Python and has no ML background. Use simple English and short sentences. Explain every new word the first time it appears.
- Use web-dev comparisons when they help: Express middleware, Redux, Promise.all, optional chaining, Chrome DevTools Network tab, Sentry, Git.
- Explain code line by line. Explain any Python syntax a TypeScript developer may not know (TypedDict, Literal, type hints, decorators, list[-1], dict.get, getattr, with open(...)).
- Teach each topic in this order: the problem -> the plain Python way -> what goes wrong -> how the tool fixes it -> the code -> what to look for in Langfuse.
- End every chapter with: (a) a 5-line summary, (b) 5 quiz questions with answers at the very end, (c) the chapter's exercise, with the expected result and hints but not the full solution.

## 2. Setup

- Local models only, through Ollama: `ChatOllama(model="qwen3.5:9b", temperature=0, reasoning=False)` from `langchain_ollama`. The option is `reasoning`, not `thinking`. No paid APIs. When a source uses OpenAI or Anthropic, rewrite the example with ChatOllama.
- Python projects use uv: `uv add <package>`, `uv run file.py`. Never pip or venv.
- Self-hosted Langfuse v4 (Docker Compose) at http://localhost:3000, Python SDK v5. A `.env` file holds `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_BASE_URL`, loaded with `load_dotenv()` from python-dotenv.
- LangGraph 1.x. Use `START` and `END` with `add_edge` (never `set_entry_point` or `set_finish_point`). `Literal["node_name", END]` is valid, as the official docs show.
- In nodes, read state with `state.get("key")` so a missing key does not crash.
- Strings in code use real `\n`. Never escape characters in code blocks.
- To draw a graph, write `app.get_graph().draw_mermaid()` to a `graph.mmd` file and view it at mermaid.live. Avoid `draw_mermaid_png()`: it calls the online mermaid.ink service and can time out.

## 3. Langfuse rules (every script)

- A LangGraph run is traced ONLY if the Langfuse CallbackHandler is passed to invoke:

```python
from langfuse.langchain import CallbackHandler
result = app.invoke(input, config={"callbacks": [CallbackHandler()]})
```

  Without it, no trace is created.
- Every script ends with `get_client().flush()` (`from langfuse import get_client`), because short scripts can exit before traces are sent.
- Only describe what the Langfuse UI really shows according to the sources. Never claim automatic behaviour the sources do not describe.

## 4. Chapter plan (the running project is a "support ticket helper")

1. Setup: Ollama, uv, self-hosted Langfuse, first trace
2. Why LangGraph (plain Python vs graph), Langfuse trace basics, routing with Command and conditional edges
3. Tools, ToolNode, bind_tools, the agent loop
4. Send, parallel workers, super-steps, reducers, max_concurrency, evaluator-optimizer loops, retry limits
5. Logging vs tracing: sessions, user IDs, tags, metadata, trace names, environments
6. Memory: checkpointers, thread_id, short-term vs long-term memory, stores
7. Human in the loop: interrupt(), Command(resume=...)
8. Streaming
9. Langfuse prompt management
10. Scores, annotation, user feedback
11. Datasets and experiments
12. LLM-as-a-judge with a local model
13. Subgraphs, token and cost tracking, dashboards
14. Capstone

Each chapter builds on the code from earlier chapters. Never restart the project. Cover only the current chapter's topics; topics of other chapters must not be taught, even if a source mentions them.

## 5. Langfuse UI reference

- A trace = one `invoke()` call. Observations = the steps inside it, nested like a waterfall (like the DevTools Network waterfall or a Sentry trace).
- One request shows several rows because the graph run, each node and each LLM call are separate observations nested inside each other.
- Observation types: CHAIN = the graph run and each node; GENERATION = an LLM call. Only generations have a model and token usage.
- The Traces view shows one row per request. The Observations view shows one row per step.
- GENERATION fields: input (the exact prompt sent), output, model, input/output/total tokens, latency, time to first token. Local models have no cost unless you define a price.
- Useful metadata: `langgraph_node` (which node), `langgraph_step` (step number), `langgraph_triggers` (why the node ran), `ls_model_name`, `ls_provider`, `ls_temperature`.
- A node's output shows only the update it returned. The graph's output shows the full final state.
- Formatted view shows readable messages; Raw view shows the real JSON (type, id, additional_kwargs, response_metadata).
- Tabs: Preview = input and output; Metadata = extra labels; Scores = grades attached to the trace; Log View = all steps as one flat list in time order.
- Checking whether thinking was off: Langfuse does not record the reasoning setting. With `reasoning=True`, ChatOllama puts the thinking text in `additional_kwargs["reasoning_content"]` and output tokens jump.
- Ollama's `response_metadata` includes `load_duration` in nanoseconds; a large value means a cold start while the model loads into memory.
- `get_client()` vs `CallbackHandler`: the handler records LangChain/LangGraph runs automatically; `get_client()` is the shared Langfuse client for `flush()`, scores, prompts, datasets and manual tracing. The handler uses the same client internally.

## 6. Facts verified on this setup (state them as facts)

Tools (Chapter 3):
- A plain function without `@tool` still works in `bind_tools` and `ToolNode` (they convert it).
- With no docstring, `@tool` and `ToolNode` raise "Function must have a docstring if description not provided.", but `bind_tools` silently sends an empty description to the LLM.
- With no type hints, the LLM receives no argument type.
- Tool descriptions are sent with every request and cost input tokens even when no tool is used.
- The prebuilt `tools_condition` returns "tools", so the tool node must be named "tools".

Parallel work (Chapter 4):
- `Send` objects must be returned from the function given to `add_conditional_edges`; `add_edge` only takes node names.
- Nodes in the same super-step run in parallel; the next super-step starts only after all of them finish, so a synthesizer runs once, after the slowest worker.
- Without a reducer, two parallel workers writing the same key raise `InvalidUpdateError: Can receive only one value per step. Use an Annotated key to handle multiple values.`
- `config={"max_concurrency": 3}` limits parallel workers inside one `invoke()`. Tested: 6 one-second workers started at 0s, 0s, 0s, 1s, 1s, 1s and took 2s.
- `llm.with_structured_output(Model)` returns a new wrapper and leaves `llm` unchanged; create one per schema.
- An evaluator loop that routes back to a node that ignores the feedback repeats the same output until the recursion limit (default 10,007 super-steps from LangGraph 1.1.4) raises `GraphRecursionError`. Route back to the node that creates content, make it read `state.get("feedback")`, and add an attempts limit.
- With `operator.add`, re-running workers appends new answers to the old ones.

Sessions (Chapter 5):
- The LangChain CallbackHandler reads these keys from `config["metadata"]`: `langfuse_session_id`, `langfuse_user_id`, `langfuse_tags` (a list) and `langfuse_trace_name`.
- Langfuse does not group runs by LangGraph's `thread_id` automatically; pass `langfuse_session_id` yourself.

LLM-as-a-judge (Chapter 12):
- Langfuse runs in Docker, so it reaches Ollama at `http://host.docker.internal:11434/v1`.
- Langfuse blocks `localhost` and `host.docker.internal` for LLM connections by default. On self-hosted Langfuse, add `LANGFUSE_LLM_CONNECTION_WHITELISTED_HOST=host.docker.internal` to the `.env` next to docker-compose.yml, then run `docker compose up -d` (data is kept).
