# Notes — What, Why, How

This is the teaching-reference companion to `PROGRESS.md`. `PROGRESS.md` is a
terse log of what was built and which bugs got fixed, in order. **This folder
is the "why does this work" reference** — the concepts, the reasoning behind
design decisions, and the mental models, organized by topic rather than by
chronological order. Updated as new topics are covered.

## Index

- [`01_lcel_and_basic_rag.md`](01_lcel_and_basic_rag.md) — the `Runnable`
  interface, LCEL composition, `RunnableParallel`/`RunnablePassthrough`, and
  the basic RAG pipeline (loader → splitter → embeddings → vectorstore →
  retriever → chain).
- [`02_advanced_rag_retrieval.md`](02_advanced_rag_retrieval.md) — Multi-Query
  Retrieval, Contextual Compression, Parent Document Retriever, and the real
  retrieval-quality diagnosis process (measuring similarity scores directly
  instead of guessing).
- [`03_structured_extraction_and_memory.md`](03_structured_extraction_and_memory.md)
  — Pydantic + `with_structured_output`, and conversational memory via
  `RunnableWithMessageHistory`.
- [`04_tool_calling.md`](04_tool_calling.md) — `@tool`, `bind_tools`, the
  manual tool-execution loop, and the small-model tool-synthesis limitation
  found while building it.
- [`05_databases.md`](05_databases.md) — the database mastery track
  (PostgreSQL → pgvector → Redis → MongoDB), started after the core LangChain
  milestones.
- [`06_pydantic_vs_sqlalchemy.md`](06_pydantic_vs_sqlalchemy.md) — what each
  one actually is, how they differ (in-memory validation vs. persistent row
  mapping), and how the project bridges them by hand in `save_financials`.
- [`07_problems_and_solutions.md`](07_problems_and_solutions.md) — every real
  bug hit across the whole project, grouped by category, with root cause and
  fix — the "I've seen something like this before" reference.

## Project context

Two projects are referenced throughout:
- **This project** (`Project1`) — a deliberate LangChain mastery exercise:
  a RAG system over 3 NVIDIA annual report PDFs, built milestone by milestone
  specifically to exercise LCEL, advanced retrieval, structured extraction,
  memory, and tool calling.
- **Sharan BOT** (`C:\Users\dihsa\Projects\Sharan BOT`) — a separate, already
  built and deployed real-world RAG project (financial education chatbot),
  referenced for comparison — e.g. it already has source citations with
  timestamps, something Project1 never built.
