# Structured Extraction and Memory

These are two separate capabilities, not part of the retrieve→answer flow —
both are things you can layer on top of a RAG chain, independently.

## Structured Extraction

**The problem it solves:** a normal RAG chain's output is always a plain
string ("$130.5 billion"). Useless if you want to *do something* with the
data programmatically — store it, compute with it, compare fields across
years — without regexing a sentence apart.

**Mechanism:** define a Pydantic `BaseModel` describing the desired shape,
then `model.with_structured_output(YourSchema)` returns a *new* Runnable
(same `.invoke()` interface as always) that returns an instance of
`YourSchema` instead of an `AIMessage`. Under the hood this uses the LLM's
native tool/function-calling ability to force valid structured output — not
a "please respond in JSON" prompt trick. Verified directly that the free
Groq model (`llama-3.1-8b-instant`) supports this correctly before building
around it.

**The actual schema used** (`src/pydantic_schemas.py`):
```python
class SegmentRevenue(BaseModel):
    segment_name: str
    revenue_millions: float

class FiscalYearFinancials(BaseModel):
    fiscal_yr: str
    total_revenue_millions: float
    segment_revenues: list[SegmentRevenue]
    gross_margin_percent: float | None = None
```
The nested `list[SegmentRevenue]` is the "complex nested schema" part — the
LLM has to correctly populate a variable-length list of sub-objects, not
just fill in scalars.

**Composition pattern — same LCEL idea as everywhere else:**
```python
extraction_chain = RunnableParallel(context=retriever | format_docs) | prompt | structured_model
```
`structured_model` (the result of `with_structured_output`) is a `Runnable`
just like `model` was — `prompt | structured_model` composes exactly like
`prompt | model` did.

**Real bug, and why it matters generally:** first version's `RunnableParallel`
only had one key (`context`) — no separate branch preserving the original
query. The prompt template said "extract financial data for **the following
year**" without ever saying *which* year, because the query string used for
retrieval search never actually reached the LLM's prompt. Result: extracted
FY2024's real numbers but mislabeled them `fiscal_yr='2026'` — the model had
to guess a year and got the wrong one, not because retrieval failed, but
because the instruction itself was missing information. **This is the exact
`RunnablePassthrough` lesson from basic RAG, recurring in a new context** —
whenever a prompt needs both the original input and something derived from
it, you need the two-branch `RunnableParallel`, not a shortcut:
```python
RunnableParallel(context=retriever | format_docs, query=RunnablePassthrough())
```

**What this project's structured extraction was *not* used for:** despite
sounding similar to "structuring output for a multi-agent system" or "for
another tool to consume," in this project it was purely "turn one LLM's
prose answer into typed, attribute-accessible data." It was never wired into
Milestone 4's tool-calling (which used separate `@tool`-decorated functions
with their own typed arguments) or into any multi-agent handoff. That's a
valid general use case for structured output — just not what happened here.

Note: as of this writing, the extracted data still only lives in memory for
one script run — this is the direct motivation for the current database
work (persisting the same shape into real Postgres tables; see
`05_databases.md`).

## Conversational Memory

**The problem it solves:** every chain built in Milestones 1–2 is stateless
— each `.invoke()` starts fresh, no memory of prior turns. A real multi-turn
session needs "how does that compare to last year?" to resolve "that" from
the previous exchange.

**Three pieces:**
1. **`MessagesPlaceholder("history")`** — a prompt component that inserts a
   *list* of past messages directly into the prompt (as real alternating
   human/AI messages), not a string summary you wrote yourself.
2. **A session-history store** — a function `get_session_history(session_id)
   -> BaseChatMessageHistory`, backed by a dict cache of
   `InMemoryChatMessageHistory` objects, one per session.
3. **`RunnableWithMessageHistory`** — wraps the *entire chain*. Before each
   `.invoke()`, fetches that session's history and injects it wherever
   `MessagesPlaceholder` expects it; after the call, appends the new
   question+answer back into that history. Invoked with
   `config={"configurable": {"session_id": "..."}}`.

**Design decision:** built as a new file
(`src/conversational_memory/conversational_chain.py`) rather than modifying
the shared `build_rag_chain()` factory in `src/chain.py`, since that function
is imported by four other files (multi-query, compression, parent-document,
extraction) — changing its signature would cascade breakage across all of
them.

**Real bugs hit, in order:**
- `get_session_id=` instead of the actual parameter name
  `get_session_history=`.
- `output_messages_key="answer"` left in from a copy-paste — unnecessary
  (and silently breaks) when the chain's output is a plain string, not a
  dict with an `"answer"` key. Only needed for dict-shaped chain outputs.
- A bare string entry in `ChatPromptTemplate.from_messages([...])` (meant to
  continue the system message) actually gets parsed as an **implicit
  separate `HumanMessage`**, not appended to the system tuple above it —
  verified directly by invoking the prompt and printing each resulting
  message's type. Fixed by merging it into one `("system", "...")` tuple.
- Missing `config={"configurable": {"session_id": ...}}` on `.invoke()` — the
  error message here was self-explanatory and named the exact fix.

`RunnableWithMessageHistory` itself is flagged deprecated in favor of
LangGraph's built-in persistence — used anyway since LangGraph is explicitly
out of scope until a later phase of this mastery project (per the original
plan: core LangChain first, LangGraph/LangSmith later).

**Known limitation carried forward:** like `ParentDocumentRetriever`'s
docstore, `InMemoryChatMessageHistory` doesn't persist across process
restarts — this is one of the concrete motivations for the Redis milestone
in the database track (a Redis-backed chat history would actually survive).
