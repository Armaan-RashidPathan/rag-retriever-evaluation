# LCEL and Basic RAG

## The `Runnable` interface — why `|` works at all

Every LCEL component — a prompt template, a chat model, an output parser, a
retriever — implements the same interface: `Runnable`. Concretely, that means
each one exposes `.invoke(input) -> output` (plus `.stream()`, `.batch()`,
and async versions). Because everything shares this interface, they compose
with `|` (the pipe operator), which Python routes to `RunnableSequence`:
the output of the left side becomes the input of the right side.

That's why `prompt | model | parser` works: `prompt.invoke(...)` returns
chat messages → fed into `model.invoke(...)` → returns an `AIMessage` → fed
into `parser.invoke(...)` → returns a plain string. Three different object
types, one uniform interface.

**Bare Python functions auto-coerce.** `retriever | format_docs` works even
though `format_docs` is a plain function — LangChain wraps it in a
`RunnableLambda` automatically whenever it's used with `|`. This also works
with the callable on the *left* side (`(lambda x: x["question"]) | retriever`,
or `itemgetter("question") | retriever`), via `Runnable.__ror__`.

## Why a RAG chain needs `RunnableParallel` + `RunnablePassthrough`

A plain prompt needs one dict, e.g. `{"question": ..., "context": ...}`. But
a retriever only produces the `context` half — it takes a question string and
returns documents. The prompt needs **both** the original question and the
retrieved context, arriving together in one dict. A linear `|` chain can't do
this — each stage only has one input flowing to one output, no way to "keep
the original input around" while also branching off to compute something new
from it.

- **`RunnablePassthrough()`** — "take whatever input I receive, return it
  unchanged." Used to preserve the original input alongside a derived value.
- **`RunnableParallel({...})`** — takes one input, runs it through *multiple*
  Runnables simultaneously, returns a dict of their results.

So the canonical basic RAG chain:
```python
RunnableParallel(
    context=retriever | format_docs,     # question -> docs -> joined string
    question=RunnablePassthrough(),      # question -> question, unchanged
)
| prompt | model | StrOutputParser()
```

`context` itself is a two-step mini-chain living *inside* the parallel dict —
LCEL composes recursively.

**A recurring bug this causes:** if `question=RunnablePassthrough()`, the
chain's raw input must already be the plain question *string*. Calling
`.invoke({"question": q})` instead of `.invoke(q)` means `RunnablePassthrough`
passes through the whole dict, and the *same* raw dict also gets fed into the
`context` branch's retriever — which expects a string query and breaks
(surfaced as confusing errors deep inside the embedding model, e.g.
`AttributeError: 'dict' object has no attribute 'replace'`). This bug
recurred multiple times across the project whenever a chain got rewritten.

## The basic RAG pipeline, piece by piece

```
raw PDFs → chunking → embedding model → vector DB (Chroma)
                                              ↓
question → embed → similarity search (no LLM call here) → retrieved docs
                                              ↓
context + question → prompt → LLM → final answer
```

Important: **retrieval itself does not call an LLM.** It's pure embedding
similarity search — the question gets embedded once, compared mathematically
against pre-embedded chunks, top-k closest returned. (Multi-Query Retrieval,
covered in the next note, is the one technique that *does* add an LLM call
before retrieval — to rewrite the question, not to do the search itself.)

- **`PyPDFLoader`** — one loader per PDF, produces one `Document` per *page*
  (not per file).
- **`RecursiveCharacterTextSplitter`** — `chunk_size`/`chunk_overlap` in
  characters. Overlap matters because tables/paragraphs get cut mid-sentence
  at arbitrary boundaries otherwise.
- **`HuggingFaceEmbeddings`** (`sentence-transformers/all-MiniLM-L6-v2`) —
  chosen as a free, local, CPU-friendly embedding model (no API key, no
  cost) over `OpenAIEmbeddings`.
- **`Chroma`** — persists to disk (`chromadb/` folder: a `chroma.sqlite3`
  metadata DB + a UUID-named folder holding the HNSW vector index files).
  Never re-embed on every run — build once, persist, and reuse
  (`build_vectorstore(force=False)` pattern).
- **`vectorstore.as_retriever()`** — wraps the vectorstore's
  `similarity_search()` method (not itself a `Runnable`) in a
  `VectorStoreRetriever` object that *is* a `Runnable`, so it can be piped
  with `|`.

## Real bugs/lessons from building this

- Hardcoded absolute paths break on other machines — use
  `Path(__file__).resolve().parent.parent / "..."`.
- Bare `from ingest import ...` (not `from src.ingest import ...`) only
  works when a script happens to be run directly from inside `src/`; it
  breaks under `python -m src.xxx` or when imported by tests. **This
  project's convention: always run as `python -m src.<module>` from the
  project root.**
- `force=True` rebuilds must delete the old collection first
  (`shutil.rmtree`) — `Chroma.from_documents` *appends*, it doesn't replace,
  so repeated force-rebuilds without deleting first silently duplicate the
  entire corpus.
- A test (`test_build_vectorstore_force_rebuild`) monkeypatched the data
  loader but not the persist-directory constant, so running `pytest` once
  wiped the real 2512-chunk vectorstore and replaced it with 2 dummy test
  documents, in place. Root-caused by directly inspecting
  `vectorstore._collection.count()` and `similarity_search()` output. Lesson:
  tests that touch a `force=True`/destructive path need a temp directory
  (`tmp_path` fixture + monkeypatching the persist-dir constant too), never
  the real one.
