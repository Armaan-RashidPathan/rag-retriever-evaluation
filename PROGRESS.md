# LangChain Mastery Project — Progress Log

A running record of what's been built, what was learned, and decisions made. Meant to be edited/revised by you as we go.

---

## Milestone 1 — LCEL Foundations + Basic RAG (Complete)

**Goal:** ingest the 3 NVIDIA annual reports, embed them, and answer questions via a basic LCEL RAG chain.

### What was built
- [`src/ingest.py`](src/ingest.py) — `load_and_split()`: loads all PDFs in `data/` via `PyPDFLoader`, splits into ~1000-char chunks (150 overlap) via `RecursiveCharacterTextSplitter`. Produces 2512 chunks from 3 reports.
- [`src/vectorstore.py`](src/vectorstore.py) — `build_vectorstore()` / `get_vectorstore()`: embeds chunks with free local `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`), persists to Chroma at `chromadb/`. `force=True` wipes and rebuilds; `force=False` (default) reuses the persisted store.
- [`src/format_docs.py`](src/format_docs.py) — joins retrieved `Document.page_content` into one string for the prompt.
- [`src/chain.py`](src/chain.py) — the LCEL chain: `RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough()) | prompt | model | StrOutputParser()`. LLM is `ChatGroq(model="llama-3.1-8b-instant")` (free tier).

### Key concepts learned
- **`Runnable` is the universal interface** — anything with `.invoke()` composes via `|`. This is why swapping `ChatOpenAI` → `ChatGroq`, or `similarity_search` → `as_retriever()`, doesn't change the rest of the chain.
- **`RunnableParallel` + `RunnablePassthrough`** solve the "the prompt needs both the original question AND derived context" problem — a plain linear chain can't branch, so `RunnableParallel` runs multiple Runnables against the same input and returns a dict.
- **Bare Python functions/lambdas auto-coerce to `RunnableLambda`** when used with `|` — no need to wrap manually.
- **`vectorstore.as_retriever()`** wraps a vectorstore's `similarity_search` in a `Runnable`, which is what makes it chainable.

### Bugs hit and root causes (worth remembering)
1. Absolute hardcoded paths in `ingest.py`/`vectorstore.py` — fixed with `Path(__file__).resolve().parent.parent / "..."`, portable across machines.
2. Typos (`iterdit`, `count_documents()`, `langchain_hugginface`) — caught by actually running the code.
3. Bare `from ingest import ...` import — broke under `python -m src.xxx` / package-style imports; fixed to `from src.ingest import ...`. **Lesson: this project must be run as `python -m src.<module>` from the project root, never `python <module>.py` from inside `src/`.**
4. `force=True` didn't clear the old collection before re-adding documents — `Chroma.from_documents` appends, doesn't replace. Fixed by `shutil.rmtree()` before rebuild.
5. **Big one:** `tests/test_vectorstore.py`'s `test_build_vectorstore_force_rebuild` monkeypatched `load_and_split` but NOT `VECTORSTORE_DIR` — so running `pytest` wiped the real persisted vectorstore (2512 real chunks) and replaced it with 2 dummy test docs, in-place. Diagnosed by inspecting `vs._collection.count()` and `similarity_search` output directly. Rebuilt via `build_vectorstore(force=True)` against the real PDFs. **The test itself is not yet fixed — still unsafe to run `pytest` until it's test-isolated (deferred to Milestone 1's task #5 / pytest lesson).**
6. Chain invoked with `chain1.invoke({"question": question})` instead of `chain1.invoke(question)` — since `question=RunnablePassthrough()` expects the chain's raw input to already be the string, not a dict.

### Known limitation (motivates Milestone 2)
Basic top-k similarity search doesn't reliably match year-specific phrasing across 3 near-identical annual reports — e.g. "NVIDIA total revenue fiscal year 2025" retrieved a *different* year's revenue table. Confirmed the LCEL/retrieval/generation plumbing is otherwise fully correct (verified with a question matching what was actually retrieved).

---

## Milestone 2 — Advanced RAG Architecture (In Progress)

**Goal:** fix the retrieval-quality gap above using Multi-Query Retrieval, Contextual Compression, and Parent Document Retrieval — and evaluate them against each other, not just against vibes.

### Plan
- **Step 0 — Eval set:** 8-10 fixed questions (easy / ambiguous-phrasing / table-dependent) with expected-answer notes, to compare strategies objectively.
- **Step 1 — Multi-Query Retrieval:** `MultiQueryRetriever.from_llm(retriever=retriever, llm=model)` — LLM rewrites the question multiple ways before retrieving.
- **Step 2 — Contextual Compression:** `ContextualCompressionRetriever` — start with `EmbeddingsFilter` (free/local), then try `LLMChainExtractor` (costs one LLM call per retrieved doc — watch Groq rate limits).
- **Step 3 — Parent Document Retriever:** new/separate index using small child chunks for search + large parent chunks (e.g. whole page) returned for context, to stop mangling financial tables. Uses `InMemoryStore` as the docstore (not persisted across restarts — noted for later).
- **Step 4 — Comparative evaluation:** run the Step 0 eval set through all four retrievers (baseline + 3 new), compare correctness, context quality, extra LLM calls, latency.

**Step 3 — Parent Document Retriever: implemented, one real bug found and fixed, result not yet re-verified.**
- Built in `src/retrievers/parent_document.py`. Refactored `src/ingest.py` to split out `load_documents()` (loading only) from `load_and_split()` (loading + splitting), since `ParentDocumentRetriever` needs raw unsplit documents to do its own two-granularity splitting (`child_splitter`: 400 chars, `parent_splitter`: 2000 chars).
- Import paths (same relocation pattern as Steps 1-2): `from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever`, `from langchain_classic.storage import InMemoryStore` (re-export of `langchain_core.stores.InMemoryStore`).
- Design decision: `InMemoryStore` never persists across runs, so `build_parent_document_retriever()` always wipes and rebuilds `chromadb_parent_child/` from scratch on every call rather than trying to reuse a persisted store — an old vectorstore with no matching docstore would silently break retrieval (child chunks found, but no parent doc to look up). Real cost: full re-embed every run, and more total chunks than Milestone 1 (400-char chunks vs. 1000-char), so this is the slowest of the three techniques to test.
- **Real bug found:** `retriever.add_documents(raw_docs)` in one call hit `chromadb.errors.InternalError: Batch size of 6444 is greater than max batch size of 5461` — Chroma caps upsert batch size, and `ParentDocumentRetriever.add_documents()` doesn't batch internally. Fixed by batching the *raw* (page-level) documents into groups of 50 before calling `add_documents()` per batch (each call gets fresh random IDs, no collision risk).
- **Same k-too-small issue as Step 2's first attempt:** `ParentDocumentRetriever`/`MultiVectorRetriever`'s `search_kwargs` defaults to `{}` (Chroma's implicit default k, ~4) — first real run retrieved 4 parent docs, none containing the FY2025 figure. Fixed by passing `search_kwargs={"k": 12}` at construction, matching Step 2's finding. **Not yet re-run to confirm this actually surfaces the correct chunk** — deferred, will show up naturally once Step 4's comparison script runs this retriever against the eval set.

**Step 4 — Comparative evaluation: skipped as a formal script; findings already gathered via manual side-by-side runs during Steps 1-3.**

Summary of the FY2025-total-revenue test case across all techniques:

| Retriever | Result | Key lesson |
|---|---|---|
| Baseline (Milestone 1) | Failed — wrong year's table | Motivated this whole milestone |
| Multi-Query | Failed — broader results, still missed target chunk | Fixes vocabulary mismatch, not "target chunk scores low for structural reasons" |
| Contextual Compression (`EmbeddingsFilter`) | **Succeeded** ("$130.5 billion") once `k` was tuned from measured scores | Real fix was `k`, not the filter itself — `EmbeddingsFilter` filters, doesn't re-rank |
| Parent Document Retriever | Implemented, same `k` fix applied, not re-verified | Real bug found/fixed (Chroma batch-size cap); result deferred |

**Milestone 2 conclusion:** the actual root cause of the FY2025 retrieval gap was insufficient `k` (too few candidates considered), not vocabulary mismatch or chunk noise — diagnosed by directly measuring cosine similarity scores rather than guessing. Multi-Query and Contextual Compression are still legitimate, separate tools for different problems (phrasing mismatch, and noise/irrelevance filtering respectively), but neither was the actual lever that mattered for this specific case. Formal `LLMChainExtractor` comparison and a from-scratch `run_eval.py` script remain deferred (tasks #11 and the original eval script) if revisited later.

## Milestone 3 — Structured Extraction + Memory (In Progress)

**Step 1 — Structured Extraction: mechanism confirmed working, one real bug found and fixed (not re-verified).**
- `src/pydantic_schemas.py`: nested Pydantic schema — `FiscalYearFinancials` containing `segment_revenues: list[SegmentRevenue]`.
- `src/extraction.py`: `model.with_structured_output(FiscalYearFinancials, method="function_calling")` — verified Groq's `llama-3.1-8b-instant` supports this correctly (tested with a trivial schema first).
- First real run returned a correctly-typed nested object, but with wrong/inconsistent data: `fiscal_yr='2026'` while the actual figures matched FY2024's table ($60,922M total revenue).
- **Root cause:** the retrieval query string was only used to search the vectorstore — it never reached the LLM prompt. The prompt said "extract... for the following year" without ever specifying *which* year, so the model guessed. This is the same `RunnablePassthrough` lesson from Milestone 1 (prompt needs both original input AND derived context) — mistakenly dropped when simplifying to a single-key `RunnableParallel` for the "no separate question needed" extraction case. Fixed by adding `query=RunnablePassthrough()` back as a second `RunnableParallel` branch and a `{query}` placeholder in the prompt. **Not yet re-run to confirm the fix resolves it** — if it still pulls the wrong year, `retriever1`'s missing explicit `k` (same Milestone 2 lesson) would be the next suspect.

**Step 2 — Conversational Memory: done.** `src/conversational_memory/conversational_chain.py` — `RunnableWithMessageHistory` wrapping the RAG chain, `MessagesPlaceholder("history")` in the prompt, `InMemoryChatMessageHistory` per session. Bugs hit along the way: wrong kwarg (`get_session_id` vs `get_session_history`), unnecessary `output_messages_key` on a plain-string-output chain (silent `KeyError` in the history callback), a bare string in `from_messages` parsed as an implicit extra `HumanMessage` rather than joining the system message, and missing `config={"configurable": {"session_id": ...}}` on invoke. `RunnableWithMessageHistory` is deprecated in favor of LangGraph persistence — used anyway since LangGraph is explicitly out of scope for this project's core-LangChain phase.

**Milestone 3 complete.**

## Milestone 4 — Tool & Function Calling (Complete)

- `src/tool_calling/tools.py`: three `@tool`-decorated functions (`calculate_growth_percent`, `get_segment_revenue` lookup, `get_current_date`), bugs fixed (wrong import, missing `datetime` import, undefined `REVENUE_DATA`). Folder was originally misnamed `tool_calling_and_agent.py` (invalid — `.py` in a directory name breaks Python's import system); renamed to `src/tool_calling/`.
- `src/tool_calling/agent.py`: manual bind-tools + execute loop (`model.bind_tools(TOOLS)` -> inspect `.tool_calls` -> run matching tool -> feed back as `ToolMessage` -> final `.invoke()`). Verified tool *selection* and argument extraction work correctly by inspecting raw `tool_calls` output directly.
- **Real finding:** the free 8B model (`llama-3.1-8b-instant`, used everywhere else in this project) reliably picks the right tool and args, but is unreliable at *synthesizing* the tool's result into a correct final answer (returned empty/vague answers, and once hallucinated a fake tool-call syntax and fabricated data instead of using the real `ToolMessage` already in context). Prompt-patching (a forceful system instruction) helped partially but not fully. Root-caused as a genuine small-model capability limit, not a code bug — confirmed by swapping to `llama-3.3-70b-versatile` (still free on Groq) for just this agent, which fixed all cases immediately, including correctly declining to call a tool for an unrelated question. Kept as a separate model instance in `agent.py` rather than changing `src/chain.py`'s shared model, to avoid touching already-verified Milestone 1-3 code.

**Milestone 4 complete — all 4 core LangChain milestones done.** Deferred/open items: task pytest lesson (#5), `LLMChainExtractor` (#11).

## Database Track — Milestone 1: PostgreSQL (Complete)

- `src/db/models.py`, `src/db/connections.py`, `src/db/persist.py` built —
  full round trip verified: LLM structured extraction → SQLAlchemy ORM
  objects → Postgres insert → read back, correct data (`year=2025,
  total_revenue_millions=130497`, segments `Compute & Networking: 116193`,
  `Graphics: 14304`).
- Real bugs fixed: missing paired `relationship()`/`back_populates` on
  `FiscalYear`, wrongly-scoped `unique=True`, bare `from models import`
  import, a `SegmentRevenue` name collision between two different modules
  (`src.db.models` vs `src.pydantic_schemas`) silently shadowing the correct
  class, and a `DetachedInstanceError` risk from reading an id after closing
  the session.
- **External issue, not a code bug:** Groq retired the entire Llama model
  family from the account mid-session (`llama-3.1-8b-instant` /
  `llama-3.3-70b-versatile` both gone). Diagnosed via `client.models.list()`,
  switched the shared model in `src/chain.py` to `openai/gpt-oss-120b`.
- This also re-surfaced the Milestone 2 "`k` too small" lesson in
  `src/extraction.py` (never previously tuned) — needed `k=20` (higher than
  Milestone 2's `k=12`) because the segment-breakdown table's actual numbers
  sit in a *second* chunk, split from its own heading by the flat 1000-char
  chunking. Full detail in `Notes/05_databases.md` and
  `Notes/07_problems_and_solutions.md` (new: a consolidated troubleshooting
  log across the entire project, organized by category).

### Progress notes

**Step 1 — Multi-Query Retrieval: implemented, working, and gave an important negative result.**
- Built in `src/retrievers/multiquery.py`: `MultiQueryRetriever.from_llm(retriever=base_retriever, llm=model)`.
- Import path note: in this project's installed version (LangChain 1.x), `MultiQueryRetriever` is NOT under `langchain.retrievers` — it moved to a separate `langchain_classic` package: `from langchain_classic.retrievers.multi_query import MultiQueryRetriever`.
- Must use `.from_llm(...)` classmethod, not the raw `MultiQueryRetriever(...)` constructor (raw constructor requires a pre-built `llm_chain`, which `.from_llm` builds for you from a plain `llm`).
- **Finding:** tested against the FY2025 total-revenue question (the same one that failed in Milestone 1). Multi-query pulled back 11 deduped documents across all 3 reports — genuinely broader than baseline — but still did NOT retrieve the chunk containing the actual figure ($130,497M, found in the 2026 report's "Geographic Revenue" table). Conclusion: multi-query fixes *vocabulary mismatch* between question and source text, but can't help when the target fact sits inside a chunk that isn't topically "about" the question (a geography-organized table, not a "total revenue" table) — no rephrasing bridges that gap. This looks like a chunking/structure problem, which points at Parent Document Retrieval (Step 3) rather than query rewriting.
- Open question to test later: does bumping base retriever's `k` (currently unset/default) surface the target chunk with a wider net? Not yet tested.

**Step 2 — Contextual Compression: implemented, plus a genuinely useful diagnostic detour.**
- Import path note (same pattern as Step 1): `ContextualCompressionRetriever` and `EmbeddingsFilter` live under `langchain_classic.retrievers` / `langchain_classic.retrievers.document_compressors`, not `langchain.retrievers`.
- Built in `src/retrievers/contextual_compression.py`: base retriever (`k=8` initially) wrapped with `ContextualCompressionRetriever(base_compressor=EmbeddingsFilter(embeddings=..., similarity_threshold=0.5), base_retriever=...)`.
- **First run:** `EmbeddingsFilter` filtered out zero documents at `threshold=0.5`. Diagnosed by computing actual cosine similarity scores manually — all 8 base-retrieved docs scored 0.68-0.81, i.e. threshold was far too permissive for how this embedding model scores dense financial-table text against a financial question. Notably, a totally irrelevant lease-schedule chunk scored *highest* (0.81) — MiniLM appears to match on surface structure ("table full of $ amounts under fiscal-year headers") more than deep topical relevance.
- **Root-caused the recurring FY2025 miss** (failed in both Step 1 and this step's first run): computed the actual target chunk's similarity score directly — the real "$130.5 billion" chunk scored 0.665, only just below the k=8 cutoff (0.677). Not a deep semantic-distance problem, just a narrow miss.
- **Fix confirmed:** raised `k` to 12 → the target chunk (page 17 of `NVIDIA-2025-Annual-Report.pdf`, "revenue surging 114% year on year to $130.5 billion") was retrieved. `EmbeddingsFilter` correctly preserved it (only dropped 1 of 12 docs — a % ratio table).
- **Honest takeaway:** for this embedding model/corpus, `k` was the lever that mattered, not the compression filter itself — `EmbeddingsFilter` at this threshold mostly passes documents through rather than meaningfully cutting noise. `EmbeddingsFilter` filters, it does not re-rank — a chunk that scores lower than an irrelevant one (like the 0.81 lease schedule) can't be filtering-boosted back to the top; only a wider `k` or a re-ranker (e.g. `CrossEncoderReranker`, seen available under the same `document_compressors` package but not yet tried) could do that.
- **End-to-end confirmation:** refactored `src/chain.py` to expose `build_rag_chain(retriever) -> Runnable` (factory function, so every retriever variant reuses the same prompt/model/parser instead of duplicating it — needed for Step 4's apples-to-apples comparison anyway). `contextual_compression.py` now does `chain = build_rag_chain(compression_retriever)`. Result: the exact FY2025 revenue question that returned "I don't know" in Milestone 1 now correctly answers **"$130.5 billion"**, fully end-to-end (k=12 base retriever + EmbeddingsFilter threshold=0.65).
- Regression caught during the `chain.py` refactor: `StrOutputParser` written without `()` (a class, not an instance) — `|` piping silently accepted it at chain-construction time but crashed at invoke with a confusing pydantic `TypeError`. Also re-introduced the `chain1.invoke({"question": ...})` dict-vs-string bug from Milestone 1 — same fix as before (`RunnablePassthrough()` needs the chain's raw input to already be the string).
- **Step 2 marked complete** — core goal (fix the FY2025 retrieval gap) proven end-to-end. `LLMChainExtractor` (second compressor flavor) deliberately deferred to a later task rather than blocking progress to Step 3.
