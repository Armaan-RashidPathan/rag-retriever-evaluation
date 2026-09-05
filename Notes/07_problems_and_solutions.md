# Problems & Solutions — Full Troubleshooting Log

Every real bug/issue hit across this project, in the order encountered, with
root cause and fix. Grouped by phase. (`PROGRESS.md` has the same events in
strict chronological order with less explanation; this file is organized for
"I'm hitting something familiar, let me scan for it.")

## Environment / tooling

| Problem | Root cause | Solution |
|---|---|---|
| Imports shown red/unresolved in the editor, but code ran fine from the terminal | Editor's language server (Pylance) pointed at a different Python than `.venv` | Created `.vscode/settings.json` with `"python.defaultInterpreterPath": ".venv\\Scripts\\python.exe"`; reload window if it recurs |
| `ModuleNotFoundError` when running a file directly (`python module.py` from inside `src/`) but it "should" work | Bare imports (`from ingest import ...`) only resolve when a script's own directory is on `sys.path` — breaks under package-style imports | Project convention: always run as `python -m src.<module>` from the project root; use `from src.xxx import yyy` everywhere |
| PowerShell `docker run \` multi-line command threw parser errors | PowerShell doesn't use `\` for line continuation like bash | Put the whole command on one line |
| `docker run` appeared to hang indefinitely | First-time image pull is slow (one layer needed several retries) — not actually stuck | Pulled the image directly in a separate terminal to watch real progress before assuming it's broken |
| Docker commands failed with a daemon-connection error even though `docker --version` worked | Docker Desktop app wasn't actually running, only the CLI was installed | Launch Docker Desktop, poll `docker ps` until it responds |

## LCEL / chain construction (recurring bug, 3+ separate times)

| Problem | Root cause | Solution |
|---|---|---|
| `chain.invoke({"question": q})` throws confusing errors deep inside the embedding model (`AttributeError: 'dict' object has no attribute 'replace'`) | `RunnablePassthrough()` expects the chain's raw input to already be the plain string — passing a dict means the *whole dict* gets sent to both the `question` slot and the retriever (which needs a string) | `chain.invoke(q)`, not `chain.invoke({"question": q})`, whenever `question=RunnablePassthrough()` |
| Extraction chain returned data for the wrong year and mislabeled it | The retrieval query string was only used for search — the prompt never actually received *which* year to target, because a single-key `RunnableParallel` dropped the original input entirely | Added a second `RunnableParallel` branch (`query=RunnablePassthrough()`) so the prompt gets both derived context AND the original input |
| `StrOutputParser` piped into a chain with no error at construction time, but crashed at invoke with a pydantic `TypeError` | Wrote `StrOutputParser` (the class) instead of `StrOutputParser()` (an instance) — LangChain tried to call the class like a function | Always instantiate: `StrOutputParser()` |

## Import paths that moved in this installed LangChain version (1.x)

Hit repeatedly across Milestone 2 — `langchain.retrievers` doesn't exist at
all in this version; everything relocated to **`langchain_classic`**:
- `MultiQueryRetriever` → `langchain_classic.retrievers.multi_query`
- `ContextualCompressionRetriever`, `EmbeddingsFilter`, `LLMChainExtractor` →
  `langchain_classic.retrievers` / `langchain_classic.retrievers.document_compressors`
- `ParentDocumentRetriever` → `langchain_classic.retrievers.parent_document_retriever`
- `InMemoryStore` → works from both `langchain_core.stores` and
  `langchain_classic.storage` (the latter re-exports the former)

**Lesson that stuck:** before building against a new class from an older
tutorial, check the actual installed package layout first (`Glob`/`Grep` in
site-packages) rather than guessing an import path and iterating on errors.

## Retrieval-quality bugs (Milestone 2, and recurring later in Milestone 3 extraction)

| Problem | Root cause | Solution |
|---|---|---|
| `MultiQueryRetriever(retriever=..., llm=...)` raised a pydantic `ValidationError` about a missing `llm_chain` field | Called the raw constructor instead of the `.from_llm(...)` classmethod, which builds `llm_chain` for you | `MultiQueryRetriever.from_llm(retriever=retriever, llm=model)` |
| `EmbeddingsFilter(similarity_threshold=0.5)` filtered out zero documents | Guessed the threshold instead of measuring — actual scores for this embedding model/domain were 0.68–0.81, so `0.5` was far too permissive | Computed real cosine similarity scores directly (`np.dot(query_vec, doc_vec) / (norms)`) before picking a threshold |
| The FY2025 revenue question kept returning the wrong year's data, across three different techniques | Retriever's default `k` (~4) was simply too small — the correct chunk scored 0.665, just below the k=8 cutoff (0.677) | Raised `k` (8→12, and later 12→20 for a harder case in extraction) — confirmed by measuring the target chunk's actual score, not guessing |
| `ParentDocumentRetriever.add_documents(raw_docs)` crashed: `Batch size of 6444 is greater than max batch size of 5461` | Chroma caps upsert batch size; `ParentDocumentRetriever` doesn't batch internally | Split `raw_docs` into batches of 50 pages before calling `add_documents()` per batch |
| Structured extraction returned `segment_revenues=[]` (empty) even though total revenue extracted correctly | The segment-breakdown table was split by chunking exactly where the numbers start ("...our two reportable segments are 'Compute & Networking' and 'Graphics':" — chunk ends there) — the continuation chunk with real numbers only appeared at rank ~19 | Raised `k` to 20 for this specific retriever so the continuation chunk made it into context (a real, deliberate limitation of flat chunking — the actual long-term fix would be Parent Document Retriever, not more `k`) |

## Structural/naming mistakes

| Problem | Root cause | Solution |
|---|---|---|
| `ModuleNotFoundError` trying to run `python -m src.tool_calling_and_agent.py.tools` | Directory was literally named `tool_calling_and_agent.py` — a `.` inside a folder name isn't a valid identifier segment for Python's import system | Renamed folder to `src/tool_calling/` |
| `vectorstore.py`'s `from langchain_hugginface import ...` failed | Typo — missing a `g` (`langchain_huggingface`) | Fixed the typo |
| `persist.py`: `SegmentRevenue(segment=..., amount=...)` failed with a validation-style error | Imported `SegmentRevenue` from **two different modules** (`src.db.models` and `src.pydantic_schemas`) — the second import silently shadowed the first, so the name pointed at the wrong (Pydantic) class with different field names | Removed the duplicate import, kept only the SQLAlchemy one |
| `models.py`: `back_populates="revenues"` on `SegmentRevenue` raised a mapper-configuration error | `back_populates` is a paired declaration — `FiscalYear` needs a matching `relationship(..., back_populates="fiscal_year")` attribute literally named `revenues`, which didn't exist yet | Added `revenues = relationship("SegmentRevenue", back_populates="fiscal_year")` to `FiscalYear` |
| `models.py`: `segment = Column(String, unique=True)` would have broken on the second fiscal year's insert | `unique=True` on `segment` alone means "Data Center" can only exist **once in the entire table**, not once per year | Removed `unique=True` (a correct version would need a composite constraint on `(fiscal_year_id, segment)`, not needed yet) |

## SQLAlchemy session lifecycle

| Problem | Root cause | Solution |
|---|---|---|
| Risk of `DetachedInstanceError` when accessing `fiscal_year_row.id` after `session.close()` | `session.commit()` expires all attributes by default (so the next read gets fresh DB state) — reading an expired attribute from a *closed* session fails, since there's no connection left to refresh from | Read `fiscal_year_row.id` into a plain variable immediately after `commit()`, before `close()` |

## External/API changes (not this project's code at all)

| Problem | Root cause | Solution |
|---|---|---|
| `groq.NotFoundError: The model 'llama-3.1-8b-instant' does not exist or you do not have access to it` — broke every chain in the project simultaneously | Groq retired/restricted the entire Llama model family from this account between sessions — an external provider change, not a code regression | Listed actually-available models via the Groq API directly (`client.models.list()`) instead of guessing a replacement; switched to `openai/gpt-oss-120b` (also independently validated — it's what the separate Sharan BOT project already uses in production) |
| After the model swap, structured extraction failed with `Tool choice is required, but model did not call a tool` | Not the new model's fault — retrieval (see above) genuinely didn't contain FY2025 data for that specific call, and the new model correctly tried to explain that in prose instead of forcing a fabricated structured answer; `with_structured_output` has no graceful fallback for a refused tool call | Fixed the underlying retrieval (`k`) rather than the symptom |

## General lessons that generalize beyond any one bug

- **When something "isn't working," verify each stage independently** rather
  than guessing at the whole pipeline — e.g. tool *selection* vs tool-result
  *synthesis* turned out to be separately testable and had different
  reliability; multi-query's rephrasings vs. the base retriever's scoring
  were similarly separable.
- **Measure, don't guess**, when a similarity/threshold-based system
  misbehaves — every retrieval fix in this project came from directly
  computing real scores, not adjusting a number and hoping.
- **A duplicated file/import is a maintenance trap** — every time this
  project copy-pasted a file or re-imported the same name from two places,
  it silently reintroduced an already-fixed bug or shadowed a needed class.
