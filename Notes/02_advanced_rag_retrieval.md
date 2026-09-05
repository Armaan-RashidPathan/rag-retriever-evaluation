# Advanced RAG Retrieval

Motivated by one concrete failure: asking "What is NVIDIA's total revenue for
fiscal year 2025?" against the baseline retriever returned the **wrong
year's** revenue table. All three techniques below are different answers to
"how do we retrieve better?" — tested against that exact same failing
question so results are comparable.

## Multi-Query Retrieval

**Idea:** don't trust one phrasing of the question. An LLM generates several
alternative phrasings, each gets its own similarity search, and all the
results get merged and deduplicated. If even one rephrasing happens to match
the source text's vocabulary better, it catches what the original phrasing
missed.

**Mechanism:** `MultiQueryRetriever.from_llm(retriever=base_retriever,
llm=model)` wraps an existing retriever — it doesn't replace it. Internally:
LLM generates ~3 rephrasings → each run through the *same* base retriever →
merged, duplicates removed by content.

**Cost:** one extra LLM call *before* every retrieval (to generate
rephrasings), on top of the final answer-generation call. Free on Groq's
tier, but real latency; would cost more on a paid API.

**Finding:** tested against the FY2025 question — pulled back 11 deduped
documents across all 3 reports (genuinely broader than baseline), but still
**did not** retrieve the chunk with the real figure ($130,497M). Conclusion:
multi-query fixes *vocabulary mismatch* between question and source text. It
does not help when the target chunk is topically distant from any reasonable
rephrasing (e.g. the number sits inside a "Geographic Revenue" table
organized by country, not framed as "total revenue" at all) — no rephrasing
bridges that kind of gap.

## Contextual Compression

**Idea:** a *different* problem — assume retrieval found roughly the right
chunks, but each one is bigger/noisier than needed. A compressor sits between
retrieval and the prompt, either trimming each document's content or
dropping irrelevant documents outright, before your chain sees them.

**`ContextualCompressionRetriever(base_compressor=..., base_retriever=...)`**
— two compressor flavors:
- **`EmbeddingsFilter`** — no LLM call. Re-embeds each retrieved document,
  compares similarity to the query, drops anything below
  `similarity_threshold`. Filters whole documents; doesn't trim content.
- **`LLMChainExtractor`** — one LLM call *per retrieved document*, asks the
  model to extract just the relevant sentence(s). Actually shrinks content,
  but costs N extra calls where N = however many docs the base retriever
  returns (real rate-limit risk even on a free tier).

**The actual diagnostic process (the important part, not just the fix):**
first attempt used `k=8` and `similarity_threshold=0.5` — nothing got
filtered. Rather than guessing a new threshold, computed the real cosine
similarity scores directly:

```python
query_vec = np.array(embeddings.embed_query(question))
doc_vec = np.array(embeddings.embed_query(doc.page_content))
score = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
```

Result: every retrieved doc scored 0.68–0.81 — `0.5` was far too permissive
given the model's actual score distribution for this domain. More
interesting: a totally irrelevant lease-schedule chunk scored *highest*
(0.81), because `all-MiniLM-L6-v2` was matching on surface structure ("a
table full of $ amounts under fiscal-year headers"), not deep topical
relevance.

Then computed the *actual target chunk's* score directly (not guessed): it
scored 0.665 — just below the `k=8` cutoff (0.677 was 8th place). Not a deep
semantic-distance problem, a narrow miss. **Raising `k` to 12 fixed it** —
confirmed end-to-end, the exact failing question now correctly answers
"$130.5 billion."

**Honest conclusion:** the fix that mattered was `k`, not the compression
filter — `EmbeddingsFilter` at this threshold barely did any work (dropped 1
of 12 docs). Real lesson: **`EmbeddingsFilter` filters, it does not re-rank.**
A chunk that scores lower than an irrelevant one (0.665 vs. 0.81) can't be
promoted back to the top by a similarity threshold — only a wider `k` or an
actual re-ranker (`CrossEncoderReranker`, seen available but not used) can
do that.

## Parent Document Retriever

**Idea:** structurally different from the other two — changes how the
**index** is built, not just how it's queried. Two splitters, two
granularities:
- **Child chunks** (small, e.g. 400 chars) — what actually gets embedded and
  searched. Small chunks make for precise similarity search.
- **Parent chunks** (large, e.g. 2000 chars) — what actually gets *returned*.
  When a child chunk matches, the retriever looks up its parent and returns
  the whole thing — so the LLM sees an intact table/paragraph, not a
  fragment.

**Pieces:** two `RecursiveCharacterTextSplitter`s (child/parent), an
`InMemoryStore` (docstore: `parent_id -> full parent Document`), a *separate*
Chroma vectorstore for child chunks only, tied together by
`ParentDocumentRetriever(vectorstore=..., docstore=..., child_splitter=...,
parent_splitter=...)`. `.add_documents(raw_docs)` does the double-splitting
and populates both stores.

**Key limitation, accepted deliberately:** `InMemoryStore` never persists
across process restarts. An old *persisted* Chroma collection with no
matching docstore would mean child chunks get found but their parent lookup
returns nothing — silently broken retrieval. So `build_parent_document_retriever()`
always wipes and rebuilds both together, every call — real cost (full
re-embed every run), accepted as correct-over-fast for a learning project.

**Real bugs found:**
- `retriever.add_documents(raw_docs)` in one call hit
  `chromadb.errors.InternalError: Batch size of 6444 is greater than max
  batch size of 5461` — Chroma caps upsert batch size, and
  `ParentDocumentRetriever` doesn't batch internally. Fixed by batching the
  *raw* (page-level) documents into groups of 50 before calling
  `add_documents()` per batch.
- Same `k`-too-small issue as Contextual Compression:
  `ParentDocumentRetriever`/`MultiVectorRetriever`'s `search_kwargs` defaults
  to `{}` (Chroma's implicit default k, ~4). Fixed with
  `search_kwargs={"k": 12}` at construction — consistent with the Step 2
  finding.

## Cross-cutting lesson: import paths moved in this LangChain version

`MultiQueryRetriever`, `ContextualCompressionRetriever`, its document
compressors, and `ParentDocumentRetriever` all live under
**`langchain_classic`** in this installed version (LangChain 1.x), not
`langchain.retrievers` as older tutorials show — that package restructured
in the 1.0 rewrite. Checked proactively for each new technique going forward
after hitting this twice.

## What was deliberately not covered

- **Reranking** (`CrossEncoderReranker`) — seen sitting right next to
  `EmbeddingsFilter` in `document_compressors`, never used. This is the
  actual fix for "an irrelevant chunk scored higher than the correct one,"
  which `EmbeddingsFilter` cannot do since it only filters.
- **Hybrid search** (keyword/BM25 + vector) — pure embedding search was the
  only retrieval signal used throughout.
- **Formal RAG evaluation frameworks** (e.g. RAGAS — faithfulness, answer
  relevancy, context precision as actual metrics). An eval question set
  (`eval/questions.py`) was built with expected answers, but a formal
  comparison script (Step 4 of the milestone) was intentionally skipped —
  findings were instead gathered from the manual side-by-side runs during
  each technique's own testing (see `PROGRESS.md` for the summary table).
- **Citation/source attribution** in the final answer — the retrieved
  `Document.metadata` always carried `source`/`page`, but no chain ever
  surfaced it to the user. (Contrast: the separate Sharan BOT project *does*
  cite source video + timestamp in every answer.)
