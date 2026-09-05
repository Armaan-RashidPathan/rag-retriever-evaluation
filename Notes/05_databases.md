# Databases (In Progress)

Started after all 4 core LangChain milestones were complete. Goal: cover the
databases relevant to this domain (RAG systems, financial NLP, multi-agent
pipelines) hands-on, extending *this same project* rather than building
disconnected toy exercises — same mentorship style as the LangChain track
(concept explained, you build it, reviewed together).

## The plan

| Milestone | DB | Hands-on tie-in to this project |
|---|---|---|
| 1 (current) | PostgreSQL | Persist Milestone 3's `FiscalYearFinancials` extraction (previously printed once and discarded) into real tables |
| 2 | pgvector (Postgres extension) | Re-embed the NVIDIA chunks into Postgres, compare SQL-based similarity search against the tuned Chroma retriever from `02_advanced_rag_retrieval.md` |
| 3 | Redis | Replace `InMemoryChatMessageHistory` / `InMemoryStore` (both noted as non-persistent limitations in the LangChain notes) with real persistent equivalents |
| 4 | MongoDB | Store something genuinely document-shaped that doesn't fit SQL well — e.g. full session logs or raw chunk+metadata records |
| stretch | Neo4j | Small GraphRAG experiment — optional, more niche |

Why Postgres before a dedicated vector DB: `pgvector` lets Milestone 1 and
Milestone 2 share the *same* database instance — SQL and vector search
together, which is what a lot of real production RAG stacks actually use,
rather than stitching two separate systems together.

## Environment setup

Postgres runs in **Docker** (not installed natively on Windows) — image
`pgvector/pgvector:pg16` (plain Postgres 16 with the `pgvector` extension
available but not yet enabled — chosen from the start so Milestone 2 doesn't
require migrating to a different image later).

```powershell
docker run -d --name rag_app_postgres -e POSTGRES_USER=rag_app -e POSTGRES_PASSWORD=Arnira -e POSTGRES_DB=rag_app -p 5432:5432 -v project1_pgdata:/var/lib/postgresql/data pgvector/pgvector:pg16
```
- `-v project1_pgdata:/var/lib/postgresql/data` — named Docker volume, so
  data survives even if the container itself is stopped/removed.
- Verify it's actually accepting connections: `docker exec rag_app_postgres
  pg_isready -U rag_app`.

Python side: `sqlalchemy` (2.0.51) + `psycopg[binary]` (psycopg 3.3.4) — the
ORM and the Postgres driver, respectively.

**Note on the container/pull:** first `docker run` attempt appeared to
"hang" — root cause was simply a slow first-time pull of the
`pgvector/pgvector:pg16` image (one layer needed several retries). Confirmed
by pulling the image directly and watching real progress rather than
guessing; once cached locally, `docker run` returned immediately on retry.

## Concept: schema design (relational modeling)

The step *before* writing queries — deciding what tables exist and how they
relate, based on the actual shape of the data. The Pydantic schema from
Milestone 3:
```python
class FiscalYearFinancials(BaseModel):
    fiscal_yr: str
    total_revenue_millions: float
    segment_revenues: list[SegmentRevenue]
    gross_margin_percent: float | None = None
```
The nested `list[SegmentRevenue]` is a **one-to-many relationship** — one
fiscal year has many segments. In SQL this becomes **two tables**, not one
flattened table with repeated columns: `fiscal_years` (one row per year) and
`segment_revenues` (one row per segment *per year*, with a foreign key
column pointing back to which `fiscal_years` row it belongs to). This is the
relational-modeling equivalent of Pydantic's nesting.

**SQLAlchemy's role:** rather than hand-writing `CREATE TABLE` SQL, tables
are defined as Python classes (an ORM) — each class becomes a table, each
attribute a column, relationships between classes become foreign keys.

## Progress

- [x] Postgres running natively (not Docker, per correction — see below) and
      verified.
- [x] SQLAlchemy + psycopg installed.
- [x] `src/db/models.py` — `FiscalYear` (`year`, `total_revenue_millions`,
      `gross_margin_percentage`) / `SegmentRevenue` (`segment`, `amount`,
      `fiscal_year_id` FK), linked by a paired `relationship(...,
      back_populates=...)` on both sides.
- [x] `src/db/connections.py` — engine + `get_session()` + `create_all`.
- [x] `src/db/persist.py` — `save_financial(data)` translates a Pydantic
      `FiscalYearFinancials` into `FiscalYear`/`SegmentRevenue` ORM rows and
      commits them; `__main__` does the full round trip (LLM extraction →
      save → read back from Postgres) and confirms correct data end to end.
- [ ] Milestone 2 (pgvector) onward.

**Milestone 1 (PostgreSQL) complete.**

### Note: Docker vs. native Postgres

Started this milestone by running Postgres in a Docker container
(`rag_app_postgres`, `pgvector/pgvector:pg16` image) — walked through
launching Docker Desktop, the image pull, and verifying with `pg_isready`.
Partway through, corrected that the actual target is a **native** Postgres
install on this machine instead, already verified independently. Everything
after that point (schema, connection string, persistence) targets the native
install; the Docker container may still exist unused.

### Unplanned real-world issue hit during this milestone: Groq model retirement

While testing `persist.py`, `llama-3.1-8b-instant` (used everywhere via
`src/chain.py`'s shared `model`) started failing with `model_not_found` — the
entire Llama family had been removed from the Groq account between sessions,
an external provider change unrelated to any code here. Diagnosed by listing
actually-available models via `client.models.list()` rather than guessing a
replacement, then switched the shared model to `openai/gpt-oss-120b`
(cross-validated: it's what the separate Sharan BOT project already uses in
production). See `07_problems_and_solutions.md` for the full detail,
including a second issue this briefly surfaced (a `with_structured_output`
call failing when the model correctly refused to fabricate data for a
retrieval gap that turned out to be the familiar `k`-too-small issue from
Milestone 2, now also hit in `src/extraction.py`, fixed the same way).

*(This file will keep growing as the database track progresses — see
`PROGRESS.md` for the terser bug/fix log alongside this conceptual one.)*
