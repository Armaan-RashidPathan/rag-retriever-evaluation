# Pydantic vs. SQLAlchemy — what they are, how they differ, how they're used here

Both look similar at a glance — a Python class with typed attributes:
```python
class FiscalYearFinancials(BaseModel):   # Pydantic
    fiscal_yr: str
    total_revenue_millions: float

class FiscalYear(Base):                  # SQLAlchemy
    __tablename__ = "fiscal_year"
    year = Column(Integer)
    total_revenue_millions = Column(Numeric)
```
That surface similarity is exactly what makes it easy to conflate them. They
solve two genuinely different problems.

## Pydantic: validating and shaping data *in memory*

A Pydantic `BaseModel` is a runtime data-validation/parsing tool. When you
create an instance, Pydantic checks the incoming values against the declared
types and either coerces them or raises a `ValidationError`. That's it — a
Pydantic object doesn't know anything about files, databases, or persistence.
The moment your Python program ends, it's gone, unless you've explicitly
written its data somewhere durable yourself.

**Where it's used in this project:**
- Everywhere LangChain itself validates its own internal objects (that's why
  malformed constructor calls throughout this project surfaced as
  `pydantic_core.ValidationError` — `Runnable`, `ChatPromptTemplate`, etc.
  are all built on Pydantic under the hood).
- Deliberately, in Milestone 3: `src/pydantic_schemas.py` defines
  `SegmentRevenue` and `FiscalYearFinancials`, used as the *contract* for
  `model.with_structured_output(FiscalYearFinancials)` — forcing an LLM's
  free-text answer into a typed, attribute-accessible object
  (`result.segment_revenues[0].segment_name` instead of regexing a string).
  That object lived only for the duration of one script run.

**Nesting in Pydantic** (`segment_revenues: list[SegmentRevenue]`) is just
ordinary Python object composition — a list containing other validated
objects. There's no database-style "relationship" concept here; it's exactly
as if you'd written a list of dataclasses.

## SQLAlchemy: mapping Python objects to *persistent database rows*

A SQLAlchemy declarative class (inheriting `Base`) defines a **table's
shape** — each class attribute is a column, with a real SQL type
(`Integer`, `Numeric`, `String`...). An *instance* of that class represents
one **row**, but critically: constructing an instance does **not** touch the
database. Nothing is written until you explicitly do so through a `Session`:

```python
row = FiscalYear(year=2025, total_revenue_millions=130497)   # in memory only
session.add(row)      # queued for insert
session.commit()      # the actual INSERT happens now
```

**Where it's used in this project:** `src/db/models.py` (`FiscalYear`,
`SegmentRevenue`), `src/db/connections.py` (engine + session setup), and
`src/db/persist.py` (translating extracted data into rows and committing
them) — see `05_databases.md`.

**Relationships in SQLAlchemy** (`relationship(..., back_populates=...)`)
are a fundamentally different mechanism from Pydantic nesting: they represent
a real **foreign-key join between two tables**, exposed on the Python side as
a collection attribute (`fiscal_year_row.revenues`). Appending a
`SegmentRevenue` to that collection is how you tell SQLAlchemy "set this
row's foreign key to point at this parent" — it fills in `fiscal_year_id`
for you at commit time. This is why `back_populates` requires a *matching*
declaration on both classes (see the earlier note in this project's
history): the two sides are describing one real join, from two directions.

## Side-by-side

| | Pydantic | SQLAlchemy (declarative ORM) |
|---|---|---|
| **Purpose** | Validate/parse data at runtime | Map Python objects to database tables/rows |
| **Base class** | `BaseModel` | `DeclarativeBase` (your `Base`) |
| **Does constructing an instance touch anything external?** | No — pure in-memory validation | No — nothing happens until `session.add()` + `session.commit()` |
| **"Nesting"/relationships** | Plain object composition (a list of other Pydantic objects) | A real foreign-key join, materialized as a collection attribute |
| **Lifespan** | Exists only for the running process, unless you persist it yourself | Represents durable rows once committed — survives process restarts |
| **Used in this project for** | Structured LLM output (`FiscalYearFinancials` from `with_structured_output`) | Persisting that same *kind* of data permanently into Postgres |

## How the two connect in this project — the bridge function

They never talk to each other automatically — there is no shared base class
or magic conversion. The connection is a function *you* write, translating
field-by-field between two independently-defined schemas that happen to
represent similar real-world data:

```python
def save_financials(data):                       # data: a Pydantic FiscalYearFinancials
    fiscal_year_row = FiscalYear(                 # building a SQLAlchemy row
        year=int(data.fiscal_yr),                 # data.fiscal_yr (Pydantic attr, str)
        total_revenue_millions=data.total_revenue_millions,
        gross_margin_percentage=data.gross_margin_percent,
    )
    for segment in data.segment_revenues:          # looping the Pydantic list
        fiscal_year_row.revenues.append(            # building + attaching SQLAlchemy rows
            SegmentRevenue(segment=segment.segment_name, amount=segment.revenue_millions)
        )
    session.add(fiscal_year_row)
    session.commit()
```

The key realization: `data.fiscal_yr` (Pydantic's naming) and `year=`
(SQLAlchemy's column naming) don't have to match, and generally won't,
because the two classes were defined independently, for different purposes,
possibly at different times. Nothing connects them by name automatically —
this function *is* the connection, written by hand, field by field.
