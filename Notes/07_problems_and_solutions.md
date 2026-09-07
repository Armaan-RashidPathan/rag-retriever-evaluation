# Problems & Solutions — Full Troubleshooting Log (Plain-Language Version)

This file walks through every real bug or issue hit across this project, in
plain language, grouped by topic. For each one: what went wrong, why it
actually happened underneath, and what fixed it. (`PROGRESS.md` has the same
events in strict chronological order, much more tersely — use this file when
you want the actual explanation, not just a reminder of what happened.)

---

## Environment and tooling problems

### Editor showed import errors, but the code ran fine in the terminal

VS Code uses a tool called Pylance to check your code and underline problems
in red before you even run it. Pylance does this by looking at whichever
Python installation it thinks your project uses. The problem was that
Pylance was looking at a *different* Python than the one in `.venv` (which
is the one that actually has all your packages installed) — so it correctly
couldn't find `langchain`, `pydantic`, etc., even though running the file
for real worked perfectly, because the terminal *was* using `.venv`.

**Fix:** created `.vscode/settings.json` telling VS Code explicitly which
Python to use:
```json
{ "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe" }
```
If it comes back after this, reload the VS Code window — Pylance sometimes
needs a nudge to notice the setting changed.

### A script worked when run one way, but not another

Early on, `python ingest.py` (run from inside the `src` folder) worked fine.
Later, importing that same file from somewhere else (like a test, or another
script) failed with `ModuleNotFoundError`. The reason: when you run a file
directly, Python quietly adds that file's own folder to its search path, so
`from ingest import ...` can find `ingest.py` sitting right next to it. But
the moment you run things the "proper" way — as a package, from the project
root — that shortcut doesn't apply anymore, and Python needs the full path:
`from src.ingest import ...`.

**The rule adopted for this whole project:** always run scripts as
`python -m src.<module_name>` from the project's root folder, and always
write imports as `from src.xxx import yyy`, never the short form. This one
rule prevented an entire category of confusing errors for the rest of the
project.

### A PowerShell command with backslash line breaks threw parser errors

A `docker run` command was written across several lines using `\` at the end
of each line to continue onto the next — this is how you do it in Bash. But
the terminal being used was PowerShell, which doesn't understand `\` as
"continue the command on the next line" the way Bash does — it just sees a
broken, incomplete command. **Fix:** put the whole command on one line
instead of trying to use PowerShell's own continuation character (which is a
backtick, and finicky even then).

### `docker run` seemed to hang forever

The very first time a Docker image is used, Docker has to download
("pull") it from the internet — and `pgvector/pgvector:pg16` is a few
hundred megabytes across many layers. One of those layers happened to need
several retries due to a flaky connection, which made the whole command sit
there with no visible output for a couple of minutes, looking exactly like
it had frozen. It hadn't — it was just slow. **How this was confirmed rather
than assumed:** ran the pull command directly and watched its real,
line-by-line progress output, which showed it actively downloading (and
retrying one layer) rather than being stuck. Once the image was fully
cached locally, running the container again was instant.

### Docker commands failed even though Docker was "installed"

`docker --version` worked, but every actual `docker run`/`docker ps`
command failed with a connection error. The Docker *command-line tool* was
installed, but the **Docker Desktop application** — which runs the actual
background service those commands talk to — wasn't running. It's the
difference between having a phone and it being turned on. **Fix:** launch
Docker Desktop, then wait (checking every few seconds) until `docker ps`
starts responding normally, which means the background service is finally
up.

---

## The recurring LCEL bug: chains needing both the original input and derived context

This exact category of bug showed up **three separate times**, in three
different chains, because it's an easy trap to fall into once you understand
the pattern but forget to apply it consistently.

### The core problem in plain terms

Imagine a chain whose prompt needs two things: the user's original question,
*and* some context you looked up based on that question (e.g. retrieved
document text). A plain, single-path chain can only pass one thing from step
to step — it can't remember the original question once it's been used to
look something up, unless you deliberately tell it to keep a copy.

`RunnablePassthrough()` is LangChain's way of saying "take whatever came in,
and also just hand it through unchanged, alongside whatever else gets
computed." Used inside `RunnableParallel`, it lets you get back a dictionary
with **both** the original input and the derived value, ready for a prompt
that needs both.

### Bug 1 — calling the chain with a dictionary when it expected a plain string

When a chain is built expecting the raw input to *already be* the plain
question text (because `RunnablePassthrough()` will just hand that same text
through), calling it like `chain.invoke({"question": my_question})` instead
of `chain.invoke(my_question)` breaks things in a very confusing way: the
*whole dictionary* gets treated as "the input," including inside the part of
the chain that does document retrieval — which expects a plain search
string, not a dictionary. The resulting error appears deep inside the
embedding model's code (`AttributeError: 'dict' object has no attribute
'replace'`), which gives no obvious hint that the real problem is one level
up, in how the chain was called. **Fix:** always match how you call
`.invoke()` to what the chain was actually built to expect.

### Bug 2 — the extraction chain silently didn't know which year to look for

A chain was built to extract financial data, with a prompt that said
"extract financial data for **the following year**." The chain looked up
context correctly, but the actual word "2025" — which was only ever used to
search the database — never made it into the text the AI model actually
read. The model had no way of knowing which year was meant, so it guessed,
and guessed wrong. The underlying design mistake: the chain only kept the
*derived* context, and threw away the *original* question along the way.
**Fix:** added a second branch to the chain (alongside the context lookup) that
preserves the original input, so the prompt template could use both the
context *and* the actual year the user asked about.

### Bug 3 — a class instead of an actual object

`StrOutputParser` is a *class* — a blueprint. `StrOutputParser()` (with the
parentheses) is an actual *usable object* built from that blueprint. Writing
`StrOutputParser` (no parentheses) into a chain didn't cause an error
immediately — LangChain is flexible enough to accept it at the point the
chain gets built — but the moment the chain was actually *run*, it produced
a confusing low-level error trying to use the blueprint as if it were a
finished object. **Fix:** always remember the parentheses when you mean "an
instance of this class," not just the class name itself.

---

## Library import paths that moved in this installed version of LangChain

Several retrieval-related classes that older tutorials import from
`langchain.retrievers` don't exist there at all in the version installed
for this project (LangChain 1.x reorganized a lot of it into a separate
package called `langchain_classic`). This came up three separate times
while building Milestone 2:

- `MultiQueryRetriever` → actually lives in
  `langchain_classic.retrievers.multi_query`
- `ContextualCompressionRetriever`, `EmbeddingsFilter`, `LLMChainExtractor` →
  `langchain_classic.retrievers` and
  `langchain_classic.retrievers.document_compressors`
- `ParentDocumentRetriever` →
  `langchain_classic.retrievers.parent_document_retriever`
- `InMemoryStore` → works from either `langchain_core.stores` or
  `langchain_classic.storage` (one just re-exports the other)

**The habit this taught:** rather than guessing an import path from an
online tutorial and running the code to see if it fails, it's faster and
more reliable to actually look inside the installed package on disk first
(searching for the class name directly in the installed files) to see
exactly where it lives in *this* specific installed version.

---

## Retrieval quality: why the same question kept getting the wrong year's data

This was the single most repeated and most instructive problem in the whole
project. The same basic question — "what was NVIDIA's total revenue for
fiscal year 2025?" — kept coming back with a *different* year's numbers,
across three completely different retrieval techniques (Multi-Query,
Contextual Compression, and later again inside structured extraction).

### The wrong way to fix this: guessing

It would be easy to just try random settings — "let's set the similarity
threshold to 0.5," "let's bump k up to 10 and see" — and hope one of them
happens to work. That approach was deliberately avoided here in favor of
**measuring the actual numbers the system was producing**, every time.

### What "measuring" actually looks like

The embedding model turns both the question and every document chunk into a
list of numbers (a vector). "Similarity" between the question and a chunk is
just a mathematical comparison between their two vectors (cosine
similarity — a number between -1 and 1, where higher means "more similar").
Instead of guessing whether a chunk was "similar enough," the actual score
was computed directly in Python and printed out, for every candidate chunk,
so the real numbers could be looked at directly.

Doing this revealed two separate, concrete problems:

1. **The similarity filter threshold was set far too low to matter.**
   A setting of `0.5` was meant to filter out irrelevant chunks. But when the
   real scores were printed, every retrieved chunk was actually scoring
   between about 0.68 and 0.85 — nowhere near as low as 0.5. So the filter
   was never removing anything at all; it was pure dead weight.

2. **A completely irrelevant chunk sometimes scored *higher* than the
   correct one.** A totally unrelated chunk about lease payment schedules
   scored 0.85, while the chunk containing NVIDIA's real 2025 revenue figure
   only scored 0.665 — lower than several irrelevant chunks. This is because
   the embedding model mostly notices *surface* similarity ("a table full of
   dollar amounts under year headers") rather than deep topic relevance. A
   similarity filter can only ever throw away low-scoring chunks — it has no
   way to promote a good chunk that happens to score lower than a bad one.
   That's why raising the similarity threshold alone could never have fixed
   this on its own.

**What actually fixed it:** since the real target chunk was scoring just
barely below the cutoff for how many results were being requested (`k`), the
practical fix was simply asking for more results — raising `k` from 8 to 12.
This was confirmed by directly checking: after raising `k`, was the correct
chunk with the real number now actually present among the results? Yes.

### The same underlying problem, showing up again in structured extraction

Later, in the part of the project that extracts structured financial data
(turning a chunk of text into a proper `FiscalYear` + segment breakdown
object), the exact same kind of failure happened again: the total revenue
number came through fine, but the *list of segment revenues* — Data Center,
Gaming, Compute & Networking, etc. — came back completely empty.

**Why:** NVIDIA's annual report has a sentence like *"Our two reportable
segments are 'Compute & Networking' and 'Graphics':"* immediately followed
by the actual dollar figures. The automatic chunking process happened to cut
the text into pieces right around that sentence — so one chunk contains the
sentence introducing the segments, and the *next* chunk contains the actual
numbers. When the real similarity scores were measured for this specific
question and this specific chunking, the chunk holding the actual numbers
turned out to rank **19th** out of the retrievable chunks, and the one
retrieving with the setup at the time only ever looked at the first several
results. The numbers were being cut off from the model entirely, not
because retrieval was "broken," but because it simply wasn't looking far
enough down the ranked list.

**The fix applied:** raise `k` again, this time to 20, so that 19th-ranked
chunk actually makes it into what the model sees.

**Why this fix is a patch, not a real solution — explained plainly:** asking
for more results (`k=20`) only works because, for *this exact wording* of
the question, the right chunk happens to land at position 19, which is close
enough to squeeze in by just asking for more. If someone phrased the
question slightly differently, or the report were formatted slightly
differently, that same chunk might rank 25th, or 40th — completely missed
again, no matter how high `k` is turned up, without also making the model
noticeably slower and more expensive to run (since a bigger `k` means more
text gets sent to the AI model every time). **This is treating a symptom,
not the disease.**

**The actual, structural fix is the Parent Document Retriever technique**
(covered in `02_advanced_rag_retrieval.md`), and here's why it fixes this
specific problem at the root instead of working around it: the real issue is
that the sentence introducing the segment breakdown and the numbers
themselves got separated by an arbitrary chunk-size cutoff. Parent Document
Retriever searches over small chunks (so matching is precise), but whenever
a small chunk matches, it hands back the **entire larger parent chunk** it
came from — meaning the sentence *and* the numbers that follow it would
end up together in the same piece of retrieved text automatically, every
single time, regardless of exact chunk boundaries or how the question is
phrased. Raising `k` is "ask for more haystack and hope the needle is
somewhere in it." Parent Document Retriever is "make sure the needle was
never separated from the part of the haystack right around it in the first
place." One is a workaround for this project's specific current chunking
setup; the other removes the reason the problem happens at all.

---

## Structural and naming mistakes

### A folder was literally named with `.py` in it

A folder got named `tool_calling_and_agent.py` — note that's a **folder**
name, not a file name, and it has a period in it. Python's import system
requires every part of a dotted path (like `src.folder_name.file_name`) to
be a valid plain name — no periods allowed inside a single segment. Having
a period in the folder's name broke this completely. **Fix:** renamed the
folder to `src/tool_calling/`, no periods.

### A one-letter typo in a package name

`langchain_hugginface` (missing a `g`) instead of `langchain_huggingface`.
A completely ordinary typo — Python couldn't find a package by that
misspelled name, so it failed with `ModuleNotFoundError`. Fixed by
correcting the spelling.

### The same class name imported from two different places, silently overwriting each other

In `persist.py`, one line imported a class called `SegmentRevenue` from the
database models file (`src.db.models`) — the version meant to represent an
actual database row. A little further down, *another* line imported a
*different* class, also called `SegmentRevenue`, from the Pydantic schema
file (`src.pydantic_schemas`) — the version meant to represent validated
data extracted from the AI model's answer. In Python, when you import two
things with the same name, the second import silently replaces the first
one — there's no warning. So the code that thought it was building a
database row was actually trying to build the *other* kind of object, which
has different fields, and failed. **Fix:** removed the second, unnecessary
import, keeping only the one actually needed in that file.

### A relationship declared on one side, but not the other

SQLAlchemy lets two related database tables "know about" each other on the
Python side — e.g. a fiscal year object that can list all its segments, and
a segment object that can look up its own fiscal year. Setting this up with
`back_populates` requires declaring it **on both classes**, each one naming
the other. One side (`SegmentRevenue`) was written first, declaring
`back_populates="revenues"` — but nothing called `revenues` existed yet on
the other class (`FiscalYear`). SQLAlchemy checks that this pairing is
consistent as soon as the models are used, and raised an error because half
of the pairing was missing. **Fix:** added the matching `revenues =
relationship(...)` declaration to `FiscalYear`.

### A uniqueness rule that was too strict for what was actually needed

A column meant to hold a segment name (like `"Data Center"`) was marked
`unique=True`. That setting means "this exact value may only ever appear
once, in the entire table" — but the actual goal was for `"Data Center"` to
be able to appear once *per fiscal year* (once for 2024, once for 2025,
etc.), which is a different, more specific kind of uniqueness rule (a
"composite" one, covering the segment name *and* the year together, not the
segment name alone). Left as originally written, inserting a second year's
data would have failed the very first time a segment name repeated. **Fix:**
removed the overly strict rule since nothing yet required it.

---

## A subtlety in how database sessions manage data

### Reading data from an object right after saving it, but after closing the connection

After saving a new row to the database and calling `commit()`, SQLAlchemy
does something that's easy to miss: it marks all the fields on that object
as "stale," so that the *next* time you read one of them, it goes back to
the database to fetch the truly up-to-date value (in case something else
changed it). That's normally invisible and helpful. The problem is that if
you *close* the connection (`session.close()`) before reading a field like
`.id`, there's no connection left for it to refresh from, and reading it
throws an error. **Fix:** read the value you need (like the new row's `id`)
into a plain variable *immediately* after `commit()`, while the connection
is still open — then it's safe to close the connection afterward, since
you're no longer relying on the object itself.

---

## Problems that came from outside this project entirely

### An AI model this project depended on was removed by the provider, with no warning

At one point, every single chain in this project suddenly started failing
with an error saying the model `llama-3.1-8b-instant` didn't exist or
wasn't accessible anymore. Nothing in the project's own code had changed —
Groq (the company hosting the free AI model being used) had simply removed
that entire family of models from what this account could access, in the
time between one working session and the next. This is a reminder that
depending on a free, externally-hosted model means depending on decisions
made by someone else's business, outside anyone's control.

**How this was diagnosed properly, instead of guessing a replacement model
name:** asked Groq's own API directly for the current list of models this
account can actually use, rather than assuming a name from memory or an old
tutorial. **Fix:** switched to `openai/gpt-oss-120b` — additionally
double-checked by noting that the separate, already-deployed Sharan BOT
project independently made the same choice for its own production use,
which added confidence it was a reasonable, working option.

### A follow-on confusion caused by the model swap (that turned out not to be about the model at all)

Right after switching models, one call started failing with an error saying
the model refused to use a required tool. At first glance this looked like
a problem with the *new* model being less capable. Looking closer, the real
story was: the retrieval step genuinely hadn't found NVIDIA's 2025 data for
that particular question (the same `k`-too-small issue described above, just
appearing again in the extraction code), and the new model was actually
behaving *correctly* — it tried to explain in plain language that it
couldn't find the data, rather than making up a fake answer just to satisfy
the requirement of returning a structured result. The real fix was fixing
retrieval (`k`), not "fixing" the model.

---

## The lessons that generalize beyond any single bug here

- **When something isn't working, check each stage on its own**, rather than
  staring at the whole pipeline and guessing. Several bugs in this project
  only became clear once a single step was tested in isolation — e.g.,
  proving that an AI model correctly *chose* the right tool to call, even
  though the *final answer* using that tool's result was wrong, showed the
  problem was in one specific spot, not everywhere.
- **Measure the real numbers instead of guessing when something involves a
  score or a threshold.** Every single retrieval problem in this project got
  solved by actually printing out real similarity scores and looking at
  them, never by adjusting a number and hoping it worked better.
- **Copy-pasting a file, or importing the same name from two different
  places, is a trap.** Every time this project did either of those, it
  quietly brought back a bug that had already been fixed once, or silently
  used the wrong version of something with the same name.
