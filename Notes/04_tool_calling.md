# Tool & Function Calling

## The mechanism

**`@tool`** (from `langchain_core.tools`) turns a plain Python function into
something an LLM can choose to invoke. The function's **docstring is not
documentation for humans here — it's what the LLM reads to decide when to
use the tool.** A vague docstring ("does math") gives the model little to go
on; a specific one ("Use this to compute percentage growth between two
numeric values") is what actually drives correct tool selection.

**`model.bind_tools([tool1, tool2, ...])`** returns a new Runnable — same
`.invoke()` interface — whose responses may include tool calls. Calling
`.invoke(messages)` returns an `AIMessage` with a `.tool_calls` list
(empty if the model decided no tool was needed for this input).

**The manual execution loop** (no LangGraph, per this project's scope):
```python
ai_message = model_with_tools.invoke(messages)
messages.append(ai_message)

if not ai_message.tool_calls:
    return ai_message.content   # model answered directly

for call in ai_message.tool_calls:
    tool_fn = TOOLS_BY_NAME[call["name"]]
    result = tool_fn.invoke(call["args"])
    messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

final_message = model_with_tools.invoke(messages)   # second call: model sees
return final_message.content                         # the real tool result(s)
```
Two model calls per tool-using question: one to decide/call the tool, one to
produce the final answer once the tool's real result is in the message list.

## Structural gotcha hit while building this

The tools file's containing folder was originally named
`tool_calling_and_agent.py` — a **directory** with `.py` in its name. This
breaks Python's import system: each segment of a dotted import path
(`src.tool_calling_and_agent.py.tools`) must be a valid identifier, and a
literal `.` inside a folder name isn't one. Renamed to `src/tool_calling/`.

## The real finding: small free models are unreliable at tool-result synthesis

Verified in three stages, each isolating a different part of the pipeline:

1. **Tool selection and argument extraction work correctly** on the free 8B
   model (`llama-3.1-8b-instant`) — confirmed by inspecting the raw
   `ai_message.tool_calls` directly: `{'name': 'get_segment_revenue', 'args':
   {'fiscal_year': '2025', 'segment_name': 'Gaming'}}` — exactly right.

2. **The second call (final answer synthesis) is where it breaks.** With the
   correct tool result already in the message list as a `ToolMessage`, the
   8B model's final answers were empty, vague ("This is the percentage
   growth between the two revenue values" — no actual number), or in one
   case **hallucinated a fake tool-call syntax and a fabricated date**
   (`2023-01-01`) instead of reading the real result already in front of it.

3. **A forceful system-message patch** ("state the exact result from the
   tool output above, don't describe it") partially helped one case but made
   another *worse* — confirming this is a genuine small-model capability
   limit, not something reliably fixable by prompting alone.

**Fix: swap to a larger free model for this step specifically —
`llama-3.3-70b-versatile`** (still free on Groq). All four test cases passed
immediately, including correctly declining to call any tool for an unrelated
question ("What is the capital of France?"). Kept as a *separate* model
instance in `src/tool_calling/agent.py` rather than changing the shared
`model` in `src/chain.py`, to avoid touching already-verified Milestone 1-3
code that didn't need this capability.

**General lesson:** tool *selection* and tool *result synthesis* are
different capabilities, and a model can be reliably good at one while being
unreliable at the other. Worth testing both independently when diagnosing a
tool-calling pipeline that "isn't working" — the failure might not be where
it looks.
