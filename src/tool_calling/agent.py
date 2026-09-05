from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq

from src.tool_calling.tools import TOOLS

load_dotenv()

# A larger free Groq model than src.chain's default (8B) — needed here because
# tool-result synthesis (correctly stating a ToolMessage's value in the final
# answer) is unreliable on the 8B model, verified directly during testing.
tool_model = ChatGroq(model="llama-3.3-70b-versatile")
model_with_tools = tool_model.bind_tools(TOOLS)

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def run_with_tools(question: str) -> str:
    messages = [HumanMessage(question)]

    ai_message = model_with_tools.invoke(messages)
    messages.append(ai_message)

    # No tool calls means the model answered directly, no lookup needed.
    if not ai_message.tool_calls:
        return ai_message.content

    for call in ai_message.tool_calls:
        tool_fn = TOOLS_BY_NAME[call["name"]]
        result = tool_fn.invoke(call["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Second call: model sees its own tool call(s) plus the real results,
    # and produces a final natural-language answer grounded in them.
    final_message = model_with_tools.invoke(messages)
    return final_message.content


if __name__ == "__main__":
    print(run_with_tools("What was NVIDIA's Gaming segment revenue in fiscal year 2025?"))
    print()
    print(run_with_tools("If revenue grew from 60922 to 130497, what's the percentage growth?"))
    print()
    print(run_with_tools("What is today's date?"))
    print()
    print(run_with_tools("What is the capital of France?"))
