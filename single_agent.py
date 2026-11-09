from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# A tool is an ordinary function with typed inputs and a clear description.
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


tools = [multiply]

model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
).bind_tools(tools)


# This is the single agent node.
def agent(state: MessagesState):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant. "
                    "Use the multiply tool for multiplication."
                )
            ),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 17 times 23?")]},
        config={"recursion_limit": 10},
    )

    for message in result["messages"]:
        message.pretty_print()
