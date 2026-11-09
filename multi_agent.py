from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str
    research_notes: str
    answer: str


model = ChatOllama(model="qwen3:8b", temperature=0)


def researcher(state: State):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a researcher. Produce concise factual notes "
                    "that help answer the question. Flag uncertainty."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )

    return {"research_notes": response.content}


def writer(state: State):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a writer. Answer the user's question using the "
                    "provided notes. Preserve any uncertainty."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Research notes:\n{state['research_notes']}"
                )
            ),
        ]
    )

    return {"answer": response.content}


builder = StateGraph(State)

builder.add_node("researcher", researcher)
builder.add_node("writer", writer)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)

graph = builder.compile()

result = graph.invoke(
    {
        "question": "Why do solar panels produce less power when hot?",
        "research_notes": "",
        "answer": "",
    }
)

print(result["answer"])
