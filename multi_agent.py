from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    hypothesis: str
    narrative: str
    extraction: str
    likelihood_hypothesis: str


model_choice = "qwen3:8b"
reasoning = False
model = ChatOllama(model=model_choice, temperature=0, reasoning=reasoning)


def evidence_extraction(state: State):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a detective. Generate an argument for given hypothesis using information from the supplied narrative. Define your evidence for each individual argument."
                )
            ),
            HumanMessage(
                content=(
                    f"Hypothesis: {state['hypothesis']}\n\n"
                    f"Narrative: {state['narrative']}"
                )
            ),
        ]
    )

    return {"extraction": response.content}


def diagnosticity(state: State):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are an intelligence analyst. Examine the given piece of evidence and determine its diagnosticity for the given hypothesis."
                )
            ),
            HumanMessage(
                content=(
                    f"Hypothesis: {state['hypothesis']}\n\n"
                    f"Evidence: {state['extraction']}"
                )
            ),
        ]
    )

    return {"diagnostic_score": response.content}


def analysis(state: State):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a detective attempting to solve a case. Based on the given evidence, determine the likelihood that the hypothesis is correct."
                )
            ),
            HumanMessage(
                content=(
                    f"Hypothesis:\n{state['hypothesis']}\n\n"
                    f"Arguments and Evidence:\n{state['extraction']}"
                )
            ),
        ]
    )

    return {"likelihood_hypothesis": response.content}


builder = StateGraph(State)

builder.add_node("extraction", evidence_extraction)
builder.add_node("diagnosticity", diagnosticity)
builder.add_node("analysis", analysis)

builder.add_edge(START, "extraction")
builder.add_edge("extraction", "diagnosticity")
builder.add_edge("diagnosticity", "analysis")
builder.add_edge("analysis", END)

graph = builder.compile()

result = graph.invoke(
    {
        "hypothesis": "Why do solar panels produce less power when hot?",
        "narrative": "",
        "extraction": "",
        "likelihood_hypothesis": "",
    }
)

print(result["likelihood_hypothesis"])
