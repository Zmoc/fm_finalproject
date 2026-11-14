import csv
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

test_list = []

with open("data/test_data/debug_run.csv", encoding="utf-8") as eval_data:
    reader = csv.DictReader(eval_data, delimiter=",")

    for index, line in enumerate(reader):
        narrative = line["narrative"]
        question = line["question"]
        choices = line["choices"]
        correct_answer = line["answer_choice"]

        answer_prompt = f"ANSWER: {correct_answer}"
        prompt_dict = {
            "index": index,
            "hypothesis": f"The murderer is {choices[0]}",
            "narrative": narrative,
            "correct_answer": answer_prompt,
        }
        test_list.append(prompt_dict)


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

for story in test_list:
    index = story["index"]
    hypothesis = story["hypothesis"]
    narrative = story["narrative"]
    label = story["correct_answer"]
    print(f"Running Story {index}")
    result = graph.invoke(
        {
            "hypothesis": hypothesis,
            "narrative": narrative,
            "extraction": "",
            "likelihood_hypothesis": "",
        }
    )

    print(result["likelihood_hypothesis"])
