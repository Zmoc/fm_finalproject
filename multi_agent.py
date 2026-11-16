import csv
import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

test_list = []

with open("data/test_data/debug_run.csv", encoding="utf-8") as eval_data:
    reader = csv.DictReader(eval_data, delimiter=",")

    for index, line in enumerate(reader):
        narrative = line["narrative"]
        question = line["question"]
        choices = re.findall(r"(\w+)", line["choices"])
        hypothesis_a = choices[0]
        hypothesis_b = choices[1]

        correct_answer = line["answer_choice"]

        answer_prompt = f"ANSWER: {correct_answer}"
        prompt_dict = {
            "index": index,
            "hypothesis": [
                f"The murderer is {hypothesis_a}",
                f"The murderer is {hypothesis_b}",
            ],
            "narrative": narrative,
            "correct_answer": answer_prompt,
        }
        test_list.append(prompt_dict)


class State(TypedDict):
    hypothesis: str
    narrative: str
    extraction: str
    diagnostic_score: str
    likelihood_hypothesis: str
    model_calls: Annotated[list[dict[str, Any]], operator.add]


def capture_metadata(agent_name, response):
    return {
        "agent": agent_name,
        "message_id": response.id,
        "response_metadata": dict(response.response_metadata),
        "usage_metadata": (
            dict(response.usage_metadata) if response.usage_metadata is not None else {}
        ),
    }


model_choice = "qwen3:8b"
reasoning = False
model = ChatOllama(model=model_choice, temperature=0, reasoning=reasoning)


def evidence_extraction(state: State):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a detective using the Analysis of Competing Hypotheses method. Generate individual arguments for the given hypothesis using information from the supplied narrative."
                    "Specify the evidence used for each individual argument.\n\n"
                    "Only generate arguments and evidence for the given hypothesis. Do not generate for the opposite hypothesis.\n\n"
                    "Use the format:\n\n"
                    "Argument A: ...\n\n"
                    "Evidence A1: ...\n\n"
                    "Evidence A2: ...\n\n"
                    "Argument B: ...\n\n"
                    "Evidence B1: ...\n\n"
                    "Evidence B2: ...\n\n"
                    "..."
                    "Argument Z: ...\n\n"
                    "Evidence Z1: ...\n\n"
                    "Evidence Z2: ...\n\n"
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
    return {
        "extraction": response.content,
        "model_calls": [capture_metadata("evidence_extraction", response)],
    }


def diagnosticity(state: State):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are an intelligence analyst using the Analysis of Competing Hypotheses method. Examine the given arguments and supporting evidence and determine how useful each is in determining the likelihood of the given hypothesis."
                    "Use the format, where [ARGUMENT] and [EVIDENCE] are the respective items and [SCORE] is a value from 1-5, with 1 being very low usefulness, 2 being low, 3 being moderate, 4 being high and 5 being very high:\n\n"
                    "[ARGUMENT]: [SCORE]\n\n"
                    "[EVIDENCE]: [SCORE]\n\n"
                    "...\n\n"
                    "[EVIDENCE]: [SCORE]\n\n"
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
    return {
        "diagnostic_score": response.content,
        "model_calls": [capture_metadata("diagnosticity", response)],
    }


def analysis(state: State):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a detective attempting to solve a case using the Analysis of Competing Hypotheses method."
                    "Based on the given evidence, determine the likelihood that the hypothesis is correct."
                    "Give the likelihood as a value on a scale of 1-5 with 1 being Very Unlikely, 2 as Unlikely, 3 as Likely, 4 as Highly Likely, and 5 as Very Highly Likely."
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

    return {
        "likelihood_hypothesis": response.content,
        "model_calls": [capture_metadata("analysis", response)],
    }


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
    print(f"Running Story {index}, Hypothesis: {hypothesis}")
    result = graph.invoke(
        {
            "hypothesis": hypothesis,
            "narrative": narrative,
            "extraction": "",
            "diagnostic_score": "",
            "likelihood_hypothesis": "",
            "model_calls": [],
        }
    )

    print(result["model_calls"])
