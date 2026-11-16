import ast
import csv
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from eval_suite import eval

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
                    "You are an intelligence analyst using the Analysis of Competing Hypotheses method. Examine the given arguments and supporting evidence and determine how useful each is in determining the likelihood of the given hypothesis.\n\n"
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
                    "You are a detective attempting to solve a case using the Analysis of Competing Hypotheses method.\n\n"
                    "Based on the given evidence, determine the likelihood that the hypothesis is correct.\n\n"
                    "Give the likelihood as a value on a scale of 1-5 with 1 being Very Unlikely, 2 as Unlikely, 3 as Likely, 4 as Highly Likely, and 5 as Very Highly Likely.\n\n"
                    "Output your final answer as a dictionary, {'suspect':SUSPECT, 'likelihood':LIKELIHOOD}, where SUSPECT is the suspect and LIKELIHOOD is the likelihood score value.\n\n"
                    "Output only the dictionary, do not include any other text or explanation.\n\n"
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


def graph():
    builder = StateGraph(State)
    builder.add_node("extraction", evidence_extraction)
    builder.add_node("diagnosticity", diagnosticity)
    builder.add_node("analysis", analysis)
    builder.add_edge(START, "extraction")
    builder.add_edge("extraction", "diagnosticity")
    builder.add_edge("diagnosticity", "analysis")
    builder.add_edge("analysis", END)

    return builder.compile()


graph = graph()
NANOSECOND_CONVERSION = 1000000000
timestamp = time.time()
eval_path = "data/eval_results/eval_results.csv"

story_count = len(test_list)
positive_count = 0.0
eval_list = []
for story in test_list:
    index = story["index"]
    hypothesisA = story["hypothesis"][0]
    hypothesisB = story["hypothesis"][1]
    narrative = story["narrative"]
    label = story["correct_answer"].split(" ")[-1]
    result_list = []

    print(f"Running Story {index}, Hypothesis: {hypothesisA}")
    resultA = graph.invoke(
        {
            "hypothesis": hypothesisA,
            "narrative": narrative,
            "extraction": "",
            "diagnostic_score": "",
            "likelihood_hypothesis": "",
            "model_calls": [],
        }
    )
    evalA = ast.literal_eval(resultA["likelihood_hypothesis"])

    print(f"Running Story {index}, Hypothesis: {hypothesisB}")
    resultB = graph.invoke(
        {
            "hypothesis": hypothesisB,
            "narrative": narrative,
            "extraction": "",
            "diagnostic_score": "",
            "likelihood_hypothesis": "",
            "model_calls": [],
        }
    )
    evalB = ast.literal_eval(resultB["likelihood_hypothesis"])

    if evalA["likelihood"] > evalB["likelihood"]:
        generated_answer = evalA["suspect"]
    elif evalA["likelihood"] < evalB["likelihood"]:
        generated_answer = evalB["suspect"]
    else:
        generated_answer = "Indecisive"

    test_result = generated_answer == label
    if test_result == True:
        positive_count += 1

    total_duration = 0.0
    load_duration = 0.0
    prompt_eval_count = 0.0
    prompt_eval_duration = 0.0
    eval_count = 0.0
    eval_duration = 0.0

    for call in resultA["model_calls"]:
        prefix = call["response_metadata"]
        print(prefix)

        total_duration += prefix["total_duration"] / NANOSECOND_CONVERSION
        load_duration += prefix["load_duration"] / NANOSECOND_CONVERSION
        prompt_eval_count += prefix["prompt_eval_count"]
        prompt_eval_duration += prefix["prompt_eval_duration"] / NANOSECOND_CONVERSION
        eval_count += prefix["eval_count"]
        eval_duration += prefix["eval_duration"] / NANOSECOND_CONVERSION

    metadata = {
        "index": index,
        "label": label,
        "generated_answer": generated_answer,
        "test_result": test_result,
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
    }
    print(f"Completed: {metadata}")
    eval_list.append(metadata)
eval(eval_list, timestamp, model_choice, reasoning, "multi", eval_path)
