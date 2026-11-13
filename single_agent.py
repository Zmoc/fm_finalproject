import csv
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

test_list = []


def eval(eval_list, timestamp, model, thinking, model_type, eval_path):
    accuracy = 0.0
    with open(f"eval_results/model_eval_{int(timestamp)}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=eval_list[0].keys())
        writer.writeheader()
        writer.writerows(eval_list)

    accuracy = accuracy_score(eval_list)

    eval_metadata = [
        {
            "timestamp": int(timestamp),
            "model": model,
            "model_type": model_type,
            "thinking": thinking,
            "accuracy": accuracy,
        }
    ]

    if Path(eval_path).is_file():
        with open("eval_results/eval_results.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=eval_metadata[0].keys())
            writer.writerows(eval_metadata)
    else:
        with open(eval_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=eval_metadata[0].keys())
            writer.writeheader()
            writer.writerows(eval_metadata)


def accuracy_score(eval_list):
    pass_count = 0
    num_tests = len(eval_list)
    for test in eval_list:
        if test["test_result"] == True:
            pass_count += 1

    accuracy = pass_count / num_tests
    return accuracy


with open("data/debug_run.csv", encoding="utf-8") as eval_data:
    reader = csv.DictReader(eval_data, delimiter=",")
    story_count = 0
    for line in reader:
        narrative = line["narrative"]
        question = line["question"]
        choices = line["choices"]
        correct_answer = line["answer_choice"]

        prompt = f"{narrative}\n\n{question}\n\nYour choices are: {choices}"
        answer_prompt = f"ANSWER: {correct_answer}"
        prompt_dict = {
            "index": story_count,
            "story_context": prompt,
            "correct_answer": answer_prompt,
        }
        test_list.append(prompt_dict)
        story_count += 1


# A tool is an ordinary function with typed inputs and a clear description.
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# tools = [multiply]

model_choice = "qwen3:8b"
reasoning = False
eval_path = "eval_results/eval_results.csv"

model = ChatOllama(
    model=model_choice,
    temperature=0,
    reasoning=reasoning,
    # client_kwargs={"timeout": 60 * 3},
)  # .bind_tools(tools)


# This is the single agent node.
def agent(state: MessagesState):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the following multiple choice question.\n"
                    "The last line of your response should be of the following format: 'ANSWER: [NAME]' (without quotes) where [NAME] is the chosen suspect.\n"
                    "Think step by step before answering.\n"
                    "Clearly specify the evidence that you are using to justify your answer.\n"
                )
            ),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
# builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
# builder.add_conditional_edges("agent", tools_condition)
# builder.add_edge("tools", "agent")

graph = builder.compile()


if __name__ == "__main__":
    NANOSECOND_CONVERSION = 1000000000
    timestamp = time.time()
    eval_list = []
    for story in test_list:
        index = story["index"]
        test_input = story["story_context"]
        label = story["correct_answer"]
        print(f"Running Story {index}")

        result = graph.invoke(
            {"messages": [HumanMessage(content=test_input)]},
            config={"recursion_limit": 10},
        )

        response = result["messages"][1]
        content = response.content
        generated_answer = content.split("\n")[-1]
        test_result = label == generated_answer

        done_reason = response.response_metadata.get("done_reason")
        total_duration = response.response_metadata.get("total_duration")
        load_duration = response.response_metadata.get("load_duration")
        prompt_eval_count = response.response_metadata.get("prompt_eval_count")
        prompt_eval_duration = response.response_metadata.get("prompt_eval_duration")
        eval_count = response.response_metadata.get("eval_count")
        eval_duration = response.response_metadata.get("eval_duration")

        metadata = {
            "index": index,
            "label": label,
            "generated_answer": generated_answer,
            "test_result": test_result,
            "done_reason": done_reason,
            "total_duration": total_duration / NANOSECOND_CONVERSION,
            "load_duration": load_duration / NANOSECOND_CONVERSION,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration / NANOSECOND_CONVERSION,
            "eval_count": eval_count,
            "eval_duration": eval_duration / NANOSECOND_CONVERSION,
        }

        print(f"Completed: {metadata}")
        eval_list.append(metadata)
    eval(eval_list, timestamp, model_choice, reasoning, "single", eval_path)
