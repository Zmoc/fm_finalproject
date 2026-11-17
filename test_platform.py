import csv
import re
import time

from eval_suite import eval
from single_agent import single_agent_graph


def metadata_extraction(index, label, results, generated_answer, test_result):
    eval_list = []
    NANOSECOND_CONVERSION = 1000000000

    total_duration = 0.0
    load_duration = 0.0
    prompt_eval_count = 0.0
    prompt_eval_duration = 0.0
    eval_count = 0.0
    eval_duration = 0.0

    for result in results:
        for call in result["model_calls"]:
            prefix = call["response_metadata"]
            total_duration += prefix["total_duration"] / NANOSECOND_CONVERSION
            load_duration += prefix["load_duration"] / NANOSECOND_CONVERSION
            prompt_eval_count += prefix["prompt_eval_count"]
            prompt_eval_duration += (
                prefix["prompt_eval_duration"] / NANOSECOND_CONVERSION
            )
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
    return eval_list


def generate_test_list():
    test_list = []
    with open("data/test_data/initial_run.csv", encoding="utf-8") as eval_data:
        reader = csv.DictReader(eval_data, delimiter=",")

        for index, line in enumerate(reader):
            narrative = line["narrative"]
            question = line["question"]
            correct_answer = line["answer_choice"]
            choices = re.findall(r"(\w+)", line["choices"])

            prompt_dict = {
                "index": index,
                "narrative": narrative,
                "hypothesis": [
                    f"The murderer is {choices[0]}",
                    f"The murderer is {choices[1]}",
                ],
                "choices": choices,
                "question": question,
                "correct_answer": correct_answer,
            }
            test_list.append(prompt_dict)
    return test_list


def single_agent_invoke(single_graph, story, index, narrative, results):
    question = story["question"]
    choices = story["choices"]

    print(f"Running Story {index}")

    result = single_graph.invoke(
        {
            "question": question,
            "narrative": narrative,
            "choices": choices,
            "messages": "",
            "model_calls": [],
        }
    )
    results.append(result)

    response = result["messages"][0]
    generated_answer = response.content.split("\n")[0]
    return generated_answer


if __name__ == "__main__":
    test_list = generate_test_list()
    model_choice = "qwen3:8b"
    reasoning = False
    eval_path = "data/eval_results/eval_results.csv"
    timestamp = time.time()

    single_graph = single_agent_graph(model_choice, 0, reasoning)

    for story in test_list:
        index = story["index"]
        narrative = story["narrative"]
        label = story["correct_answer"]
        results = []
        generated_answer = single_agent_invoke(
            single_graph, story, index, narrative, results
        )
        test_result = label == generated_answer
        eval_list = metadata_extraction(
            index, label, results, generated_answer, test_result
        )
    eval(eval_list, timestamp, model_choice, reasoning, "single", eval_path)
