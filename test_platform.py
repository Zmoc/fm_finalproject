import csv
import re
import time

from eval_suite import eval
from single_agent import single_agent_graph

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


model_choice = "qwen3:8b"
reasoning = False
eval_path = "data/eval_results/eval_results.csv"
single_graph = single_agent_graph(model_choice, 0, reasoning)


if __name__ == "__main__":
    NANOSECOND_CONVERSION = 1000000000
    timestamp = time.time()
    eval_list = []

    for story in test_list:
        index = story["index"]
        label = story["correct_answer"]
        question = story["question"]
        narrative = story["narrative"]
        choices = story["choices"]
        print(f"Running Story {index}")

        result = single_graph.invoke(
            {
                "question": question,
                "narrative": narrative,
                "choices": choices,
            }
        )

        response = result["messages"][1]
        content = response.content
        generated_answer = content.split("\n")[-1]
        test_result = label == generated_answer

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
