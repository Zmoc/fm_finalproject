import csv
import re
from pathlib import Path


def scoring(eval_list):
    pass_count = 0
    indecisive_count = 0
    num_tests = len(eval_list)
    for test in eval_list:
        if test["generated_answer"] == "Indecisive":
            indecisive_count += 1
        if test["test_result"] == True:
            pass_count += 1

    accuracy = pass_count / num_tests
    indecisive_rate = indecisive_count / num_tests
    return accuracy, indecisive_rate


def eval(eval_list, timestamp, model, thinking, model_type, test_data_path, eval_path):
    accuracy = 0.0
    with open(
        f"data/eval_results/model_eval_{int(timestamp)}.csv", "w", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=eval_list[0].keys())
        writer.writeheader()
        writer.writerows(eval_list)

    accuracy, indecisive_rate = scoring(eval_list)

    eval_metadata = [
        {
            "timestamp": int(timestamp),
            "test_dataset": test_data_path,
            "model": model,
            "model_type": model_type,
            "thinking": thinking,
            "accuracy": accuracy,
            "indecisive_rate": indecisive_rate,
        }
    ]

    if Path(eval_path).is_file():
        with open(eval_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=eval_metadata[0].keys())
            writer.writerows(eval_metadata)
    else:
        with open(eval_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=eval_metadata[0].keys())
            writer.writeheader()
            writer.writerows(eval_metadata)


def capture_metadata(agent_name, response):
    return {
        "agent": agent_name,
        "message_id": response.id,
        "response_metadata": dict(response.response_metadata),
        "usage_metadata": (
            dict(response.usage_metadata) if response.usage_metadata is not None else {}
        ),
    }


def metadata_extraction(index, label, results, generated_answer, test_result):
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

    print(f"Completed {index}")
    return metadata


def generate_test_list(input_file):
    test_list = []
    with open(input_file, encoding="utf-8") as eval_data:
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
