import csv
from pathlib import Path


def accuracy_score(eval_list):
    pass_count = 0
    num_tests = len(eval_list)
    for test in eval_list:
        if test["test_result"] == True:
            pass_count += 1

    accuracy = pass_count / num_tests
    return accuracy


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
