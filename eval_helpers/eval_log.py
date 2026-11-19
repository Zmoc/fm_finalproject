import csv
from pathlib import Path


def log_exception(
    index, timestamp, exception_message, model, temperature, reasoning, content
):
    log_path = f"data/logs/exception_log_{timestamp}.csv"

    exception_metadata = {
        "timestamp": timestamp,
        "index": index,
        "model": model,
        "temperature": temperature,
        "reasoning": reasoning,
        "exception_message": exception_message,
        "content": content,
    }
    print(exception_metadata)

    if Path(log_path).is_file():
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=exception_metadata.keys())
            writer.writerows(exception_metadata)
    else:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=exception_metadata.keys())
            writer.writeheader()
            writer.writerows(exception_metadata)
