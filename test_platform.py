import time

from agents.agents import (
    multi_agent_graph,
    multi_agent_invoke,
    single_agent_graph,
    single_agent_invoke,
)
from eval_helpers.eval_suite import eval, generate_test_list, metadata_extraction

if __name__ == "__main__":
    test_data_path = "data/test_data/debug_run.csv"
    test_list = generate_test_list(test_data_path)
    """    model_list = [
        "mistral:7b",
        "llama3.1:8b",
        "qwen3:8b",
        "gemma3:12b",
        "qwen3:14b",
    ]"""
    model_list = ["qwen3:8b"]
    seed = 42
    reasoning = False
    eval_path = "data/eval_results/eval_results.csv"

    single_agent_run = True
    multi_agent_run = False
    agent_type = ""

    for model in model_list:
        kwargs = {"model": model, "reasoning": reasoning, "seed": seed}
        timestamp = time.time()
        if single_agent_run == True:
            single_graph = single_agent_graph(**kwargs)
            agent_type = "single"

        if multi_agent_run == True:
            multi_graph = multi_agent_graph(**kwargs)
            agent_type = "multi"

        print(f"Running {agent_type} agent using {model}")

        eval_list = []
        for story in test_list:
            index = story["index"]
            label = story["correct_answer"]

            if single_agent_run == True:
                generated_answer, results = single_agent_invoke(
                    single_graph, story, index
                )

            if multi_agent_run == True:
                generated_answer, results = multi_agent_invoke(
                    multi_graph, story, index
                )

            test_result = label == generated_answer
            eval_list.append(
                metadata_extraction(
                    index, label, results, generated_answer, test_result
                )
            )

        eval(
            eval_list,
            timestamp,
            model,
            reasoning,
            agent_type,
            test_data_path,
            eval_path,
        )
