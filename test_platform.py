import time

from agents.multi_agent import multi_agent_graph, multi_agent_invoke
from agents.single_agent import single_agent_graph, single_agent_invoke
from eval_helpers.eval_suite import eval, generate_test_list, metadata_extraction

if __name__ == "__main__":
    test_data_path = "data/test_data/debug_run.csv"
    test_list = generate_test_list(test_data_path)
    model_list = [
        "qwen3:4b",
        "mistral:7b",
        "llama3.1:8b",
        "qwen3:8b",
        "gemma3:12b",
        "qwen3:14b",
    ]
    seed = 42
    reasoning = False
    eval_path = "data/eval_results/eval_results.csv"

    single_agent_run = True
    multi_agent_run = False
    agent_type = ""

    for model in model_list:
        timestamp = time.time()
        if single_agent_run == True:
            single_graph = single_agent_graph(model, reasoning, seed)
            agent_type = "single"

        if multi_agent_run == True:
            multi_graph = multi_agent_graph(model, reasoning, seed)
            agent_type = "multi"

        print(f"Running {agent_type} agent using {model}")

        eval_list = []
        for story in test_list:
            index = story["index"]
            narrative = story["narrative"]
            label = story["correct_answer"]

            if single_agent_run == True:
                generated_answer, results = single_agent_invoke(
                    single_graph, story, index, narrative
                )

            if multi_agent_run == True:
                generated_answer, results = multi_agent_invoke(
                    multi_graph, story, index, narrative
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
