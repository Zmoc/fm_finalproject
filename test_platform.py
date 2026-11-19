import time

from agents.multi_agent import multi_agent_graph, multi_agent_invoke
from agents.single_agent import single_agent_graph, single_agent_invoke
from eval_helpers.eval_suite import eval, generate_test_list, metadata_extraction

if __name__ == "__main__":
    test_data_path = "data/test_data/short_run.csv"
    test_list = generate_test_list(test_data_path)
    model_choice = "qwen3:8b"
    reasoning = True
    temperature = 0
    timeout = 60.0
    eval_path = "data/eval_results/eval_results.csv"
    timestamp = time.time()

    single_agent_run = True
    multi_agent_run = False
    agent_type = ""

    if single_agent_run == True:
        single_graph = single_agent_graph(model_choice, temperature, reasoning, timeout)
        single_graph.step_timeout = timeout
        agent_type = "single"

    if multi_agent_run == True:
        multi_graph = multi_agent_graph(model_choice, temperature, reasoning, timeout)
        multi_graph.step_timeout = timeout
        agent_type = "multi"

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
            metadata_extraction(index, label, results, generated_answer, test_result)
        )

    eval(
        eval_list,
        timestamp,
        model_choice,
        reasoning,
        agent_type,
        test_data_path,
        eval_path,
    )
