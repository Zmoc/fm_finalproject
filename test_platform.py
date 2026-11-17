import time

from eval_suite import eval, generate_test_list, metadata_extraction
from multi_agent import multi_agent_graph, multi_agent_invoke
from single_agent import single_agent_graph, single_agent_invoke

if __name__ == "__main__":
    test_list = generate_test_list()
    model_choice = "qwen3:8b"
    reasoning = False
    eval_path = "data/eval_results/eval_results.csv"
    timestamp = time.time()

    single_agent_run = False
    multi_agent_run = True

    if single_agent_run == True:
        single_graph = single_agent_graph(model_choice, 0, reasoning)

    if multi_agent_run == True:
        multi_graph = multi_agent_graph(model_choice, 0, reasoning)

    for story in test_list:
        index = story["index"]
        narrative = story["narrative"]
        label = story["correct_answer"]
        results = []

        if single_agent_run == True:
            generated_answer = single_agent_invoke(
                single_graph, story, index, narrative, results
            )

        if multi_agent_run == True:
            generated_answer = multi_agent_invoke(
                multi_graph, story, index, narrative, results
            )

        test_result = label == generated_answer
        eval_list = metadata_extraction(
            index, label, results, generated_answer, test_result
        )
    eval(eval_list, timestamp, model_choice, reasoning, "single", eval_path)
