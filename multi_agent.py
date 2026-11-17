import ast
import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


def multi_agent_graph(model, temperature, reasoning):
    class State(TypedDict):
        hypothesis: str
        narrative: str
        extraction: str
        diagnostic_score: str
        likelihood_hypothesis: str
        model_calls: Annotated[list[dict[str, Any]], operator.add]

    def capture_metadata(agent_name, response):
        return {
            "agent": agent_name,
            "message_id": response.id,
            "response_metadata": dict(response.response_metadata),
            "usage_metadata": (
                dict(response.usage_metadata)
                if response.usage_metadata is not None
                else {}
            ),
        }

    model = ChatOllama(model=model, temperature=temperature, reasoning=reasoning)

    def evidence_extraction(state: State):
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        """
                        You are a detective using the Analysis of Competing Hypotheses method.
                        Make a list of significant evidence and arguments for and against the given hypothesis.

                        The output should be in JSON format where each argument is listed with its supporting evidence.
                        Only output the JSON file. Do not output any other text. Use the following format:

                        {
                            "arguments_for": {
                                "argument_id": {
                                    "argument": "argument_text",
                                    "evidence": [
                                        {
                                            "evidence_id": "evidence_text",
                                            ...
                                        }
                                    ]
                                }
                            },
                            "arguments_against": {
                                "argument_id": {
                                    "argument": "argument_text",
                                    "evidence": [
                                        {
                                            "evidence_id": "evidence_text",
                                            ...
                                        }
                                    ]
                                }
                            }
                        }
                        """
                    )
                ),
                HumanMessage(
                    content=(
                        f"Hypothesis: {state['hypothesis']}\n\n"
                        f"Narrative: {state['narrative']}"
                    )
                ),
            ]
        )

        return {
            "extraction": response.content,
            "model_calls": [capture_metadata("evidence_extraction", response)],
        }

    def diagnosticity(state: State):
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        """
                        You are an intelligence analyst using the Analysis of Competing Hypotheses method.
                        Examine the given arguments and supporting evidence and determine the diagnosticity of each.
                        Diagnosticity refers to how useful the information is in determining the likelihood of the stated hypothesis.

                        Assign a score to each argument and individual piece of evidence from 1-5 where 1 is least diagnostic and 5 is the most diagnostic.
                        A piece of evidence that is likely to be seen regardless of the hypothesis being True or False has a diagnostic value of 1.
                        A piece of evidence that can only appear if the hypothesis is exclusively True or False (but not both) has a diagnostic value of 5.

                        Output your scores in JSON format where each argument and supporting evidence is listed with their scores.
                        Only output the JSON file. Do not output any other text. Use the following format:

                        {
                            "arguments_for": {
                                "argument_id": {
                                    "argument": "argument_text",
                                    "argument_diagnosticity_score": "diagnosticity_score",
                                    "evidence": [
                                        {
                                            "evidence_id": "evidence_text",
                                            "evidence_diagnosticity_score": "diagnosticity_score"
                                        }
                                    ]
                                }
                            },
                            "arguments_against": {
                                "argument_id": {
                                    "argument": "argument_text",
                                    "argument_diagnosticity_score": "diagnosticity_score",
                                    "evidence": [
                                        {
                                            "evidence_id": "evidence_text",
                                            "evidence_diagnosticity_score": "diagnosticity_score"
                                        }
                                    ]
                                }
                            }
                        }
                        """
                    )
                ),
                HumanMessage(
                    content=(
                        f"Hypothesis: {state['hypothesis']}\n\n"
                        f"Arguments and Evidence: {state['extraction']}"
                    )
                ),
            ]
        )
        key_evidence = []
        try:
            scores = json.loads(response.content)
            argument_threshold = 1
            evidence_threshold = 1
            for key in scores:
                for argument_data in scores[key].values():
                    if (
                        argument_data["argument_diagnosticity_score"]
                        >= argument_threshold
                    ):
                        for evidence_data in argument_data["evidence"]:
                            if (
                                evidence_data["evidence_diagnosticity_score"]
                                >= evidence_threshold
                            ):
                                key_evidence.append(evidence_data["evidence"])

        except Exception as e:
            print(e)
            key_evidence.append("None")

        return {
            "diagnostic_score": key_evidence,
            "model_calls": [capture_metadata("diagnosticity", response)],
        }

    def analysis(state: State):
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        """
                        You are a detective attempting to solve a case using the Analysis of Competing Hypotheses method.
                        Based the key evidence, determine the likelihood that the stated hypothesis is correct.

                        Give the likelihood as a value on a scale of 1-5 with 1 being Very Unlikely and 5 as Very Highly Likely.
                        Output your final answer as a dictionary: 
                        
                        Only output the dictionary. Do not output any other text. Use the following format: {'suspect':suspect_name, 'likelihood':likelihood_score}
                        """
                    )
                ),
                HumanMessage(
                    content=(
                        f"Hypothesis:\n{state['hypothesis']}\n\n"
                        f"Key Evidence:\n{state['diagnostic_score']}"
                    )
                ),
            ]
        )

        return {
            "likelihood_hypothesis": response.content,
            "model_calls": [capture_metadata("analysis", response)],
        }

    builder = StateGraph(State)
    builder.add_node("extraction", evidence_extraction)
    builder.add_node("diagnosticity", diagnosticity)
    builder.add_node("analysis", analysis)

    builder.add_edge(START, "extraction")
    builder.add_edge("extraction", "diagnosticity")
    builder.add_edge("diagnosticity", "analysis")
    builder.add_edge("analysis", END)
    return builder.compile()


def multi_agent_invoke(multi_graph, story, index, narrative, results):
    evals = []

    for hypothesis in story["hypothesis"]:
        print(f"Running Story {index}, Hypothesis: {hypothesis}")
        result = multi_graph.invoke(
            {
                "hypothesis": hypothesis,
                "narrative": narrative,
                "extraction": "",
                "diagnostic_score": "",
                "likelihood_hypothesis": "",
                "model_calls": [],
            }
        )
        evals.append(ast.literal_eval(result["likelihood_hypothesis"]))
        results.append(result)

    evalA = evals[0]
    print(evalA)

    evalB = evals[1]
    print(evalB)

    if evalA["likelihood"] > evalB["likelihood"]:
        generated_answer = evalA["suspect"]
    elif evalA["likelihood"] < evalB["likelihood"]:
        generated_answer = evalB["suspect"]
    else:
        generated_answer = "Indecisive"
    return generated_answer
