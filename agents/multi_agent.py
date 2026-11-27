import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from eval_helpers.eval_suite import capture_metadata


def multi_agent_graph(model, temperature, reasoning, timeout, topK, topP, seed):
    class State(TypedDict):
        summary: str
        hypothesis: str
        narrative: str
        extraction: str
        key_evidence: list
        chosen_suspect: str
        model_calls: Annotated[list[dict[str, Any]], operator.add]

    model = ChatOllama(
        model=model,
        temperature=temperature,
        reasoning=reasoning,
        format="json",
        top_k=topK,
        top_p=topP,
        client_kwargs={"timeout": timeout},
        seed=seed,
    )

    def build_graph(State, summarizer, evidence_extraction, diagnosticity, analysis):
        builder = StateGraph(State)
        builder.add_node("summarizer", summarizer)
        builder.add_node("extraction", evidence_extraction)
        builder.add_node("diagnosticity", diagnosticity)
        builder.add_node("analysis", analysis)

        builder.add_edge(START, "summarizer")
        builder.add_edge("summarizer", "extraction")
        builder.add_edge("extraction", "diagnosticity")
        builder.add_edge("diagnosticity", "analysis")
        builder.add_edge("analysis", END)
        return builder.compile()

    def summarizer(state: State):
        summary_system_prompt = """
                        You are a detective reviewing a case file.
                        
                        Summarize the narrative impartially.
                        Do not speculate who the murderer is.
                        Do not specify anything as key evidence.
                        Only report the background of the case.

                        Return a JSON object with this structure:
                        {
                           "summary": "summary of the narrative"
                        }
                        """

        summary_input_prompt = f"Narrative: {state['narrative']}"

        response = model.invoke(
            [
                SystemMessage(content=(summary_system_prompt)),
                HumanMessage(content=(summary_input_prompt)),
            ]
        )

        return {
            "summary": response.content,
            "model_calls": [capture_metadata("summarizer", response)],
        }

    def evidence_extraction(state: State):
        evidence_system_prompt = """
                        You are a detective using the Analysis of Competing Hypotheses method.
                        Make a list of significant evidence and arguments for and against the given hypothesis.
                        Return a JSON object with this structure:

                        {
                            "arguments_for": {
                                argument1: {
                                    "argument": "Description of the specific argument in support of the given hypothesis",
                                    "evidence": [
                                        {
                                            "evidence_id": argument1_evidence1,
                                            "evidence_text": "Description of the specific evidence",
                                        }
                                    ]
                                }
                            },
                            "arguments_against": {
                                argument2: {
                                    "argument": "Description of the specific argument against the given hypothesis",
                                    "evidence": [
                                        {
                                            "evidence_id": argument2_evidence1,
                                            "evidence_text": "Description of the specific evidence",
                                        }
                                    ]
                                }
                            }
                        }
                        """

        evidence_input_prompt = (
            f"Hypothesis: {state['hypothesis']}\n\nNarrative: {state['narrative']}"
        )

        response = model.invoke(
            [
                SystemMessage(content=(evidence_system_prompt)),
                HumanMessage(content=(evidence_input_prompt)),
            ]
        )

        return {
            "extraction": response.content,
            "model_calls": [capture_metadata("evidence_extraction", response)],
        }

    def diagnosticity(state: State):
        diagnostic_system_prompt = """
                        You are an intelligence analyst using the Analysis of Competing Hypotheses method.
                        Examine the given arguments and supporting evidence and determine the diagnosticity of each.
                        
                        Diagnosticity refers to how useful the information is in determining the likelihood of the stated hypothesis.
                        Assign a score to each argument and individual piece of evidence from 1-5 where 1 is least diagnostic and 5 is the most diagnostic.
                        
                        A piece of evidence that is likely to be seen regardless of the hypothesis being True or False has a diagnostic value of 1.
                        A piece of evidence that can only appear if the hypothesis is exclusively True or False (but not both) has a diagnostic value of 5.
                        
                        Return a JSON object with this structure:
                        {
                            "arguments_for": {
                                argument1: {
                                    "argument": "Description of the specific argument in support of the given hypothesis",
                                    "argument_diagnosticity_score": 3,
                                    "evidence": [
                                        {
                                            "evidence_id": argument1_evidence1,
                                            "evidence_text": "Description of the specific evidence",
                                            "evidence_diagnosticity_score": 3
                                        }
                                    ]
                                }
                            },
                            "arguments_against": {
                                argument2: {
                                    "argument": "Description of the specific argument against the given hypothesis",
                                    "argument_diagnosticity_score": 3,
                                    "evidence": [
                                        {
                                            "evidence_id": argument2_evidence1,
                                            "evidence_text": "Description of the specific evidence",
                                            "evidence_diagnosticity_score": 3
                                        }
                                    ]
                                }
                            }
                        }
                        """

        diagnostic_input_prompt = (
            f"Hypothesis: {state['hypothesis']}\n\n"
            f"Arguments and Evidence: {state['extraction']}"
        )
        response = model.invoke(
            [
                SystemMessage(content=(diagnostic_system_prompt)),
                HumanMessage(content=(diagnostic_input_prompt)),
            ]
        )

        key_evidence = evidence_filtering(response)

        return {
            "key_evidence": key_evidence,
            "model_calls": [capture_metadata("diagnosticity", response)],
        }

    def evidence_filtering(response):
        key_evidence = []

        scores = json.loads(response.content, strict=False)
        argument_threshold = 1
        evidence_threshold = 1
        for key in scores:
            try:
                for argument_data in scores[key].values():
                    if (
                        argument_data["argument_diagnosticity_score"]
                        >= argument_threshold
                    ):
                        if not argument_data["evidence"]:
                            print(scores)
                        else:
                            for evidence_data in argument_data["evidence"]:
                                if (
                                    evidence_data["evidence_diagnosticity_score"]
                                    >= evidence_threshold
                                ):
                                    key_evidence.append(evidence_data["evidence_text"])
            except Exception as e:
                print(e)
                print(response.content)

        return key_evidence

    def analysis(state: State):
        analysis_system_prompt = """
                        You are a detective attempting to solve a case using the Analysis of Competing Hypotheses method.
                        Based on the narrative and key evidence, determine who the most likely culprit is.
                        
                        Return a JSON object with this structure:
                        {"suspect": "suspect name"}
                        """

        analysis_input_prompt = (
            f"Hypothesis:\n{state['hypothesis']}\n\n"
            f"Summary:\n{state['summary']}\n\n"
            f"Narrative:\n{state['narrative']}\n\n"
            f"Key Evidence:\n{state['key_evidence']}\n\n"
        )
        response = model.invoke(
            [
                SystemMessage(content=(analysis_system_prompt)),
                HumanMessage(content=(analysis_input_prompt)),
            ]
        )

        return {
            "chosen_suspect": response.content,
            "model_calls": [capture_metadata("analysis", response)],
        }

    return build_graph(State, summarizer, evidence_extraction, diagnosticity, analysis)


def multi_agent_invoke(graph, story, index, narrative):
    results = []

    def hypothesis_choose(evals):
        evalA = evals[0]
        evalB = evals[1]

        if evalA["suspect"] == evalB["suspect"]:
            chosen_suspect = evalA["suspect"]
        else:
            chosen_suspect = "Indecisive"
        return chosen_suspect

    evals = []
    try:
        for hypothesis in story["hypothesis"]:
            print(f"Running Story {index}, Hypothesis: {hypothesis}")

            invocation_state = {
                "summary": "",
                "hypothesis": hypothesis,
                "narrative": narrative,
                "extraction": "",
                "key_evidence": "",
                "chosen_suspect": "",
                "model_calls": [],
            }

            result = graph.invoke(invocation_state)
            results.append(result)
            suspect = result["chosen_suspect"]
            evals.append(json.loads(suspect))

        generated_answer = hypothesis_choose(evals)

    except TimeoutError:
        generated_answer = "Timeout"

    return generated_answer, results
