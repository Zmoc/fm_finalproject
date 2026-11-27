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
        narrative: str
        choices: str
        extraction: str
        key_evidence: list
        suspect: str
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
                        You are a detective attempting to solve a murder.
                        
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
                        You are a detective attempting to solve a murder.
                        Extract all relevent evidence in the narrative.

                        Return a JSON object with this structure:
                        {
                            "evidence": [
                                "Evidence1 Description",
                                "Evidence2 Description",
                                "Evidence3 Description",
                                "Evidence4 Description"
                                ...
                                "EvidenceN Description"
                            ]
                        }
                        """

        evidence_input_prompt = (
            f"Narrative: {state['narrative']}\n\nPossible Suspects: {state['choices']}"
        )

        response = model.invoke(
            [
                SystemMessage(content=(evidence_system_prompt)),
                HumanMessage(content=(evidence_input_prompt)),
            ]
        )

        return {
            "extraction": json.loads(response.content),
            "model_calls": [capture_metadata("evidence_extraction", response)],
        }

    def diagnosticity(state: State):
        response_list = []
        diagnostic_system_prompt = """
                        You are a detective attempting to solve a murder.
                        Using the narrative, if the given piece of evidence is Consistent or Inconsistent with each suspect.
                        
                        A piece of evidence is consistent with a given suspect if it is possible to see it if they are culprit.
                        A piece of evidence is inconsistent with a given suspect if you could NOT possibly see it if they are culprit.
                        A piece of evidence can be consistent with both suspects.
                        A piece of evidence can likewise be inconsistent with both suspects.
                        
                        Return a JSON object with this structure:
                        {
                            "Name of Suspect": "Consistent",
                        }
                        """

        evidence_list = state["extraction"]
        for evidence in evidence_list["evidence"]:
            diagnostic_input_prompt = (
                f"Narrative: {state['narrative']}\n\n",
                f"Evidence: {evidence}\n\n",
                f"Suspect 1: {state['choices'][0]}",
                f"Suspect 2: {state['choices'][1]}",
            )

            response = model.invoke(
                [
                    SystemMessage(content=(diagnostic_system_prompt)),
                    HumanMessage(content=(diagnostic_input_prompt)),
                ]
            )
            response_list.append((evidence, json.loads(response.content)))

        key_evidence = evidence_filtering(response_list, state["choices"])

        return {
            "key_evidence": key_evidence,
            "model_calls": [capture_metadata("diagnosticity", response)],
        }

    def evidence_filtering(response_list, choices):
        key_evidence = []

        for response in response_list:
            print(response)
            suspect1_rating = response[1][choices[0]]
            suspect2_rating = response[1][choices[1]]

            if suspect1_rating != suspect2_rating:
                key_evidence.append(response[0])

        return key_evidence

    def analysis(state: State):
        analysis_system_prompt = """
                        You are a detective attempting to solve a murder.
                        Based on the available information, determine who the most likely culprit is.

                        Return a JSON object with this structure:
                        {"suspect": "suspect name"}
                        """

        analysis_input_prompt = (
            f"Summary:\n{state['summary']}\n\n"
            f"Evidence:\n{state['key_evidence']}\n\n"
            f"Available suspects: {state['choices']}"
        )
        response = model.invoke(
            [
                SystemMessage(content=(analysis_system_prompt)),
                HumanMessage(content=(analysis_input_prompt)),
            ]
        )

        return {
            "suspect": json.loads(response.content),
            "model_calls": [capture_metadata("analysis", response)],
        }

    return build_graph(State, summarizer, evidence_extraction, diagnosticity, analysis)


def multi_agent_invoke(graph, story, index, narrative):
    results = []
    choices = story["choices"]

    try:
        print(f"Running Story {index}")

        invocation_state = {
            "summary": "",
            "narrative": narrative,
            "choices": choices,
            "extraction": "",
            "key_evidence": "",
            "suspect": "",
            "model_calls": [],
        }

        result = graph.invoke(invocation_state)
        results.append(result)
        generated_answer = result["suspect"]["suspect"]

    except TimeoutError:
        generated_answer = "Timeout"

    return generated_answer, results
