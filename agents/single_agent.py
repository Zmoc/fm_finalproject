import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph

from eval_helpers.eval_suite import capture_metadata


def single_agent_graph(model, temperature, reasoning, timeout, top_K, top_P, seed):
    class State(TypedDict):
        narrative: str
        choices: str
        messages: str
        model_calls: Annotated[list[dict[str, Any]], operator.add]

    model = ChatOllama(
        model=model,
        temperature=temperature,
        reasoning=reasoning,
        format="json",
        seed=seed,
        top_k=top_K,
        top_p=top_P,
        client_kwargs={"timeout": timeout},
    )

    def build_graph(State, single_agent):
        builder = StateGraph(State)
        builder.add_node("agent", single_agent)
        builder.add_edge(START, "agent")
        return builder.compile()

    def single_agent(state: State):
        system_prompt = """
                        You are a detective attempting to solve a murder.
                        Based on the available information, determine who the most likely culprit is.

                        Return a JSON object with this structure:
                        {"suspect": "suspect name"}
                        """

        input_prompt = (
            f"Narrative: {state['narrative']}\n\nAvailable suspects: {state['choices']}"
        )
        response = model.invoke(
            [
                SystemMessage(content=(system_prompt)),
                HumanMessage(content=(input_prompt)),
            ]
        )
        return {
            "messages": [response],
            "model_calls": [capture_metadata("single_agent", response)],
        }

    return build_graph(State, single_agent)


def single_agent_invoke(graph, story, index, narrative):
    results = []
    choices = story["choices"]

    print(f"Running Story {index}")

    invocation_state = {
        "narrative": narrative,
        "choices": choices,
        "messages": "",
        "model_calls": [],
    }

    try:
        result = graph.invoke(invocation_state)
        results.append(result)
        generated_answer = json.loads(result["messages"][0].content)["suspect"]

    except TimeoutError:
        generated_answer = "Timeout"

    return generated_answer, results
