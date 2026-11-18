import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph

from eval_helpers.eval_suite import capture_metadata


def single_agent_graph(model, temperature, reasoning):
    class State(TypedDict):
        question: str
        narrative: str
        choices: str
        messages: str
        model_calls: Annotated[list[dict[str, Any]], operator.add]

    model = ChatOllama(
        model=model,
        temperature=temperature,
        reasoning=reasoning,
    )

    def build_graph(State, agent):
        builder = StateGraph(State)
        builder.add_node("agent", agent)
        builder.add_edge(START, "agent")
        return builder.compile()

    def agent(state: State):
        system_prompt = """
                        Answer the following multiple choice question.
                        Think step by step before answering.
                        Clearly specify the evidence that you are using to justify your answer.
                        If neither choice seems likely, output "Indecisive"
                        Output only the name of the chosen suspect or Indecisive, no other words, descriptions, or formatting.
                        """

        input_prompt = (
            f"{state['narrative']}\n\n"
            f"{state['question']}\n\n"
            f"Your choices are: {state['choices']}"
        )
        response = model.invoke(
            [
                SystemMessage(content=(system_prompt)),
                HumanMessage(content=(input_prompt)),
            ]
        )
        return {
            "messages": [response],
            "model_calls": [capture_metadata("evidence_extraction", response)],
        }

    return build_graph(State, agent)


def single_agent_invoke(graph, story, index, narrative):
    results = []

    question = story["question"]
    choices = story["choices"]

    print(f"Running Story {index}")

    invocation_state = {
        "question": question,
        "narrative": narrative,
        "choices": choices,
        "messages": "",
        "model_calls": [],
    }
    result = graph.invoke(invocation_state)
    results.append(result)

    response = result["messages"][0]
    generated_answer = response.content.split("\n")[0]
    return generated_answer, results
