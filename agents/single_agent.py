import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph


def single_agent_graph(model, temperature, reasoning):
    class State(TypedDict):
        question: str
        narrative: str
        choices: str
        messages: str
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

    model = ChatOllama(
        model=model,
        temperature=temperature,
        reasoning=reasoning,
    )

    def agent(state: State):
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        """
                        Answer the following multiple choice question.
                        Think step by step before answering.
                        Clearly specify the evidence that you are using to justify your answer.
                        If neither choice seems likely, output "Indecisive"
                        Output only the name of the chosen suspect or Indecisive, no other words, descriptions, or formatting.
                        """
                    )
                ),
                HumanMessage(
                    content=(
                        f"{state['narrative']}\n\n"
                        f"{state['question']}\n\n"
                        f"Your choices are: {state['choices']}"
                    )
                ),
            ]
        )
        return {
            "messages": [response],
            "model_calls": [capture_metadata("evidence_extraction", response)],
        }

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    return builder.compile()


def single_agent_invoke(single_graph, story, index, narrative, results):
    question = story["question"]
    choices = story["choices"]

    print(f"Running Story {index}")

    result = single_graph.invoke(
        {
            "question": question,
            "narrative": narrative,
            "choices": choices,
            "messages": "",
            "model_calls": [],
        }
    )
    results.append(result)

    response = result["messages"][0]
    generated_answer = response.content.split("\n")[0]
    return generated_answer
