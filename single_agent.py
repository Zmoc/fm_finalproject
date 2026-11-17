from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph


def single_agent_graph(model, temperature, reasoning):
    class State(TypedDict):
        question: str
        narrative: str
        choices: str

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
                        The last line of your response should be only the name of the chosen suspect or Indecisive, no other words, descriptions, or formatting.
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
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    return builder.compile()
