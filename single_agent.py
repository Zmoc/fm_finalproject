from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph


def single_agent_graph(model, temperature, reasoning):
    model = ChatOllama(
        model=model,
        temperature=temperature,
        reasoning=reasoning,
    )

    def agent(state: MessagesState):
        response = model.invoke(
            [
                SystemMessage(
                    content=(
                        """
                        Answer the following multiple choice question.
                        Think step by step before answering.
                        Clearly specify the evidence that you are using to justify your answer.
                        If neither choice seems likely, output "Indecisive"
                        The last line of your response should be only the name of the chosen suspect or Indecisive, no other words or descriptions.
                        """
                    )
                ),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    return builder.compile()
