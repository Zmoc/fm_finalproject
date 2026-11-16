from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph

model_choice = "qwen3:8b"
reasoning = False


def create_model(model_choice, reasoning):
    return ChatOllama(
        model=model_choice,
        temperature=0,
        reasoning=reasoning,
    )


def agent(state: MessagesState, model_choice=model_choice, reasoning=reasoning):
    response = create_model(model_choice, reasoning).invoke(
        [
            SystemMessage(
                content=(
                    "Answer the following multiple choice question.\n"
                    "The last line of your response should be of the following format: 'ANSWER: [NAME]' (without quotes) where [NAME] is the chosen suspect.\n"
                    "Think step by step before answering.\n"
                    "Clearly specify the evidence that you are using to justify your answer.\n"
                )
            ),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def graph():
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    return builder.compile()
