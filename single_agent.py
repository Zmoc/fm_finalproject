from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
import csv

with open("data/murder_mystery.csv", encoding="utf-8") as eval_data:
    reader = csv.DictReader(eval_data, delimiter=",")

    line = next(reader)
    narrative = line["narrative"]
    question = line["question"]
    choices = line["choices"]
    correct_answer = line["answer_choice"]

    prompt = f"{narrative}\n\n{question}\n\nYour choices are: {choices}"
    answer_prompt = f"ANSWER: {correct_answer}"
    print(prompt)
    print(answer_prompt)


# A tool is an ordinary function with typed inputs and a clear description.
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


tools = [multiply]

model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
).bind_tools(tools)


# This is the single agent node.
def agent(state: MessagesState):
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the following multiple choice question."
                    "The last line of your response should be of the following format: 'ANSWER: [NAME]' (without quotes) where [NAME] is the chosen suspect."
                    "Think step by step before answering."
                    "Clearly specify the evidence that you are using to justify your answer."
                )
            ),
            *state["messages"],
        ]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"recursion_limit": 10},
    )

    response = result["messages"][1]
    content = response.content
    answer = content.split("\n")[2]
    result = answer_prompt == answer
