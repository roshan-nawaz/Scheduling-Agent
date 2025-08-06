from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Union, Annotated, Sequence
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def extract_nlu_info(state: AgentState) -> AgentState:

    system_prompt = SystemMessage(content="""
You are a scheduling assistant that extracts structured scheduling parameters from natural language queries.

Your job is to extract the following fields:
1. **JobID**: The job identifier (e.g., JOB001). If not mentioned, return "UNKNOWN".
2. **Constraints**: A list of constraints or hints from the query, such as "ASAP", "skill match", "proximity", "availability", "zone".
3. **ConstraintPriority**: If the query implies a priority (e.g., "as soon as possible", "skill is more important than distance"), return the constraints in that order. Otherwise, use the default: ["ASAP", "skill match", "proximity"].
4. **Intent**: The user's scheduling intent, typically one of ["schedule", "assign", "reschedule"].

Output strictly in the following JSON format:
{
  "JobID": "<JOB_ID or 'UNKNOWN'>",
  "Constraints": ["<constraint_1>", "<constraint_2>", ...],
  "ConstraintPriority": ["<constraint_1>", "<constraint_2>", ...],
  "Intent": "<intent>"
}

Examples:
Query: "Please assign JOB004 urgently to someone nearby."
→
{
  "JobID": "JOB004",
  "Constraints": ["ASAP", "proximity"],
  "ConstraintPriority": ["ASAP", "proximity", "skill match"],
  "Intent": "assign"
}

Query: "Schedule a job for tomorrow. Make sure the technician is skilled in plumbing."
→
{
  "JobID": "UNKNOWN",
  "Constraints": ["skill match", "availability"],
  "ConstraintPriority": ["skill match", "availability", "proximity"],
  "Intent": "schedule"
}
""")

    response = llm.invoke([system_prompt] + state['messages'])
    return {"messages": [response]}

graph = StateGraph(AgentState)

graph.add_node("nlu", extract_nlu_info)
graph.add_edge(START, "nlu")
graph.add_edge("nlu", END)
app = graph.compile()

result = app.invoke({"messages": [HumanMessage(content="Can you assign any one for job 123?")]})
print(result["messages"][-1].content)








