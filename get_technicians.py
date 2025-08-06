from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Union, Annotated, Sequence
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import pandas as pd

from dotenv import load_dotenv

load_dotenv()


EXCEL_FILE_PATH = "data.xlsx"
xls = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
# print("Available sheet names:", xls.keys())

jobs_df = xls['Jobs']
resource_df = xls['Resources']
location_df = xls['Locations']


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]



@tool
def get_technicians(job_id:str) -> str:
    """
    Given a job ID, returns the technicians who operate in the same zone as the job location.
    
    Args:
        job_id (str): The unique identifier for the job (e.g., 'JOB003').
        
    Returns:
        str: A string describing the available technicians, or an error message
             if the job, location, or technicians cannot be found.
    """
    print(f"Getting technicians for job {job_id}\n")

    job_row = jobs_df[jobs_df["JobID"] == job_id]
    if job_row.empty:
        return f"No job found with ID {job_id}"
    
    location_id = job_row.iloc[0]["LocationID"]
    location_row = location_df[location_df["LocationID"] == location_id]
    if location_row.empty:
        return f"No location found with ID {location_id}"
    
    zone = location_row.iloc[0]["Zone"]

    techs = resource_df[resource_df["LocationZones"].str.contains(zone, na=False)]

    if techs.empty:
        return f"No technicians available in zone {zone}"
    
    technician_names = techs["Name"].tolist()
    return f"Technicians available in zone {zone}: {', '.join(technician_names)}"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [get_technicians]
llm_with_tools = llm.bind_tools(tools)

def model_call(state: AgentState) -> AgentState:


    sys_prompt_content =SystemMessage(content="""
    You are a helpful scheduling assistant. A tool has just provided its output.
    Your task is to concisely summarize this output and provide a clear, informative response to the user.
    Do not include technical details about tool calls or JSON structures in your final response.
    Just give the user the information they requested based on the tool's result.
    """
    )
    
    response = llm_with_tools.invoke(state["messages"] + [sys_prompt_content])
    
    return {"messages": state["messages"] + [response]}


def shld_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:   
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
graph.set_entry_point("agent")

tool_node = ToolNode(tools)
graph.add_node("tools",tool_node)

graph.add_conditional_edges(
    "agent",
    shld_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph.add_edge("tools", "agent")


app = graph.compile()


result = app.invoke({"messages": [HumanMessage(content="Which technicians can attend JOB003?")]})
print(result["messages"][-1].content)

