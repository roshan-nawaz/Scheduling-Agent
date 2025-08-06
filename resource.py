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
    job_id: str       
    constraints: List[str] 
    priority: List[str] 
    intent: str         
    technician_names: List[str]
    travel_times: dict




