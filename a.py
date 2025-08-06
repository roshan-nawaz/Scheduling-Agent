import pandas as pd
import json
import re
import googlemaps
import os

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated, Sequence
from langgraph.prebuilt import ToolNode
from langgraph.graph import add_messages

from dotenv import load_dotenv
load_dotenv()

# --- Data Loading & State (No Changes) ---
DEFAULT_SCHEDULING_PRIORITY = ["ASAP", "skill match", "proximity"]
EXCEL_FILE_PATH = "data.xlsx"
xls = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
jobs_df = xls['Jobs']
resource_df = xls['Resources']
location_df = xls['Locations']
policy_df = xls['Sheet1']

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    job_id: str
    constraints: List[str]
    priority: List[str]
    intent: str
    technician_names: List[str]
    travel_times: dict
    top3_technicians: List[dict]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

GOOGLE_MAPS_API = os.getenv("GOOGLE_MAPS_API")

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API)

# --- NLU Node (No Changes) ---
def extract_nlu_info(state: AgentState) -> dict:
    # This function is fine as is.
    # (Code omitted for brevity, it's the same as your original)
    system_prompt_content = SystemMessage(content=f"""
You are a scheduling assistant that extracts structured scheduling parameters from natural language queries. Your job is to extract the following fields: JobID, Constraints, ConstraintPriority, Intent. Output strictly in the following JSON format:
{{"JobID": "<JOB_ID or 'UNKNOWN'>", "Constraints": [], "ConstraintPriority": [], "Intent": "<intent>"}}""")
    user_message = state['messages'][-1]
    response = llm.invoke([system_prompt_content, user_message])
    llm_output_content = response.content
    print("NLU Raw Output from LLM:\n", llm_output_content)
    try:
        parsed = json.loads(re.search(r"\{.*\}", llm_output_content, re.DOTALL).group(0))
        return {
            "messages": [response],
            "job_id": parsed.get("JobID", "UNKNOWN"),
            "constraints": parsed.get("Constraints", []),
            "priority": parsed.get("ConstraintPriority", DEFAULT_SCHEDULING_PRIORITY),
            "intent": parsed.get("Intent", "UNKNOWN"),
        }
    except Exception as e:
        print(f"Error parsing NLU, using defaults: {e}")
        return {"messages": [response]}


# --- HELPER FUNCTIONS (No Changes) ---
def clean_lat_lon(latlon): return str(latlon).split("°")[0].strip()
def get_travel_time(origin, destination):
    try:
        result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode="driving")
        return round(result["rows"][0]["elements"][0]["duration"]["value"] / 60, 2)
    except Exception: return float('inf')

# --- MODIFIED TOOLS ---
# Tools now return a JSON string. This string contains a dictionary
# with keys that match the AgentState for easy updating.

@tool
def get_technicians(job_id: str) -> str:
    """
    Given a job ID, returns technicians in the same zone.
    The output is a JSON string mapping 'technician_names' to a list of names.
    """
    print(f"\n--- TOOL: get_technicians for {job_id} ---")
    job_row = jobs_df[jobs_df["JobID"] == job_id]
    location_id = job_row.iloc[0]["LocationID"]
    location_row = location_df[location_df["LocationID"] == location_id]
    zone = location_row.iloc[0]["Zone"]
    techs = resource_df[resource_df["LocationZones"].str.contains(zone, na=False)]
    technician_names = techs["Name"].tolist()
    print(f"Found: {technician_names}")
    return json.dumps({"technician_names": technician_names})

@tool
def get_travel_times(job_id: str, technician_names: List[str]) -> str:
    """
    Calculates travel times for a list of technicians to a job site.
    The output is a JSON string mapping 'travel_times' to a dictionary of {name: time}.
    """
    print(f"\n--- TOOL: get_travel_times for {technician_names} ---")
    job_row = jobs_df[jobs_df["JobID"] == job_id]
    location_id = job_row.iloc[0]["LocationID"]
    location_row = location_df[location_df["LocationID"] == location_id]
    job_coord = f"{clean_lat_lon(location_row.iloc[0]['Latitude'])},{clean_lat_lon(location_row.iloc[0]['Longitude'])}"
    
    travel_times = {}
    for tech_name in technician_names:
        tech_row = resource_df[resource_df["Name"] == tech_name]
        tech_coord = f"{clean_lat_lon(tech_row.iloc[0]['BaseLocationLat'])},{clean_lat_lon(tech_row.iloc[0]['BaseLocationLon'])}"
        travel_times[tech_name] = get_travel_time(tech_coord, job_coord)
    print(f"Calculated: {travel_times}")
    return json.dumps({"travel_times": travel_times})

@tool
def score_and_rank_technicians(job_id: str, technician_names: List[str], travel_times: dict, priority: List[str]) -> str:
    """
    Scores and ranks technicians based on skills, travel time, and priority.
    The output is a JSON string mapping 'top3_technicians' to a list of scored technician dicts.
    """
    print("\n--- TOOL: score_and_rank_technicians ---")
    job_row = jobs_df[jobs_df["JobID"] == job_id]
    required_skills = [s.strip().lower() for s in job_row.iloc[0]["SkillsRequired"].split(',')]
    
    valid_times = [t for t in travel_times.values() if t != float('inf')]
    min_time, max_time = (min(valid_times), max(valid_times)) if valid_times else (0, 1)
    time_range = max_time - min_time if max_time != min_time else 1.0

    skill_weight = 0.7 if priority and priority[0] == 'skill match' else 0.3
    
    scores = []
    for tech_name in technician_names:
        tech_row = resource_df[resource_df["Name"] == tech_name]
        tech_skills = [s.strip().lower() for s in tech_row.iloc[0]["Skills"].split(',')]
        skill_score = sum(1 for s in required_skills if s in tech_skills) / len(required_skills) if required_skills else 0
        
        time = travel_times.get(tech_name, float('inf'))
        travel_score = 1 - ((time - min_time) / time_range) if time != float('inf') else 0
        final_score = (skill_weight * skill_score) + ((1 - skill_weight) * travel_score)
        
        scores.append({"name": tech_name, "final_score": final_score})

    top3 = sorted(scores, key=lambda x: x["final_score"], reverse=True)[:3]
    print(f"Ranked Top 3: {top3}")
    return json.dumps({"top3_technicians": top3})

# --- Agent and Graph Definition ---
tools = [get_technicians, get_travel_times, score_and_rank_technicians]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    """The agent node that decides the next action."""
    print("\n--- AGENT: Deciding next step... ---")
    # The agent now has access to the full, updated state
    response = llm_with_tools.invoke(state['messages'])
    # The response will either be a tool call or a final answer
    return {"messages": [response]}

# ✨ NEW NODE TO UPDATE STATE FROM TOOL OUTPUT ✨
def state_updater_node(state: AgentState) -> dict:
    """Parses the last ToolMessage and updates the state."""
    last_message = state['messages'][-1]
    if isinstance(last_message, ToolMessage):
        print("--- UPDATER: Updating state from tool output ---")
        tool_output = json.loads(last_message.content)
        # The keys in tool_output ('technician_names', etc.)
        # will automatically update the corresponding keys in the AgentState
        return tool_output
    return {}

def should_continue(state: AgentState) -> str:
    """Conditional edge to decide whether to continue or end."""
    last_message = state['messages'][-1]
    # If the agent made a tool call, loop back to the tools.
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    # Otherwise, the agent is done, so we end.
    return "end"

# Define the graph
graph = StateGraph(AgentState)

graph.add_node("nlu", extract_nlu_info)
graph.add_node("agent", agent_node)
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)
graph.add_node("state_updater", state_updater_node)

graph.set_entry_point("nlu")
graph.add_edge("nlu", "agent")

# This is the main agentic loop
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END}
)
graph.add_edge("tools", "state_updater")
graph.add_edge("state_updater", "agent")

app = graph.compile()

# --- Run the application ---
query = "Can you assign JOB001 to someone nearby, skill is more important and time can be taken?. Which technicians can attend JOB001"
inputs = {"messages": [HumanMessage(content=query)]}
final_state = app.invoke(inputs)

# --- Print Final Output ---
print("\n" + "="*50)
print("              Final Result")
print("="*50)
print("Final Summary from Agent:")
# The final answer is the content of the last message
print(final_state['messages'][-1].content)
print("="*50)