import pandas as pd
import json       
import re           
import googlemaps
import os

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated, Sequence
from langgraph.prebuilt import ToolNode 
from langgraph.graph import add_messages



from dotenv import load_dotenv
load_dotenv()


DEFAULT_SCHEDULING_PRIORITY = ["ASAP", "skill match", "proximity"]
EXCEL_FILE_PATH = "data.xlsx"
xls = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
# print("Available sheet names:", xls.keys())

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
    top3_technicians : List[dict]



def extract_nlu_info(state: AgentState) -> AgentState:
    system_prompt_content = SystemMessage(content=f"""
You are a scheduling assistant that extracts structured scheduling parameters from natural language queries.

Your job is to extract the following fields:
1. **JobID**: The job identifier (e.g., JOB001). If not mentioned, return "UNKNOWN".
2. **Constraints**: A list of constraints or hints from the query, such as "ASAP", "skill match", "proximity", "availability", "zone".
3. **ConstraintPriority**: If the query implies a priority (e.g., "as soon as possible", "skill is more important than distance"), return the constraints in that order. Otherwise, use the default: {json.dumps(DEFAULT_SCHEDULING_PRIORITY)}.
4. **Intent**: The user's scheduling intent, typically one of ["schedule", "assign", "reschedule", "query_status", "find_technician", "general_query"].

Output strictly in the following JSON format:
{{
  "JobID": "<JOB_ID or 'UNKNOWN'>",
  "Constraints": ["<constraint_1>", "<constraint_2>", ...],
  "ConstraintPriority": ["<constraint_1>", "<constraint_2>", ...],
  "Intent": "<intent>"
}}

Examples:
Query: "Please assign JOB004 urgently to someone nearby."
→
{{
  "JobID": "JOB004",
  "Constraints": ["ASAP", "proximity"],
  "ConstraintPriority": ["ASAP", "proximity", "skill match"],
  "Intent": "assign"
}}

Query: "Schedule a job for tomorrow. Make sure the technician is skilled in plumbing."
→
{{
  "JobID": "UNKNOWN",
  "Constraints": ["skill match", "availability"],
  "ConstraintPriority": ["skill match", "availability", "proximity"],
  "Intent": "schedule"
}}
"""
    )
    user_message = state['messages'][-1]

    response = llm.invoke([system_prompt_content, user_message])
    llm_output_content = response.content
    print("NLU Raw Output from LLM:\n", llm_output_content) 

    parsed_data = {
        "JobID": "UNKNOWN",
        "Constraints": [],
        "ConstraintPriority": DEFAULT_SCHEDULING_PRIORITY, 
        "Intent": "UNKNOWN"
    }

    try:
        json_match = re.search(r"\{.*\}", llm_output_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            parsed_data["JobID"] = parsed.get("JobID", "UNKNOWN")
            parsed_data["Constraints"] = parsed.get("Constraints", [])
            parsed_data["ConstraintPriority"] = parsed.get("ConstraintPriority", DEFAULT_SCHEDULING_PRIORITY)
            parsed_data["Intent"] = parsed.get("Intent", "UNKNOWN")
        else:
            print("Warning: No JSON block found in LLM response. Using default NLU values.")
    except json.JSONDecodeError as e:
        print("Using default NLU values.")
    except Exception as e:
        print("Using default NLU values.")
    
    return {
        "messages": state['messages'] + [response], # Correctly append LLM's response
        "job_id": parsed_data["JobID"],
        "constraints": parsed_data["Constraints"],
        "priority": parsed_data["ConstraintPriority"], # Storing the determined priority
        "intent": parsed_data["Intent"]
    }


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



GOOGLE_MAPS_API = os.getenv("GOOGLE_MAPS_API")

gmaps = googlemaps.Client(key=GOOGLE_MAPS_API)


def clean_lat_lon(latlon):
    return str(latlon).split("°")[0].strip()

def get_travel_time(origin, destination) -> float:

    """using google maps api"""
    try:

        result = gmaps.distance_matrix(origins = [origin], destinations = [destination], mode = "driving")
        duration = result["rows"][0]["elements"][0]["duration"]["value"]
        return round(duration / 60, 2)
    except Exception as e:
        print(f"Error getting travel time: {e}")
        return None


@tool
def get_travel_times(job_id: str, technician_names: list) -> dict:
    """
    Calculates travel times (in minutes) between a job location and each technician using Google Maps API.
    
    Args:
        job_id (str): The job identifier.
        technician_names (list): List of technician names.

    Returns:
        dict: Mapping of technician names to travel time in minutes.
    """
    job_row = jobs_df[jobs_df["JobID"] == job_id]
    location_id = job_row.iloc[0]["LocationID"]
    location_row = location_df[location_df["LocationID"] == location_id]

    job_lat = clean_lat_lon(location_row.iloc[0]["Latitude"])
    job_long = clean_lat_lon(location_row.iloc[0]["Longitude"])
    job_coord = f"{job_lat},{job_long}"

    travel_time = {}
    for tech_name in technician_names:
        tech_row = resource_df[resource_df["Name"] == tech_name]
        tech_lat = clean_lat_lon(tech_row.iloc[0]["BaseLocationLat"])
        tech_long = clean_lat_lon(tech_row.iloc[0]["BaseLocationLon"])
        tech_coord = f"{tech_lat},{tech_long}"
        travel_time[tech_name] = get_travel_time(tech_coord, job_coord)

    return travel_time


@tool
def score_and_rank_technicians(job_id: str, technician_names: list, travel_times: dict) -> list:
    """
    Computes and ranks technicians for a job using skill match and travel time.

    Args:
        job_id (str): The job identifier.
        technician_names (list): List of technician names.
        travel_times (dict): Technician name to travel time mapping.

    Returns:
        list: Top 3 technician dicts with name, skill_score, travel_score, and final_score.
    """
    job_row = jobs_df[jobs_df["JobID"] == job_id]
    tech_skill_req = job_row.iloc[0]["SkillsRequired"]
    required_skills = [s.strip().lower() for s in tech_skill_req.split(',')]

    valid_times = [t for t in travel_times.values() if t is not None]
    if not valid_times:
        return []

    min_time = min(valid_times)
    max_time = max(valid_times)
    time_range = max_time - min_time if max_time != min_time else 1.0

    tech_skill_score = []
    for tech_name in technician_names:
        tech_row = resource_df[resource_df["Name"] == tech_name]
        tech_skills_str = tech_row.iloc[0]["Skills"]
        tech_skills = [s.strip().lower() for s in tech_skills_str.split(',')]

        matched_skills = sum(1 for skill in required_skills if skill in tech_skills)
        skill_score = matched_skills / len(required_skills) if required_skills else 0

        travel_time = travel_times[tech_name]
        travel_score = 1 - ((travel_time - min_time) / time_range) if travel_time is not None else 0

        final_score = 0.5 * skill_score + 0.5 * travel_score
        tech_skill_score.append({
            "name": tech_name,
            "skill_score": skill_score,
            "travel_score": travel_score,
            "final_score": final_score
        })

    ranks = sorted(tech_skill_score, key=lambda x: x["final_score"], reverse=True)
    return ranks[:3]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [get_technicians, get_travel_times, score_and_rank_technicians]
llm_with_tools = llm.bind_tools(tools)



def model_call(state: AgentState) -> AgentState:

    sys_prompt_content =SystemMessage(content="""
You are a helpful scheduling assistant.

You must decide whether to call another tool or respond directly.

If the user's query involves:
- scoring technicians,
- calculating skill or travel scores,
- ranking technicians,
then call the tool `score_and_rank_technicians` with the job_id, technician_names, and travel_times.

Otherwise, summarize the previous tool outputs.

DO NOT skip scoring if the user requests it.
"""
    )
    
    response = llm_with_tools.invoke(state["messages"] + [sys_prompt_content])

    content = response.content
    technician_names = re.findall(r"- ([A-Za-z ]+)", content)
    
    return {
        "messages": [response],
        "technician_names": technician_names
    }


def shld_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:   
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("nlu", extract_nlu_info)
graph.add_node("agent", model_call)

tool_node = ToolNode(tools)    
graph.add_node("tools", tool_node)

graph.set_entry_point("nlu")
graph.add_edge("nlu", "agent")

graph.add_conditional_edges(
    "agent",
    shld_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "agent")    # Loop back for multi-step tool reasoning

app = graph.compile()


result = app.invoke({"messages": [HumanMessage(content="Can you assign JOB001 to someone nearby, skill is more important and time can be taken?. Which technicians can attend JOB001. Give scores of technicians who are eligible?")]})
# print("\nFinal NLU Output:")
print(f"JobID: {result['job_id']}")
# print(f"Constraints: {result['constraints']}")
# print(f"Priority: {result['priority']}")
# print(f"Intent: {result['intent']}")
# print(result["messages"][-1].content)


# print(result['technician_names'])
# print(result['travel_times'])
print(result["messages"][-1].content)