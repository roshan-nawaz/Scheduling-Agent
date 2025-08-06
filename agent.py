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


llm = ChatOpenAI(model="gpt-4o", temperature=0)

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

def get_travel_times_node(state: AgentState) -> AgentState:

    """
    Calculate travel times between technicians and the job location.

    This tool retrieves the coordinates for the specified job and each technician,
    then uses Google Maps API to calculate the driving time (in minutes) for each technician
    to reach the job site.

    Args:
        state (AgentState): The current workflow state, containing:
            - 'job_id': Job identifier.
            - 'technician_names': List of eligible technician names.

    Returns:
        dict: A dictionary with one key:
            - 'travel_times': Dict mapping technician names to travel time in minutes (float).
    """

    job_id = state["job_id"]
    names = state["technician_names"]


    job_row = jobs_df[jobs_df["JobID"] == job_id]
    location_id = job_row.iloc[0]["LocationID"]
    location_row = location_df[location_df["LocationID"] == location_id]

    job_lat = clean_lat_lon(location_row.iloc[0]["Latitude"])
    job_long = clean_lat_lon(location_row.iloc[0]["Longitude"])
    job_coord = f"{job_lat},{job_long}"

    travel_time = {}
    for tech_name in names:
        tech_row = resource_df[resource_df["Name"] == tech_name]
        tech_lat = clean_lat_lon(tech_row.iloc[0]["BaseLocationLat"])
        tech_long = clean_lat_lon(tech_row.iloc[0]["BaseLocationLon"])    
        tech_coord = f"{tech_lat},{tech_long}"

        travel_time[tech_name] = get_travel_time(tech_coord, job_coord)

    return {
        # "messages":state["messages"],
        "travel_times": travel_time
    }

def score_rank_node(state: AgentState) -> AgentState:

    """
    Compute composite scores and rank technicians for a job.

    This tool compares each technician's skills to the job's required skills and
    incorporates normalized travel time to compute a composite suitability score.
    The top 3 technicians (by score) are selected and returned.

    Args:
        state (AgentState): The current workflow state, containing:
            - 'job_id': Job identifier.
            - 'technician_names': List of eligible technician names.
            - 'travel_times': Dict mapping technician names to travel time in minutes.

    Returns:
        dict: A dictionary with one key:
            - 'top3_technicians': List of up to 3 technician dictionaries, each with:
                - 'name': Technician's name
                - 'skill_score': Fraction of required skills matched (0.0 to 1.0)
                - 'travel_score': Normalized travel score (0.0 to 1.0, lower time = higher score)
                - 'final_score': Composite score (float, 0.0 to 1.0)
    """

    job_id = state["job_id"]
    travel_times = state["travel_times"]
    technician_names = state["technician_names"]


    job_row = jobs_df[jobs_df["JobID"] == job_id]
    tech_skill_req = job_row.iloc[0]["SkillsRequired"]
    required_skills = [s.strip().lower() for s in tech_skill_req.split(',')]

    valid_times = [t for t in travel_times.values() if t is not None]
    if not valid_times:
        # print("No valid travel times found")
        return {
            "messages": state["messages"],
            "top3_technicians": []
        }

    min_time = min(valid_times)
    max_time = max(valid_times)
    time_range = max_time - min_time if max_time != min_time else 1.0

    tech_skill_score = []
    for tech_name in technician_names:
        tech_row = resource_df[resource_df["Name"] == tech_name]
        tech_skills_str = tech_row.iloc[0]["Skills"]
        tech_skills = [s.strip().lower() for s in tech_skills_str.split(',')]

        print(f"Technician: {tech_name}, Required: {required_skills}, Tech Skills: {tech_skills}")
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
    top3 = ranks[:3]

    for name in top3:
        print(f"{name['name']} - Skill: {name['skill_score']:.2f} - Travel: {name['travel_score']:.2f} - Final: {name['final_score']:.2f}")

    return {
        # "messages": state["messages"],
        "top3_technicians": top3
    }

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
graph.add_node("travel", get_travel_times_node)
graph.add_node("score", score_rank_node)

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


graph.add_edge("tools", "agent")
graph.add_edge("agent", "travel")
# graph.add_edge("travel", END)
graph.add_edge("travel", "score")
graph.add_edge("score", END)

app = graph.compile()

result = app.invoke({"messages": [HumanMessage(content="Can you assign JOB001 to someone nearby, skill is more important and time can be taken?. Which technicians can attend JOB001")]})
# print("\nFinal NLU Output:")
# print(f"JobID: {result['job_id']}")
# print(f"Constraints: {result['constraints']}")
# print(f"Priority: {result['priority']}")
# print(f"Intent: {result['intent']}")
# print(result["messages"][-1].content)

# print(result['technician_names'])
# print(result['travel_times'])
print(result['top3_technicians'])