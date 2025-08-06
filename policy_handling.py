import pandas as pd # You'll need this if you ever use external data again.
import json         # Needed for parsing JSON output from LLM
import re           # Needed for robust JSON parsing from LLM response

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool # Not used in this NLU-only snippet, but good to keep if you expand
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Union, Annotated, Sequence
from langgraph.prebuilt import ToolNode # Not used in this NLU-only snippet, but good to keep if you expand
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
load_dotenv()


DEFAULT_SCHEDULING_PRIORITY = ["ASAP", "skill match", "proximity"]
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
print(df.columns)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 
    job_id: str       
    constraints: List[str] 
    priority: List[str] 
    intent: str         


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


graph = StateGraph(AgentState)

graph.add_node("nlu", extract_nlu_info)
graph.add_edge(START, "nlu")
graph.add_edge("nlu", END)

app = graph.compile()

result1 = app.invoke({"messages": [HumanMessage(content="Can you assign JOB004 urgently to someone nearby, skill is more important?")]})
print("\nFinal NLU Output:")
print(f"JobID: {result1['job_id']}")
print(f"Constraints: {result1['constraints']}")
print(f"Priority: {result1['priority']}")
print(f"Intent: {result1['intent']}")