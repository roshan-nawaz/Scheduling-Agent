import uvicorn

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import app 
from langchain_core.messages import HumanMessage

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods = ["*"],
)

class ScheduleRequest(BaseModel):
    query:str

@api.get("/")
def root():
    return {"message": "FastAPI is running!"}


@api.post("/schedule")
async def schedule_agent(req : ScheduleRequest):
    result = app.invoke({"messages":[HumanMessage(content = req.query)]})

    return {
        "job_id": result["job_id"],
        "constraints": result["constraints"],
        "priority": result["priority"],
        "intent": result["intent"],
        "final_message": result["messages"][-1].content,
        "technician_names": result["technician_names"],
        "travel_times": result["travel_times"],
        "top3_technicians": result["top3_technicians"]
    }


if __name__ == "__main__":

    uvicorn.run("main:api", host = "127.0.0.1", port = 8000, reload = True)

