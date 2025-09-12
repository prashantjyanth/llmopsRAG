# builder = WorkflowBuilder()

# # Test the workflow with a sample input
# thread_id = "test_conversation"
# print(f"\n🧪 Testing conversation memory (Thread: {thread_id})...")
# # First interaction
# while True:
#     user_input = input("You: ").strip()
#     if user_input.lower() in ['quit', 'exit']:
#         print("👋 Goodbye!")
#         break
#     response = builder.run(user_input, thread_id)
#     print(f"Response: {response['messages']}")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.workflow_builder import WorkflowBuilder
from prometheus_client import Gauge
from langgraph.checkpoint.memory import InMemorySaver


ORCH_HEALTH = Gauge("orchestrator_health", "Health status of orchestrator service")

app = FastAPI()
builder: WorkflowBuilder | None = None   # will init on startup

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class AgentRunRequest(BaseModel):
    agent_name: str
    message: str
    thread_id: str = "default"

@app.post("/orch")
def chat(request: ChatRequest):
    try:
        result = builder.run(request.message, request.thread_id)
        return {"messages": result.get("messages", []), "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.on_event("startup")
def startup_event():
    global builder
    ORCH_HEALTH.set(1)
    builder = WorkflowBuilder(checkpointer=InMemorySaver())

@app.on_event("shutdown")
def shutdown_event():
    ORCH_HEALTH.set(0)
    builder = None

@app.get("/health")
def health():
    ORCH_HEALTH.set(1)  # ✅ mark healthy
    return {"status": "healthy"}

@app.get("/agents")
def list_agents():
    try:
        return {"agents": list(builder.agents.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/run")
def run_agent(request: AgentRunRequest):
    try:
        agent = builder.agents.get(request.agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        # Assuming agent.run exists and works like builder.run
        result = agent.run(request.message, request.thread_id)
        return {"result": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))