import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(title="Genesis AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

# Singleton agent
_agent = None
def get_agent():
    global _agent
    if _agent is None:
        from app.core.agent import GenesisAgent
        _agent = GenesisAgent()
    return _agent

class MessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class PublishRequest(BaseModel):
    author: str; model_id: str; title: str; description: str
    task: str; model_type: str
    metrics: Dict[str, float] = {}
    price_per_call: float = 0.01

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/api/agent/conversations")
async def agent_conversations():
    return {"conversations": []}

@app.get("/api/agent/chat/{conversation_id}")
async def agent_chat(conversation_id: str):
    return {"messages": []}

@app.post("/api/agent/message")
async def agent_message(req: MessageRequest):
    agent = get_agent()
    response = agent.process(req.message)
    return {"response": response.message, "conversation_id": req.conversation_id or "new", "data": response.data}

@app.get("/api/marketplace/listings")
async def marketplace_listings(task: Optional[str] = None):
    from app.core.marketplace import Marketplace
    mp = Marketplace()
    return {"listings": mp.search(task=task) if task else mp.search()}

@app.post("/api/marketplace/publish")
async def marketplace_publish(req: PublishRequest):
    from app.core.marketplace import Marketplace
    mp = Marketplace()
    lid = mp.publish(author=req.author, model_id=req.model_id, title=req.title, description=req.description, task=req.task, model_type=req.model_type, metrics=req.metrics, price_per_call=req.price_per_call)
    return {"listing_id": lid}

@app.get("/api/marketplace/stats")
async def marketplace_stats():
    from app.core.marketplace import Marketplace
    return Marketplace().get_platform_stats()

@app.get("/api/registry/models")
async def registry_models(task: Optional[str] = None):
    from app.core.engine.registry import ModelRegistry
    return {"models": ModelRegistry().list_models(task=task)}

@app.get("/api/registry/models/{model_id}")
async def registry_model(model_id: str):
    from app.core.engine.registry import ModelRegistry
    return ModelRegistry().get_record(model_id)

@app.get("/api/registry/best")
async def registry_best(task: str = "binary_classification", metric: str = "accuracy"):
    from app.core.engine.registry import ModelRegistry
    return {"best": ModelRegistry().find_best(task, metric)}

@app.get("/api/registry/stats")
async def registry_stats():
    from app.core.engine.registry import ModelRegistry
    return ModelRegistry().get_stats()

@app.get("/api/memory/facts")
async def memory_facts():
    from app.core.memory.graph import KnowledgeGraph
    return {"facts": KnowledgeGraph().get_facts_about("user")}

@app.get("/api/memory/persona")
async def memory_persona():
    from app.core.memory.graph import KnowledgeGraph
    from app.core.memory.persona import PersonaGenerator
    return {"persona": PersonaGenerator(KnowledgeGraph()).generate()}

@app.get("/api/memory/graph")
async def memory_graph():
    from app.core.memory.graph import KnowledgeGraph
    return KnowledgeGraph().to_dict()

@app.get("/api/datasets/search")
async def dataset_search(q: str):
    from app.core.engine.dataset_finder import DatasetFinder
    results = DatasetFinder().search(q)
    return {"query": q, "results": [{"title": d.title, "url": d.url, "source": d.source} for d in results]}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.4.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
