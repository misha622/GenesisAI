import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, Request
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.staticfiles import StaticFiles
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

app = FastAPI(title="Genesis AI", version="0.1.0")

# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Статика
app.mount("/static", StaticFiles(directory="static"), name="static")

# Модели запросов
class MessageRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class PublishRequest(BaseModel):
    author: str
    model_id: str
    title: str
    description: str
    task: str
    model_type: str
    metrics: Dict[str, float] = {}
    price_per_call: float = 0.01

# ---- Главная ----
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ---- Agent API ----
@app.get("/api/agent/conversations")
async def agent_conversations():
    from app.core.agent import GenesisAgent
    agent = GenesisAgent()
    return {"conversations": []}

@app.get("/api/agent/chat/{conversation_id}")
async def agent_chat(conversation_id: str):
    return {"messages": []}

@app.post("/api/agent/message")
async def agent_message(req: MessageRequest):
    from app.core.agent import GenesisAgent
    agent = GenesisAgent()
    response = agent.process(req.message)
    return {
        "response": response.message,
        "conversation_id": req.conversation_id or "new",
        "data": response.data,
    }

# ---- Marketplace API ----
@app.get("/api/marketplace/listings")
async def marketplace_listings(task: Optional[str] = None):
    from app.core.marketplace import Marketplace
    mp = Marketplace()
    results = mp.search(task=task) if task else mp.search()
    return {"listings": results}

@app.post("/api/marketplace/publish")
async def marketplace_publish(req: PublishRequest):
    from app.core.marketplace import Marketplace
    mp = Marketplace()
    lid = mp.publish(
        author=req.author,
        model_id=req.model_id,
        title=req.title,
        description=req.description,
        task=req.task,
        model_type=req.model_type,
        metrics=req.metrics,
        price_per_call=req.price_per_call,
    )
    return {"listing_id": lid}

@app.get("/api/marketplace/stats")
async def marketplace_stats():
    from app.core.marketplace import Marketplace
    mp = Marketplace()
    return mp.get_platform_stats()

# ---- Registry API ----
@app.get("/api/registry/models")
async def registry_models(task: Optional[str] = None):
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    models = reg.list_models(task=task)
    return {"models": models}

@app.get("/api/registry/models/{model_id}")
async def registry_model(model_id: str):
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    return reg.get_record(model_id)

@app.get("/api/registry/best")
async def registry_best(task: str = "binary_classification", metric: str = "accuracy"):
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    best = reg.find_best(task, metric)
    return {"best": best}

@app.get("/api/registry/stats")
async def registry_stats():
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    return reg.get_stats()

# ---- Memory API ----
@app.get("/api/memory/facts")
async def memory_facts():
    from app.core.memory.graph import KnowledgeGraph
    from app.core.memory.episodic import FactExtractor
    kg = KnowledgeGraph()
    facts = kg.get_facts_about("user")
    return {"facts": facts}

@app.get("/api/memory/persona")
async def memory_persona():
    from app.core.memory.graph import KnowledgeGraph
    from app.core.memory.persona import PersonaGenerator
    kg = KnowledgeGraph()
    pg = PersonaGenerator(kg)
    persona = pg.generate()
    return {"persona": persona}

@app.get("/api/memory/graph")
async def memory_graph():
    from app.core.memory.graph import KnowledgeGraph
    kg = KnowledgeGraph()
    return kg.to_dict()

# ---- Health ----
@app.get("/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

