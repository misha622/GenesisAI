import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

app = FastAPI(title="Genesis AI", version="0.9.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

_agents: Dict[str, Any] = {}
def get_agent(conversation_id: str = "default"):
    if conversation_id not in _agents:
        from app.core.agent import GenesisAgent
        _agents[conversation_id] = GenesisAgent()
    return _agents[conversation_id]

class MessageRequest(BaseModel):
    message: str; conversation_id: Optional[str] = None
    projects: Optional[List[Dict]] = None

class PublishRequest(BaseModel):
    author: str; model_id: str; title: str; description: str
    task: str; model_type: str
    metrics: Dict[str, float] = {}; price_per_call: float = 0.01

@app.get("/")
async def root(): return FileResponse("static/index.html")

@app.get("/api/agent/conversations")
async def agent_conversations():
    from app.db.database import GenesisDB
    return {"conversations": GenesisDB().list_chats()}

@app.get("/api/agent/chat/{conversation_id}")
async def agent_chat(conversation_id: str):
    from app.db.database import GenesisDB
    return {"messages": GenesisDB().get_messages(conversation_id)}

@app.post("/api/agent/message")
async def agent_message(req: MessageRequest):
    cid = req.conversation_id or "default"
    agent = get_agent(cid)
    response = agent.process(req.message, cid, req.projects)
    return {"response": response.message, "conversation_id": cid, "data": response.data}

@app.post("/api/agent/chat/{conversation_id}/rename")
async def rename_chat(conversation_id: str, data: Dict[str, Any]):
    from app.db.database import GenesisDB
    GenesisDB().rename_chat(conversation_id, data.get('title', conversation_id))
    return {"status": "renamed"}

@app.delete("/api/agent/chat/{conversation_id}")
async def delete_chat(conversation_id: str):
    from app.db.database import GenesisDB
    GenesisDB().delete_chat(conversation_id)
    return {"status": "deleted"}

@app.get("/api/marketplace/listings")
async def marketplace_listings(task: Optional[str] = None):
    from app.core.marketplace import Marketplace
    return {"listings": Marketplace().search(task=task) if task else Marketplace().search()}

@app.post("/api/marketplace/publish")
async def marketplace_publish(req: PublishRequest):
    from app.core.marketplace import Marketplace
    lid = Marketplace().publish(author=req.author, model_id=req.model_id, title=req.title, description=req.description, task=req.task, model_type=req.model_type, metrics=req.metrics, price_per_call=req.price_per_call)
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
    from app.db.database import GenesisDB
    return {"facts": GenesisDB().get_facts()}

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
    return {"query": q, "results": [{"title": d.title, "url": d.url, "source": d.source} for d in DatasetFinder().search(q)]}

@app.get("/api/projects")
async def get_projects():
    from app.db.database import GenesisDB
    return {"projects": GenesisDB().get_projects()}

@app.post("/api/projects")
async def save_projects(data: Dict[str, Any]):
    from app.db.database import GenesisDB
    db = GenesisDB()
    for p in data.get("projects", []):
        db.add_project(p.get("name",""), p.get("desc",""), p.get("task","other"), p.get("icon","🤖"))
    return {"status": "saved"}

@app.get("/api/keys")
async def list_keys(user_name: str = None):
    from app.db.database import GenesisDB
    return {"keys": GenesisDB().list_api_keys(user_name)}

@app.post("/api/keys/create")
async def create_key(data: dict):
    from app.db.database import GenesisDB
    key = GenesisDB().create_api_key(data.get("user_name", "anonymous"), data.get("calls_limit", 100))
    return key

@app.delete("/api/keys/{api_key}")
async def delete_key(api_key: str):
    from app.db.database import GenesisDB
    GenesisDB().delete_api_key(api_key)
    return {"status": "deleted"}

@app.get("/api/usage")
async def usage_stats(api_key: str = None):
    from app.db.database import GenesisDB
    return {"usage": GenesisDB().get_usage_stats(api_key)}

@app.post("/api/models/{model_id}/predict")
async def model_predict(model_id: str, api_key: str, data: dict = None):
    from app.db.database import GenesisDB
    db = GenesisDB()
    if not db.validate_api_key(api_key):
        return {"error": "Invalid API key"}, 401
    if not db.use_api_key(api_key, model_id):
        return {"error": "API call limit exceeded"}, 429
    from app.core.engine.registry import ModelRegistry
    reg = ModelRegistry()
    try:
        model = reg.get_model(model_id)
        features = data.get("features", []) if data else []
        if features:
            import numpy as np
            prediction = model.predict(np.array(features).reshape(1, -1))
            return {"prediction": prediction.tolist(), "model_id": model_id}
        return {"message": "Model loaded", "model_id": model_id, "metrics": reg.get_record(model_id).get("metrics", {})}
    except Exception as e:
        return {"error": str(e)}, 500

@app.post("/api/codegen/generate")
async def generate_code(data: dict):
    from app.core.codegen.generator import CodeGenerator
    from app.db.database import GenesisDB
    gen = CodeGenerator(GenesisDB())
    description = data.get("description", "")
    code = gen.generate_from_description(description)
    path = gen.save_code(code)
    return {"code_path": path, "code": code}

@app.get("/api/codegen/list")
async def list_generated():
    import os, glob
    files = glob.glob("generated/*.py")
    return {"files": [{"name": os.path.basename(f), "path": f} for f in files]}

@app.get("/health")
async def health(): return {"status": "healthy", "version": "0.9.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
