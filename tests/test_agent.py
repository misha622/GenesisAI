import pytest, pandas as pd, numpy as np, tempfile, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.agent import GenesisAgent, Intent, AgentResponse
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry

class TestGenesisAgent:
    @pytest.fixture
    def agent(self):
        d = tempfile.mkdtemp()
        return GenesisAgent(profiler=DatasetProfiler(), trainer=DatasetTrainer(model_dir=d), registry=ModelRegistry(storage_dir=d))

    @pytest.fixture
    def sample_csv(self):
        df = pd.DataFrame({"age": [25,30,35,40,45,28,33,38,43,48], "salary": [50,60,75,90,110,55,70,85,100,120], "churn": [0,1,0,0,1,0,1,0,0,1]})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f: df.to_csv(f, index=False); p = f.name
        yield p; os.unlink(p)

    def test_agent_creation(self, agent):
        assert agent is not None

    def test_process_analyze(self, agent, sample_csv):
        resp = agent.process(f"проанализируй {sample_csv}")
        assert resp.success

    def test_process_train(self, agent, sample_csv):
        agent.process(f"проанализируй {sample_csv}")
        resp = agent.process("обучи xgboost на churn")
        assert resp.success

    def test_process_find_best(self, agent, sample_csv):
        agent.process(f"проанализируй {sample_csv}")
        agent.process("обучи xgboost на churn")
        resp = agent.process("покажи лучшую модель")
        assert isinstance(resp, AgentResponse)

    def test_process_search(self, agent):
        resp = agent.process("найди датасет для churn prediction")
        assert isinstance(resp, AgentResponse)

    def test_process_unknown(self, agent):
        resp = agent.process("какая сегодня погода")
        assert isinstance(resp, AgentResponse)
