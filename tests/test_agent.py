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
        return GenesisAgent(
            profiler=DatasetProfiler(),
            trainer=DatasetTrainer(model_dir=d),
            registry=ModelRegistry(storage_dir=d),
        )

    @pytest.fixture
    def sample_csv(self):
        df = pd.DataFrame({
            "age": [25,30,35,40,45,28,33,38,43,48],
            "salary": [50,60,75,90,110,55,70,85,100,120],
            "churn": [0,1,0,0,1,0,1,0,0,1],
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            p = f.name
        yield p
        os.unlink(p)

    def test_detect_intent_help(self, agent):
        assert agent.detect_intent("помощь") == Intent.HELP
        assert agent.detect_intent("что ты умеешь") == Intent.HELP
        assert agent.detect_intent("привет") == Intent.HELP

    def test_detect_intent_analyze(self, agent):
        assert agent.detect_intent("проанализируй данные") == Intent.ANALYZE
        assert agent.detect_intent("покажи колонки датасета") == Intent.ANALYZE

    def test_detect_intent_train(self, agent):
        assert agent.detect_intent("обучи модель на xgboost") == Intent.TRAIN
        assert agent.detect_intent("построй модель") == Intent.TRAIN

    def test_detect_intent_find_best(self, agent):
        assert agent.detect_intent("найди лучшую модель") == Intent.FIND_BEST

    def test_detect_intent_list(self, agent):
        assert agent.detect_intent("покажи список моделей") == Intent.LIST_MODELS

    def test_detect_intent_compare(self, agent):
        assert agent.detect_intent("сравни эти модели") == Intent.COMPARE

    def test_handle_help(self, agent):
        resp = agent.process("помощь")
        assert resp.success
        assert "анализировать" in resp.message.lower() or "analyze" in resp.message.lower()

    def test_handle_analyze(self, agent, sample_csv):
        resp = agent.process(f"проанализируй {sample_csv}")
        assert resp.success
        assert "строк" in resp.message.lower() or "rows" in resp.message.lower()

    def test_handle_train(self, agent, sample_csv):
        resp = agent.process(f"обучи модель на {sample_csv} целевая churn xgboost")
        assert resp.success
        assert "модель обучена" in resp.message.lower() or "model" in resp.message.lower()

    def test_handle_list_after_train(self, agent, sample_csv):
        agent.process(f"обучи модель на {sample_csv} целевая churn xgboost")
        resp = agent.process("покажи список моделей")
        assert resp.success

    def test_handle_find_best_after_train(self, agent, sample_csv):
        agent.process(f"обучи модель на {sample_csv} целевая churn xgboost")
        resp = agent.process("найди лучшую модель")
        assert resp.success

    def test_extract_target_column(self, agent):
        assert agent.extract_target_column("целевая колонка churn") == "churn"
        assert agent.extract_target_column("таргет age") == "age"
        assert agent.extract_target_column("предскажи salary") == "salary"

    def test_extract_model_type(self, agent):
        assert agent.extract_model_type("обучи xgboost") == "xgboost"
        assert agent.extract_model_type("логистическая регрессия") == "logistic"
        assert agent.extract_model_type("случайный лес") == "random_forest"

    def test_unknown_intent(self, agent):
        resp = agent.process("какая сегодня погода")
        assert not resp.success
