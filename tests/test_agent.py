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

    def test_detect_intent_help(self, agent): assert agent.detect_intent("помощь") == Intent.HELP
    def test_detect_intent_analyze(self, agent): assert agent.detect_intent("проанализируй данные") == Intent.ANALYZE
    def test_detect_intent_train(self, agent): assert agent.detect_intent("обучи модель") == Intent.TRAIN
    def test_detect_intent_find_best(self, agent): assert agent.detect_intent("покажи лучшую модель") == Intent.FIND_BEST
    def test_detect_intent_list(self, agent): assert agent.detect_intent("покажи список моделей") == Intent.LIST_MODELS
    def test_detect_intent_dataset_search(self, agent): assert agent.detect_intent("найди датасет для churn") == Intent.DATASET_SEARCH
    def test_detect_intent_install(self, agent): assert agent.detect_intent("установи датасет 1") == Intent.INSTALL_DATASET
    def test_handle_help(self, agent): assert agent.process("помощь").success
    def test_handle_analyze(self, agent, sample_csv): assert agent.process(f"проанализируй {sample_csv}").success
    def test_handle_train(self, agent, sample_csv): assert agent.process(f"обучи модель на {sample_csv} целевая churn xgboost").success
    def test_handle_list_after_train(self, agent, sample_csv): agent.process(f"обучи модель на {sample_csv} целевая churn xgboost"); assert agent.process("покажи список моделей").success
    def test_handle_find_best_after_train(self, agent, sample_csv): agent.process(f"обучи модель на {sample_csv} целевая churn xgboost"); assert agent.process("покажи лучшую модель").success
    def test_handle_dataset_search(self, agent): resp = agent.process("найди датасет для churn prediction"); assert resp.success; assert len(resp.data.get("datasets", [])) > 0
    def test_extract_target_column(self, agent): assert agent.extract_target_column("целевая колонка churn") == "churn"
    def test_extract_model_type(self, agent): assert agent.extract_model_type("обучи xgboost") == "xgboost"
    def test_unknown_intent(self, agent): assert not agent.process("какая сегодня погода").success

