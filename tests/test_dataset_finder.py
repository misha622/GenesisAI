import pytest, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.core.engine.dataset_finder import DatasetFinder, DatasetInfo

class TestDatasetFinder:
    @pytest.fixture
    def finder(self): return DatasetFinder()

    def test_search_churn(self, finder):
        results = finder.search("churn prediction")
        assert len(results) > 0
        assert any("churn" in d.title.lower() or "churn" in d.tags for d in results)

    def test_search_regression(self, finder):
        results = finder.search("housing prices", task="regression")
        assert len(results) > 0
        assert all(d.task == "regression" for d in results)

    def test_search_nlp(self, finder):
        results = finder.search("sentiment analysis")
        assert len(results) > 0

    def test_search_nonexistent(self, finder):
        results = finder.search("xyzabc123")
        assert len(results) == 0

    def test_format_for_chat(self, finder):
        results = finder.search("churn")
        formatted = finder.format_for_chat(results)
        assert "Telco" in formatted or "Churn" in formatted
        assert "⭐" in formatted

    def test_get_sources(self, finder):
        sources = finder.get_sources_summary()
        assert "kaggle" in sources
