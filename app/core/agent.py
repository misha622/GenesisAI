"""Genesis Agent - dialog AI assistant with dataset search."""

import json, os, re, pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry

class Intent:
    ANALYZE = "analyze"
    TRAIN = "train"
    FIND_BEST = "find_best"
    LIST_MODELS = "list_models"
    COMPARE = "compare"
    PREDICT = "predict"
    DATASET_SEARCH = "dataset_search"
    INSTALL_DATASET = "install_dataset"
    HELP = "help"
    UNKNOWN = "unknown"

class AgentResponse:
    def __init__(self, message: str, data: Optional[Dict] = None, success: bool = True):
        self.message = message; self.data = data or {}; self.success = success; self.timestamp = datetime.now().isoformat()

class GenesisAgent:
    INTENT_KEYWORDS = {
        Intent.ANALYZE: ["анализ","профиль","посмотри","изучи","покажи данные","колонки","столбцы","пропуски","analyze","profile","explore","inspect","columns"],
        Intent.TRAIN: ["обучи","тренируй","построй модель","обучение","train","fit","build model","xgboost","logistic"],
        Intent.FIND_BEST: ["найди лучшую","лучшая модель","best model","топ модель","самая точная","покажи лучшую"],
        Intent.LIST_MODELS: ["список моделей","все модели","какие модели","list models","покажи модели","мои модели"],
        Intent.COMPARE: ["сравни","сравнение","compare","что лучше"],
        Intent.PREDICT: ["предскажи","прогноз","predict","получить результат"],
        Intent.DATASET_SEARCH: ["найди датасет","поищи данные","dataset","нужны данные","хочу создать ии","создать ии","собери данные","какие датасеты","подбери датасет","где взять данные"],
        Intent.INSTALL_DATASET: ["установи","скачай","загрузи датасет","install","download","под номером","выбираю","давай этот"],
        Intent.HELP: ["помощь","что ты умеешь","команды","help","справка","начнём","привет","hi","hello"],
    }

    def __init__(self, profiler=None, trainer=None, registry=None):
        self.profiler = profiler or DatasetProfiler()
        self.trainer = trainer or DatasetTrainer()
        self.registry = registry or ModelRegistry()
        self.current_dataset: Optional[str] = None
        self.last_search_results: List = []
        self.conversation_history: List[Dict] = []

    def detect_intent(self, message: str) -> str:
        message_lower = message.lower()
        # Priority checks`n        if "лучшую модель" in message_lower or "лучшая модель" in message_lower or "best model" in message_lower: return Intent.FIND_BEST
        if "датасет" in message_lower or "dataset" in message_lower:
            if any(w in message_lower for w in ["установи","скачай","install","download","номер","выбираю"]):
                return Intent.INSTALL_DATASET
            return Intent.DATASET_SEARCH
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > 0: scores[intent] = score
        if not scores: return Intent.UNKNOWN
        return max(scores, key=scores.get)

    def extract_file_path(self, message: str) -> Optional[str]:
        patterns = [r'[\w\\/:.-]+\.(?:csv|json|parquet|xlsx?)', r'["\']([\w\\/:.-]+\.(?:csv|json|parquet|xlsx?))["\']']
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                path = match.group(1) if match.lastindex else match.group(0)
                if os.path.exists(path): return path
        return self.current_dataset

    def extract_target_column(self, message: str) -> Optional[str]:
        patterns = [r'(?:целевая|таргет|target|предска(?:жи|зать)|predict)\s+(?:колонка|столбец|переменная)?\s*[=:]*\s*["\']?(\w+)["\']?', r'target[=:]\s*["\']?(\w+)["\']?']
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match: return match.group(1)
        return None

    def extract_model_type(self, message: str) -> str:
        m = message.lower()
        if "логистическ" in m or "logistic" in m: return "logistic"
        if "случайн" in m or "random" in m or "лес" in m: return "random_forest"
        if "xgboost" in m or "xgb" in m: return "xgboost"
        if "линейн" in m or "linear" in m: return "linear"
        return "xgboost"

    def process(self, message: str) -> AgentResponse:
        self.conversation_history.append({"role":"user","message":message,"timestamp":datetime.now().isoformat()})
        intent = self.detect_intent(message)
        handlers = {
            Intent.HELP: self._handle_help, Intent.ANALYZE: self._handle_analyze,
            Intent.TRAIN: self._handle_train, Intent.FIND_BEST: self._handle_find_best,
            Intent.LIST_MODELS: self._handle_list_models, Intent.COMPARE: self._handle_compare,
            Intent.DATASET_SEARCH: self._handle_dataset_search, Intent.INSTALL_DATASET: self._handle_install_dataset,
        }
        if intent in handlers: return handlers[intent](message)
        return self._handle_unknown(message)

    def _handle_help(self, message="") -> AgentResponse:
        return AgentResponse(message="Я Genesis AI Agent. Умею:\n📊 Анализировать датасеты\n🤖 Обучать модели\n🔍 Искать лучшую модель\n📋 Список моделей\n🔎 Искать датасеты\n📥 Устанавливать датасеты\n\nЧто делать?")

    def _handle_analyze(self, message: str) -> AgentResponse:
        fp = self.extract_file_path(message)
        if not fp: return AgentResponse(message="Укажите путь к файлу", success=False)
        try:
            self.profiler.run_full_profile(fp); self.current_dataset = fp
            s = self.profiler.get_summary()
            return AgentResponse(message=f"Анализ {s['file_info']['name']}:\n• Строк: {s['file_info']['rows']}\n• Колонок: {s['file_info']['columns']}\n• Задача: {s['suggested_ml_task']['task']}\n• Таргет: {s['suggested_ml_task']['target_column']}\n\nОбучить модель?", data=s)
        except Exception as e: return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _handle_train(self, message: str) -> AgentResponse:
        fp = self.extract_file_path(message)
        if not fp: return AgentResponse(message="Укажите датасет", success=False)
        tc = self.extract_target_column(message); mt = self.extract_model_type(message)
        try:
            if self.current_dataset != fp: self.profiler.run_full_profile(fp); self.current_dataset = fp
            ps = self.profiler.get_summary(); task = ps["suggested_ml_task"]["task"]
            if not tc: tc = ps["suggested_ml_task"]["target_column"]
            result = self.trainer.train(df=self.profiler.df, target_col=tc, task=task, model_type=mt)
            mid = self.registry.register(model=pickle.load(open(result.model_path,"rb")), metrics=result.metrics, task=task, model_type=mt, features=self.profiler.profile.feature_columns, target=tc)
            s = self.trainer.get_summary()
            return AgentResponse(message=f"Модель обучена!\n• ID: {mid}\n• Тип: {mt}\n• Метрики: {s['metrics']}\n• Признаки: {list(s['feature_importance'].keys())[:5]}", data={"model_id":mid,**s})
        except Exception as e: return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _handle_find_best(self, message: str) -> AgentResponse:
        m = message.lower(); task = "binary_classification"; metric = "accuracy"
        if "регресс" in m or "regression" in m: task = "regression"; metric = "r2"
        best = self.registry.find_best(task, metric)
        if not best: return AgentResponse(message=f"Нет моделей для '{task}'", success=False)
        return AgentResponse(message=f"Лучшая для {task}:\n• ID: {best['model_id']}\n• Тип: {best['model_type']}\n• Метрики: {best['metrics']}", data=best)

    def _handle_list_models(self, message: str) -> AgentResponse:
        models = self.registry.list_models()
        if not models: return AgentResponse(message="Реестр пуст", success=False)
        return AgentResponse(message="\n".join([f"• {r['model_id'][:50]}... — {r['task']}" for r in models[:5]]), data={"models":models})

    def _handle_compare(self, message: str) -> AgentResponse:
        ids = re.findall(r'(\w+_\w+_\d{8}_\d{6}(?:_\d+)?)', message)
        if len(ids) < 2: return AgentResponse(message="Укажите ID моделей", success=False)
        df = self.registry.compare_models(ids)
        return AgentResponse(message=f"Сравнение:\n{df.to_string()}", data={"comparison":df.to_dict()})

    def _handle_dataset_search(self, message: str) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        finder = DatasetFinder()
        query = message
        for prefix in ["хочу создать ии","создать ии","найди датасет для","найди датасет","поищи данные","подбери датасет"]:
            if prefix in message.lower(): query = message.lower().split(prefix)[-1].strip(); break
        datasets = finder.search(query, max_results=5)
        self.last_search_results = datasets
        formatted = finder.format_for_chat(datasets)
        return AgentResponse(message=formatted, data={"datasets":[{"title":d.title,"url":d.url,"source":d.source} for d in datasets]})

    def _handle_install_dataset(self, message: str) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        finder = DatasetFinder()
        match = re.search(r'(\d+)', message)
        if not match: return AgentResponse(message="Напишите номер датасета. Например: «установи 1»", success=False)
        num = int(match.group(1))
        if not self.last_search_results: self.last_search_results = finder.search("", max_results=5)
        if num < 1 or num > len(self.last_search_results): return AgentResponse(message=f"Номер от 1 до {len(self.last_search_results)}", success=False)
        chosen = self.last_search_results[num-1]
        try:
            path = finder.install_dataset(chosen.url); self.current_dataset = path
            self.profiler.run_full_profile(path); s = self.profiler.get_summary()
            return AgentResponse(message=f"Установил: **{chosen.title}**\nПуть: {path}\n\nАнализ:\n• Строк: {s['file_info']['rows']}\n• Колонок: {s['file_info']['columns']}\n• Задача: {s['suggested_ml_task']['task']}\n\nОбучить модель?", data={"dataset_path":path,"profile":s})
        except Exception as e: return AgentResponse(message=f"Не удалось: {e}\nОткройте вручную:\n{chosen.url}", success=False)

    def _handle_unknown(self, message: str) -> AgentResponse:
        return AgentResponse(message=f"Не понял: «{message}»\nПопробуйте: «анализ», «обучи», «найди датасет», «помощь»", success=False)



