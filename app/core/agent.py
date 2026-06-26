"""Genesis Agent - REAL LLM-powered AI assistant."""

import json, os, re, pickle, numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry

class Intent:
    ANALYZE = "analyze"; TRAIN = "train"; FIND_BEST = "find_best"
    LIST_MODELS = "list_models"; COMPARE = "compare"; DATASET_SEARCH = "dataset_search"
    INSTALL_DATASET = "install_dataset"; HELP = "help"; UNKNOWN = "unknown"

class Intent:
    ANALYZE = "analyze"; TRAIN = "train"; FIND_BEST = "find_best"
    LIST_MODELS = "list_models"; COMPARE = "compare"; DATASET_SEARCH = "dataset_search"
    INSTALL_DATASET = "install_dataset"; HELP = "help"; UNKNOWN = "unknown"

class Intent:
    ANALYZE = "analyze"; TRAIN = "train"; FIND_BEST = "find_best"
    LIST_MODELS = "list_models"; COMPARE = "compare"; DATASET_SEARCH = "dataset_search"
    INSTALL_DATASET = "install_dataset"; HELP = "help"; UNKNOWN = "unknown"

class AgentResponse:
    def __init__(self, message: str, data=None, success: bool = True):
        self.message = message; self.data = data or {}; self.success = success
        self.timestamp = datetime.now().isoformat()

def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

class GenesisAgent:
    SYSTEM_PROMPT = """Ты Genesis AI Agent - программа для AutoML. Ты НЕ разговариваешь с пользователем. Ты ТОЛЬКО вызываешь инструменты через JSON.

Инструменты:
- analyze_dataset(path) - анализ CSV файла
- train_model(target_column, model_type) - обучение модели
- find_best_model(task) - лучшая модель
- search_datasets(query) - поиск датасетов
- list_models() - список моделей

ПРАВИЛА:
1. НИКОГДА не пиши текст вне JSON
2. ВСЕГДА отвечай ТОЛЬКО {"tool": "...", "args": {...}, "message": "..."}
3. Если пользователь просит проанализировать - вызывай analyze_dataset
4. Если обучить - train_model
5. Если лучшую модель - find_best_model
6. Не рассуждай, не объясняй, не описывай данные. ТОЛЬКО JSON."""

    def __init__(self, profiler=None, trainer=None, registry=None):
        self.profiler = profiler or DatasetProfiler()
        self.trainer = trainer or DatasetTrainer()
        self.registry = registry or ModelRegistry()
        self.current_dataset = None
        self.last_search_results = []
        self.conversation_history = []
        self._init_llm()

    def _init_llm(self):
        try:
            from llama_cpp import Llama
            model_path = "models/qwen2.5-7b-instruct-q4_k_m.gguf"
            if os.path.exists(model_path):
                self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)
                print("[OK] LLM loaded")
            else:
                self.llm = None
        except Exception as e:
            print(f"[WARN] LLM: {e}")
            self.llm = None

    def _call_llm(self, msg: str) -> Dict:
        if not self.llm:
            return {"tool": None, "message": self._fallback(msg)}
        prompt = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        try:
            output = self.llm(prompt, max_tokens=256, stop=["<|im_end|>"], temperature=0.1)
            text = output["choices"][0]["text"].strip()
            if text.startswith("{"):
                return json.loads(text)
            return {"tool": None, "message": text}
        except:
            return {"tool": None, "message": self._fallback(msg)}

    def _fallback(self, msg: str) -> str:
        m = msg.lower()
        if any(w in m for w in ["анализ","проанализируй"]):
            fp = self._extract_path(msg)
            if fp: return self._do_analyze(fp).message
        return "Я понимаю естественный язык. Спросите про ML!"

    def process(self, message: str) -> AgentResponse:
        self.conversation_history.append({"role":"user","message":message})
        if self.llm:
            result = self._call_llm(message)
            tool = result.get("tool")
            args = result.get("args", {})
            if tool == "analyze_dataset":
                path = args.get("path", self._extract_path(message))
                if path: return self._do_analyze(path)
            elif tool == "train_model":
                return self._do_train(message, args)
            elif tool == "find_best_model":
                return self._do_find_best(args.get("task", "binary_classification"))
            elif tool == "list_models":
                return self._do_list_models()
            elif tool == "search_datasets":
                return self._do_search(message)
            elif tool == "install_dataset":
                return self._do_install(str(args.get("number", "1")))
            return AgentResponse(message=result.get("message", self._fallback(message)))
        return AgentResponse(message=self._fallback(message))

    def _extract_path(self, message: str):
        patterns = [r'([\w\\/:.-]+\.(?:csv|json|parquet|xlsx?))']
        for p in patterns:
            m = re.search(p, message, re.IGNORECASE)
            if m:
                path = m.group(1)
                if os.path.exists(path): return path
        return self.current_dataset

    def _do_analyze(self, fp: str) -> AgentResponse:
        try:
            self.profiler.run_full_profile(fp); self.current_dataset = fp
            s = self.profiler.get_summary()
            return AgentResponse(message=f"Анализ {s['file_info']['name']}:\n• Строк: {s['file_info']['rows']}\n• Колонок: {s['file_info']['columns']}\n• Задача: {s['suggested_ml_task']['task']}\n• Таргет: {s['suggested_ml_task']['target_column']}\n\nОбучить модель?", data=_to_native(s))
        except Exception as e: return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _do_train(self, message: str, args: Dict = None) -> AgentResponse:
        fp = self._extract_path(message)
        if not fp: return AgentResponse(message="Загрузите датасет: «проанализируй datasets/churn_demo.csv»", success=False)
        tc = (args.get("target") or args.get("target_column")) if args else None
        mt = (args.get("model_type") or args.get("model_type")) if args else "xgboost"
        if not tc:
            for p in [r'(?:целевая|таргет|target|на)\s+(\w+)']:
                m = re.search(p, message, re.IGNORECASE)
                if m: tc = m.group(1); break
        try:
            if self.current_dataset != fp: self.profiler.run_full_profile(fp); self.current_dataset = fp
            ps = self.profiler.get_summary(); task = ps["suggested_ml_task"]["task"]
            if not tc: tc = ps["suggested_ml_task"]["target_column"]
            result = self.trainer.train(df=self.profiler.df, target_col=tc, task=task, model_type=mt)
            mid = self.registry.register(model=pickle.load(open(result.model_path,"rb")), metrics=result.metrics, task=task, model_type=mt, features=self.profiler.profile.feature_columns, target=tc)
            s = self.trainer.get_summary()
            return AgentResponse(message=f"Модель обучена!\n• ID: {mid}\n• Тип: {mt}\n• Метрики: {s['metrics']}\n• Признаки: {list(s['feature_importance'].keys())[:5]}", data=_to_native({"model_id":mid,**s}))
        except Exception as e: return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _do_find_best(self, task="binary_classification") -> AgentResponse:
        metric = "r2" if "regression" in task else "accuracy"
        best = self.registry.find_best(task, metric)
        if not best: return AgentResponse(message="Нет моделей. Обучите: «проанализируй datasets/churn_demo.csv» → «обучи xgboost»", success=False)
        return AgentResponse(message=f"Лучшая модель:\n• ID: {best['model_id']}\n• Тип: {best['model_type']}\n• Метрики: {best['metrics']}", data=_to_native(best))

    def _do_list_models(self) -> AgentResponse:
        models = self.registry.list_models()
        if not models: return AgentResponse(message="Реестр пуст", success=False)
        return AgentResponse(message="\n".join([f"• {r['model_id'][:50]}... — {r['task']}" for r in models[:5]]))

    def _do_search(self, message: str) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        finder = DatasetFinder()
        query = message
        for prefix in ["найди датасет","поищи","search","find dataset"]:
            if prefix in message.lower(): query = message.lower().split(prefix)[-1].strip(); break
        datasets = finder.search(query, max_results=5)
        self.last_search_results = datasets
        return AgentResponse(message=finder.format_for_chat(datasets), data=_to_native({"datasets":[{"title":d.title,"url":d.url} for d in datasets]}))

    def _do_install(self, num_str: str) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        finder = DatasetFinder()
        num = int(num_str) if num_str.isdigit() else 1
        if not self.last_search_results: self.last_search_results = finder.search("", max_results=5)
        if num < 1 or num > len(self.last_search_results): return AgentResponse(message=f"Номер от 1 до {len(self.last_search_results)}", success=False)
        chosen = self.last_search_results[num-1]
        return AgentResponse(message=f"Датасет: **{chosen.title}**\n{chosen.url}\n\nСкачайте и загрузите: «проанализируй [путь]»\nДемо: datasets/churn_demo.csv")
