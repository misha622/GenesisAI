"""Genesis Agent - LLM-powered with chat history + projects."""

import json, os, re, pickle, numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry

class Intent:
    ANALYZE = "analyze"; TRAIN = "train"; FIND_BEST = "find_best"
    LIST_MODELS = "list_models"; DATASET_SEARCH = "dataset_search"
    INSTALL_DATASET = "install_dataset"; HELP = "help"; UNKNOWN = "unknown"

class AgentResponse:
    def __init__(self, message: str, data=None, success: bool = True):
        self.message = message; self.data = data or {}; self.success = success
        self.timestamp = datetime.now().isoformat()

def _to_native(obj):
    if isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_to_native(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    return obj

class GenesisAgent:
    SYSTEM_PROMPT = """Ты Genesis AI Agent — AutoML ассистент. Отвечай на том же языке, что и пользователь.
Инструменты: analyze_dataset, train_model, find_best_model, list_models, search_datasets.
Отвечай ТОЛЬКО в JSON: {"tool": "...", "args": {}, "message": "..."}."""

    def __init__(self, profiler=None, trainer=None, registry=None):
        self.profiler = profiler or DatasetProfiler()
        self.trainer = trainer or DatasetTrainer()
        self.registry = registry or ModelRegistry()
        self.current_dataset = None
        self.last_search_results = []
        self.conversation_history = []
        self.chat_store = self._load_chats()
        self._init_llm()

    def _load_chats(self):
        path = "chats.json"
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
        return {}

    def _save_chats(self):
        with open("chats.json", 'w', encoding='utf-8') as f:
            json.dump(self.chat_store, f, indent=2, ensure_ascii=False)

    def _load_user_projects(self):
        try:
            with open('projects.json', 'r') as f: return json.load(f)
        except: return []

    def _init_llm(self):
        try:
            from llama_cpp import Llama
            model_path = "models/qwen2.5-7b-instruct-q4_k_m.gguf"
            if os.path.exists(model_path):
                self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)
                print("[OK] LLM loaded")
            else: self.llm = None
        except Exception as e:
            print(f"[WARN] LLM: {e}"); self.llm = None

    def _call_llm(self, msg: str) -> Dict:
        if not self.llm:
            return {"tool": None, "message": "Привет! Я Genesis AI Agent."}
        prompt = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        try:
            output = self.llm(prompt, max_tokens=256, stop=["<|im_end|>"], temperature=0.1)
            text = output["choices"][0]["text"].strip()
            if text.startswith("{"): return json.loads(text)
            return {"tool": None, "message": text}
        except:
            return {"tool": None, "message": "Привет! Как я могу помочь?"}

    def process(self, message: str, conversation_id: str = None, user_projects: List[Dict] = None) -> AgentResponse:
        cid = conversation_id or "default"
        if cid not in self.chat_store:
            self.chat_store[cid] = {"messages": [], "created": datetime.now().isoformat()}
        self.chat_store[cid]["messages"].append({"role":"user","content":message,"timestamp":datetime.now().isoformat()})
        
        projects = user_projects or self._load_user_projects()
        
        # ПРИВЕТСТВИЕ С ПРОЕКТАМИ — до LLM
        if projects:  # показывать проекты при любом приветствии
            first_msg = message.lower().strip().rstrip('!,.')
            greetings = ["привет","hello","hi","здравствуйте","здарова","хай","прив","добрый день","доброе утро","добрый вечер"]
            if any(g in first_msg for g in greetings):
                lines = ["Здравствуйте! Вот ваши проекты:"]
                for p in projects[:5]:
                    name = p.get('name','Проект')
                    task = p.get('task','').replace('_',' ')
                    lines.append(f"• {name} ({task})")
                lines.append("\nНад каким сегодня работаем?")
                msg = "\n".join(lines)
                self.chat_store[cid]["messages"].append({"role":"assistant","content":msg,"timestamp":datetime.now().isoformat()})
                self._save_chats()
                return AgentResponse(message=msg)
        
        if self.llm:
            result = self._call_llm(message)
            tool = result.get("tool")
            args = result.get("args", {})
            if tool in ('analyze_dataset','load_data','load_dataset','analyze'):
                path = args.get("path", self._extract_path(message))
                if path: return self._do_analyze(path)
            elif tool in ('train_model','train','fit_model'):
                return self._do_train(message, args)
            elif tool in ('find_best_model','best_model','find_best'):
                return self._do_find_best(args.get("task", "binary_classification"))
            elif tool in ('list_models','list','show_models'):
                return self._do_list_models()
            elif tool in ('search_datasets','search','find_datasets'):
                return self._do_search(message, args)
            elif tool in ('install_dataset','install','download_dataset'):
                return self._do_install(str(args.get("number", "1")))
            resp_msg = result.get("message", "Привет!")
            self.chat_store[cid]["messages"].append({"role":"assistant","content":resp_msg,"timestamp":datetime.now().isoformat()})
            self._save_chats()
            return AgentResponse(message=resp_msg)
        return AgentResponse(message="Привет! Я Genesis AI Agent.")

    def _extract_path(self, message: str):
        for p in [r'([\w\\/:.-]+\.(?:csv|json|parquet|xlsx?))']:
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
        if not fp: return AgentResponse(message="Загрузите датасет: datasets/churn_demo.csv", success=False)
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
            return AgentResponse(message=f"Модель обучена!\n• ID: {mid}\n• Тип: {mt}\n• Метрики: {s['metrics']}", data=_to_native({"model_id":mid,**s}))
        except Exception as e: return AgentResponse(message=f"Ошибка: {e}", success=False)

    def _do_find_best(self, task="binary_classification") -> AgentResponse:
        metric = "r2" if "regression" in task else "accuracy"
        best = self.registry.find_best(task, metric)
        if not best: return AgentResponse(message="Нет моделей.", success=False)
        return AgentResponse(message=f"Лучшая модель:\n• ID: {best['model_id']}\n• Метрики: {best['metrics']}", data=_to_native(best))

    def _do_list_models(self) -> AgentResponse:
        models = self.registry.list_models()
        if not models: return AgentResponse(message="Реестр пуст", success=False)
        return AgentResponse(message="\n".join([f"• {r['model_id'][:50]}... — {r['task']}" for r in models[:5]]))

    def _do_search(self, message: str, args: Dict = None) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        finder = DatasetFinder()
        query = (args or {}).get('query', '') if args else ''
        if not query:
            query = message
        # Автоперевод через Google Translate API
        if query and any(ord(c) > 127 for c in query):
            try:
                import urllib.request, urllib.parse, json
                url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(query)
                resp = urllib.request.urlopen(url, timeout=3)
                data = json.loads(resp.read())
                translated = ''.join([s[0] for s in data[0] if s[0]])
                if translated:
                    query = translated
            except:
                pass
        datasets = finder.search(query, max_results=5)
        self.last_search_results = datasets
        return AgentResponse(message=finder.format_for_chat(datasets))

    def _do_install(self, num_str: str) -> AgentResponse:
        from app.core.engine.dataset_finder import DatasetFinder
        finder = DatasetFinder()
        num = int(num_str) if num_str.isdigit() else 1
        if not self.last_search_results: self.last_search_results = finder.search("", max_results=5)
        if num < 1 or num > len(self.last_search_results): return AgentResponse(message=f"Номер от 1 до {len(self.last_search_results)}", success=False)
        chosen = self.last_search_results[num-1]
        return AgentResponse(message=f"Датасет: **{chosen.title}**\n{chosen.url}\n\nДемо: datasets/churn_demo.csv")

    def get_chat_history(self, conversation_id: str) -> List[Dict]:
        return self.chat_store.get(conversation_id, {}).get("messages", [])

    def list_conversations(self) -> List[Dict]:
        return [{"id": cid, "title": data["messages"][0]["content"][:40] if data.get("messages") else "Empty", "date": data.get("created","")} for cid, data in self.chat_store.items()]
