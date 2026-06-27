"""Genesis Agent - LLM-powered with SQLite + Memory + Code Generator + Web Research."""

import json, os, re, pickle, numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry

class Intent:
    ANALYZE = "analyze"; TRAIN = "train"; FIND_BEST = "find_best"
    LIST_MODELS = "list_models"; DATASET_SEARCH = "dataset_search"
    INSTALL_DATASET = "install_dataset"; GENERATE_CODE = "generate_code"
    RESEARCH = "research_project"; HELP = "help"; UNKNOWN = "unknown"

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
Инструменты: analyze_dataset, train_model, find_best_model, list_models, search_datasets, generate_code, research_project.
Отвечай ТОЛЬКО в JSON: {"tool": "...", "args": {}, "message": "..."}."""

    ALLOWED_DIRS = ["./datasets", "./models", "./static", "./generated"]

    def __init__(self, profiler=None, trainer=None, registry=None):
        self.profiler = profiler or DatasetProfiler()
        self.trainer = trainer or DatasetTrainer()
        self.registry = registry or ModelRegistry()
        self.current_dataset = None
        self.last_search_results = []
        self.conversation_history = []
        from app.db.database import GenesisDB
        self.db = GenesisDB()
        self._init_llm()

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
        if not self.llm: return {"tool": None, "message": "Привет! Я Genesis AI Agent."}
        prompt = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        try:
            output = self.llm(prompt, max_tokens=256, stop=["<|im_end|>"], temperature=0.1)
            text = output["choices"][0]["text"].strip()
            if text.startswith("{"): return json.loads(text)
            return {"tool": None, "message": text}
        except json.JSONDecodeError:
            return {"tool": None, "message": "Извините, не понял."}
        except Exception as e:
            return {"tool": None, "message": "Извините, произошла ошибка."}

    def _get_memory_context(self) -> str:
        facts = self.db.get_facts("user")
        if not facts: return ""
        lines = ["\n## Память о пользователе:"]
        for f in facts[:5]:
            lines.append(f"- {f['predicate']}: {f['object']} (уверенность: {f['confidence']})")
        return "\n".join(lines)

    def _extract_and_save_facts(self, message: str):
        patterns = {
            "name": r"(?:меня зовут|я|моё имя)\s+(\w+)",
            "likes": r"(?:я люблю|обожаю|мне нравится|предпочитаю)\s+(.+?)(?:[.,!]|$)",
            "profession": r"(?:я работаю|моя профессия|я по профессии)\s+(.+?)(?:[.,!]|$)",
            "goal": r"(?:я хочу|моя цель|я планирую|я собираюсь)\s+(.+?)(?:[.,!]|$)",
        }
        for predicate, pattern in patterns.items():
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if 2 < len(value) < 100:
                    self.db.add_fact("user", predicate, value, 0.8)

    def _do_research(self, message: str, args: Dict = None) -> AgentResponse:
        from app.core.web_search import ProjectResearcher
        researcher = ProjectResearcher()
        description = (args or {}).get("description", message) if args else message
        try:
            research = researcher.research_project(description)
            summary = researcher.format_research_summary(research)
            return AgentResponse(
                message=summary[:1500] + "\n\nПродолжить? Могу найти датасеты или обучить модель.",
                data={"research": research}
            )
        except Exception as e:
            return AgentResponse(message=f"Ошибка исследования: {e}", success=False)

    def _do_generate_code(self, message: str, args: Dict = None) -> AgentResponse:
        from app.core.codegen.generator import CodeGenerator
        gen = CodeGenerator(self.db)
        description = (args or {}).get('description', message) if args else message
        try:
            code = gen.generate_from_description(description)
            path = gen.save_code(code)
            return AgentResponse(
                message=f"Сгенерировал ML-пайплайн!\nСохранён в: {path}\n\n```python\n{code[:800]}\n```\n\nЗапустить: python {path}",
                data={"code_path": path, "code": code[:500]}
            )
        except Exception as e:
            return AgentResponse(message=f"Ошибка генерации кода: {e}", success=False)

    def process(self, message: str, conversation_id: str = None, user_projects: List[Dict] = None) -> AgentResponse:
        cid = conversation_id or "default"
        self.db.create_chat(cid)
        self.db.add_message(cid, "user", message)
        self._extract_and_save_facts(message)
        msg_count = self.db.get_message_count(cid)
        projects = user_projects or self.db.get_projects()
        memory_context = self._get_memory_context()

        if msg_count == 1 and projects:
            first_msg = message.lower().strip().rstrip('!,.')
            greetings = ["привет","hello","hi","здравствуйте","здарова","хай","прив","добрый день","доброе утро","добрый вечер"]
            if any(g == first_msg for g in greetings):
                lines = ["Здравствуйте! Вот ваши проекты:"]
                for p in projects[:5]:
                    name = p.get('name','Проект'); task = p.get('task','').replace('_',' ')
                    lines.append(f"• {name} ({task})")
                lines.append("\nНад каким сегодня работаем?")
                msg = "\n".join(lines)
                self.db.add_message(cid, "assistant", msg)
                return AgentResponse(message=msg)

        if self.llm:
            result = self._call_llm(message + memory_context)
            tool = result.get("tool"); args = result.get("args", {})
            if tool in ('analyze_dataset','load_data','load_dataset','analyze'):
                path = args.get("path", self._extract_path(message))
                if path: return self._do_analyze(path)
                return AgentResponse(message="Укажите путь к файлу.", success=False)
            elif tool in ('train_model','train','fit_model'):
                return self._do_train(message, args)
            elif tool in ('find_best_model','best_model','find_best'):
                return self._do_find_best(args.get("task", "binary_classification"))
            elif tool in ('list_models','list','show_models'):
                return self._do_list_models()
            elif tool in ('search_datasets','search','find_datasets'):
                return self._do_search(message, args)
            elif tool in ('research_project','research','investigate'):
                return self._do_research(message, args)
            elif tool in ('generate_code','codegen','generate_pipeline'):
                return self._do_generate_code(message, args)
            elif tool in ('install_dataset','install','download_dataset'):
                return self._do_install(str(args.get("number", "1")))
            resp_msg = result.get("message", "Привет!")
            self.db.add_message(cid, "assistant", resp_msg)
            return AgentResponse(message=resp_msg)

        fp = self._extract_path(message)
        if fp: return self._do_analyze(fp)
        return AgentResponse(message="Привет! Я Genesis AI Agent. Спросите про ML!")

    def _validate_path(self, path: str) -> bool:
        real_path = os.path.realpath(path)
        for allowed in self.ALLOWED_DIRS:
            if real_path.startswith(os.path.realpath(allowed)): return True
        return False

    def _extract_path(self, message: str):
        for p in [r'([\w\\/:.-]+\.(?:csv|json|parquet|xlsx?))']:
            m = re.search(p, message, re.IGNORECASE)
            if m:
                path = m.group(1)
                if os.path.exists(path) and self._validate_path(path): return path
        return self.current_dataset

    def _do_analyze(self, fp: str) -> AgentResponse:
        if not self._validate_path(fp): return AgentResponse(message="Доступ запрещён.", success=False)
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
            model = pickle.load(open(result.model_path, "rb"))
            mid = self.registry.register(model=model, metrics=result.metrics, task=task, model_type=mt, features=self.profiler.profile.feature_columns, target=tc)
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
        if not query: query = message
        if query and any(ord(c) > 127 for c in query):
            try:
                import urllib.request, urllib.parse
                url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" + urllib.parse.quote(query)
                resp = urllib.request.urlopen(url, timeout=3)
                data = json.loads(resp.read())
                translated = ''.join([s[0] for s in data[0] if s[0]])
                if translated: query = translated
            except: pass
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
        return self.db.get_messages(conversation_id)

    def list_conversations(self) -> List[Dict]:
        return self.db.list_chats()
