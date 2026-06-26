"""
Genesis Agent - диалоговый ИИ-помощник.

Принимает запросы пользователя на естественном языке,
определяет намерение (intent) и вызывает нужный модуль:
- Profiler: "проанализируй датасет"
- Trainer: "обучи модель"
- Registry: "найди лучшую модель"
"""

import json
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Пути к модулям движка
from app.core.engine.profiler import DatasetProfiler
from app.core.engine.trainer import DatasetTrainer
from app.core.engine.registry import ModelRegistry


class Intent:
    """Намерение, извлечённое из сообщения пользователя."""
    ANALYZE = "analyze"           # анализ датасета
    TRAIN = "train"               # обучение модели
    FIND_BEST = "find_best"       # найти лучшую модель
    LIST_MODELS = "list_models"   # список моделей
    COMPARE = "compare"           # сравнить модели
    PREDICT = "predict"           # предсказание
    HELP = "help"                 # справка
    UNKNOWN = "unknown"           # не распознано


class AgentResponse:
    """Ответ агента пользователю."""
    def __init__(self, message: str, data: Optional[Dict] = None, success: bool = True):
        self.message = message
        self.data = data or {}
        self.success = success
        self.timestamp = datetime.now().isoformat()


class GenesisAgent:
    """
    Главный агент платформы Genesis AI.
    
    Принимает сообщения пользователя, распознаёт намерения,
    вызывает соответствующие модули движка.
    """
    
    # Ключевые слова для определения намерений
    INTENT_KEYWORDS = {
        Intent.ANALYZE: [
            "анализ", "профиль", "посмотри", "изучи", "покажи данные",
            "что в датасете", "колонки", "столбцы", "пропуски",
            "analyze", "profile", "explore", "inspect", "columns",
        ],
        Intent.TRAIN: [
            "обучи", "тренируй", "построй модель", "модель", "обучение",
            "train", "fit", "build model", "xgboost", "logistic",
        ],
        Intent.FIND_BEST: ["найди", "лучш", "best", "топ", "сам", 
            "найди лучшую", "лучшая модель", "best model", "топ модель",
            "самая точная", "покажи лучшую",
        ],
        Intent.LIST_MODELS: [
            "список моделей", "все модели", "какие модели", "list models",
            "покажи модели", "мои модели",
        ],
        Intent.COMPARE: [
            "сравни", "сравнение", "compare", "что лучше",
        ],
        Intent.PREDICT: [
            "предскажи", "прогноз", "predict", "получить результат",
        ],
        Intent.HELP: [
            "помощь", "что ты умеешь", "команды", "help", "справка",
            "начнём", "привет", "hi", "hello",
        ],
    }
    
    def __init__(
        self,
        profiler: Optional[DatasetProfiler] = None,
        trainer: Optional[DatasetTrainer] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        self.profiler = profiler or DatasetProfiler()
        self.trainer = trainer or DatasetTrainer()
        self.registry = registry or ModelRegistry()
        self.current_dataset: Optional[str] = None  # путь к загруженному датасету
        self.conversation_history: List[Dict] = []
    
    def detect_intent(self, message: str) -> str:
        """
        Определение намерения пользователя по ключевым словам.
        
        В реальной системе здесь будет LLM (GPT/Llama),
        сейчас используем простой keyword matching.
        """
        message_lower = message.lower()
        
        # Считаем совпадения по каждой категории
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in message_lower)
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return Intent.UNKNOWN
        
        # Возвращаем намерение с максимальным счётом
        return max(scores, key=scores.get)
    
    def extract_file_path(self, message: str) -> Optional[str]:
        """Извлечение пути к файлу из сообщения."""
        # Ищем слова заканчивающиеся на .csv, .json, .parquet
        import re
        patterns = [
            r'[\w\\/:.-]+\.(?:csv|json|parquet|xlsx?)',
            r'["\']([\w\\/:.-]+\.(?:csv|json|parquet|xlsx?))["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                path = match.group(1) if match.lastindex else match.group(0)
                if os.path.exists(path):
                    return path
        
        # Если нет пути, но есть текущий датасет
        if self.current_dataset:
            return self.current_dataset
        
        return None
    
    def extract_target_column(self, message: str) -> Optional[str]:
        """Извлечение целевой колонки из сообщения."""
        import re
        # Паттерны: "целевая колонка X", "таргет X", "предскажи X", "target=X"
        patterns = [
            r'(?:целевая|таргет|target|предска(?:жи|зать)|predict)\s+(?:колонка|столбец|переменная)?\s*[=:]*\s*["\']?(\w+)["\']?',
            r'target[=:]\s*["\']?(\w+)["\']?',
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def extract_model_type(self, message: str) -> str:
        """Извлечение типа модели из сообщения."""
        m = message.lower()
        if "логистическ" in m or "logistic" in m: return "logistic"
        if "случайн" in m or "random" in m or "лес" in m: return "random_forest"
        if "xgboost" in m or "xgb" in m: return "xgboost"
        if "линейн" in m or "linear" in m: return "linear"
        return "xgboost"  # default
    
    def extract_model_ids(self, message: str) -> List[str]:
        """Извлечение ID моделей из сообщения."""
        import re
        # Ищем ID в формате binary_classification_xgboost_20260101_120000
        pattern = r'(\w+_\w+_\d{8}_\d{6}(?:_\d+)?)'
        return re.findall(pattern, message)
    
    def process(self, message: str) -> AgentResponse:
        """
        Обработка сообщения пользователя.
        
        Returns:
            AgentResponse с ответом и данными
        """
        # Сохраняем в историю
        self.conversation_history.append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Определяем намерение
        intent = self.detect_intent(message)
        
        # Обрабатываем
        if intent == Intent.HELP:
            return self._handle_help()
        elif intent == Intent.ANALYZE:
            return self._handle_analyze(message)
        elif intent == Intent.TRAIN:
            return self._handle_train(message)
        elif intent == Intent.FIND_BEST:
            return self._handle_find_best(message)
        elif intent == Intent.LIST_MODELS:
            return self._handle_list_models(message)
        elif intent == Intent.COMPARE:
            return self._handle_compare(message)
        else:
            return self._handle_unknown(message)
    
    def _handle_help(self) -> AgentResponse:
        return AgentResponse(
            message=(
                "Я — Genesis AI Agent. Я умею:\n"
                "📊 Анализировать датасеты: «проанализируй data.csv»\n"
                "🤖 Обучать модели: «обучи модель на churn с xgboost»\n"
                "🔍 Искать лучшую модель: «найди лучшую модель для классификации»\n"
                "📋 Показывать список: «покажи все модели»\n"
                "⚖️ Сравнивать: «сравни модель_1 и модель_2»\n\n"
                "Что будем делать?"
            ),
            data={"intent": Intent.HELP},
        )
    
    def _handle_analyze(self, message: str) -> AgentResponse:
        file_path = self.extract_file_path(message)
        if not file_path:
            return AgentResponse(
                message="Укажите путь к файлу. Например: «проанализируй C:/data/housing.csv»",
                success=False,
            )
        
        try:
            self.profiler.run_full_profile(file_path)
            self.current_dataset = file_path
            summary = self.profiler.get_summary()
            
            return AgentResponse(
                message=(
                    f"Проанализировал {summary['file_info']['name']}:\n"
                    f"• Строк: {summary['file_info']['rows']}\n"
                    f"• Колонок: {summary['file_info']['columns']}\n"
                    f"• Задача: {summary['suggested_ml_task']['task']}\n"
                    f"• Целевая переменная: {summary['suggested_ml_task']['target_column']}\n"
                    f"• Уверенность: {summary['suggested_ml_task']['confidence']}\n\n"
                    "Хотите обучить модель?"
                ),
                data=summary,
            )
        except Exception as e:
            return AgentResponse(
                message=f"Ошибка при анализе: {str(e)}",
                success=False,
            )
    
    def _handle_train(self, message: str) -> AgentResponse:
        file_path = self.extract_file_path(message)
        if not file_path:
            return AgentResponse(
                message="Укажите датасет. Например: «обучи модель на data.csv»",
                success=False,
            )
        
        target_col = self.extract_target_column(message)
        model_type = self.extract_model_type(message)
        
        try:
            # Профилируем если ещё не
            if self.current_dataset != file_path:
                self.profiler.run_full_profile(file_path)
                self.current_dataset = file_path
            
            profile_summary = self.profiler.get_summary()
            task = profile_summary["suggested_ml_task"]["task"]
            
            if not target_col:
                target_col = profile_summary["suggested_ml_task"]["target_column"]
            
            # Обучаем
            result = self.trainer.train(
                df=self.profiler.df,
                target_col=target_col,
                task=task,
                model_type=model_type,
            )
            
            # Регистрируем
            model_id = self.registry.register(
                model=pickle.load(open(result.model_path, "rb")),
                metrics=result.metrics,
                task=task,
                model_type=model_type,
                features=self.profiler.profile.feature_columns,
                target=target_col,
            )
            
            summary = self.trainer.get_summary()
            
            return AgentResponse(
                message=(
                    f"Модель обучена и сохранена!\n"
                    f"• ID: {model_id}\n"
                    f"• Тип: {model_type}\n"
                    f"• Задача: {task}\n"
                    f"• Метрики: {summary['metrics']}\n"
                    f"• Важные признаки: {list(summary['feature_importance'].keys())[:5]}\n\n"
                    "Можете спросить «найди лучшую модель» или «покажи список»"
                ),
                data={"model_id": model_id, **summary},
            )
        except Exception as e:
            import traceback
            return AgentResponse(
                message=f"Ошибка при обучении: {str(e)}",
                success=False,
                data={"traceback": traceback.format_exc()},
            )
    
    def _handle_find_best(self, message: str) -> AgentResponse:
        m = message.lower()
        task = "binary_classification"
        metric = "accuracy"
        
        if "регресс" in m or "regression" in m:
            task = "regression"
            metric = "r2"
        elif "мульти" in m or "multiclass" in m:
            task = "multiclass_classification"
        
        best = self.registry.find_best(task, metric)
        
        if not best:
            return AgentResponse(
                message=f"Нет моделей для задачи '{task}'. Обучите первую модель!",
                success=False,
            )
        
        return AgentResponse(
            message=(
                f"Лучшая модель для {task}:\n"
                f"• ID: {best['model_id']}\n"
                f"• Тип: {best['model_type']}\n"
                f"• Метрики: {best['metrics']}\n"
                f"• Создана: {best['created_at']}"
            ),
            data=best,
        )
    
    def _handle_list_models(self, message: str) -> AgentResponse:
        task = None
        m = message.lower()
        if "классифик" in m or "classif" in m: task = "binary_classification"
        if "регресс" in m or "regression" in m: task = "regression"
        
        models = self.registry.list_models(task=task)
        
        if not models:
            return AgentResponse(
                message="В реестре пока нет моделей. Обучите первую!",
                success=False,
            )
        
        lines = [f"Всего моделей: {len(models)}"]
        for rec in models[:5]:
            lines.append(f"• {rec['model_id'][:50]}... — {rec['task']} ({rec['model_type']})")
        
        return AgentResponse(
            message="\n".join(lines),
            data={"models": models},
        )
    
    def _handle_compare(self, message: str) -> AgentResponse:
        ids = self.extract_model_ids(message)
        
        if len(ids) < 2:
            return AgentResponse(
                message="Укажите ID моделей для сравнения. Пример: «сравни model_id_1 и model_id_2»",
                success=False,
            )
        
        df = self.registry.compare_models(ids)
        
        return AgentResponse(
            message=f"Сравнение {len(ids)} моделей:\n{df.to_string()}",
            data={"comparison": df.to_dict()},
        )
    
    def _handle_unknown(self, message: str) -> AgentResponse:
        return AgentResponse(
            message=(
                f"Не совсем понял запрос: «{message}»\n"
                "Попробуйте: «проанализируй датасет», «обучи модель», "
                "«покажи список моделей», «помощь»"
            ),
            success=False,
            data={"intent": Intent.UNKNOWN},
        )


# Необходимый импорт для _handle_train
import pickle

