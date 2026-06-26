"""
Model Registry - хранилище обученных моделей с метаданными.

Сохраняет модели, метрики, датасеты в локальном хранилище.
Позволяет искать, сравнивать и загружать модели.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np


class ModelRecord:
    """Запись об одной модели в реестре."""
    def __init__(
        self,
        model_id: str,
        model_type: str,
        task: str,
        metrics: Dict[str, float],
        features: List[str],
        target: str,
        created_at: str,
        model_path: str,
        metadata: Optional[Dict] = None,
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.task = task
        self.metrics = metrics
        self.features = features
        self.target = target
        self.created_at = created_at
        self.model_path = model_path
        self.metadata = metadata or {}


class ModelRegistry:
    """
    Реестр моделей.
    
    Хранит все обученные модели, их метаданные и метрики.
    Позволяет искать лучшую модель под задачу.
    """
    
    def __init__(self, storage_dir: str = "./registry"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.storage_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.index_path = self.storage_dir / "index.json"
        self._index: Dict[str, Dict] = self._load_index()
    
    def _load_index(self) -> Dict[str, Dict]:
        """Загрузка индекса моделей из JSON."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Сохранение индекса в JSON."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)
    
    def register(
        self,
        model,
        metrics: Dict[str, float],
        task: str,
        model_type: str,
        features: List[str],
        target: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Регистрация новой модели в реестре.
        
        Args:
            model: обученная модель (sklearn/xgboost объект)
            metrics: словарь метрик
            task: тип задачи
            model_type: тип модели
            features: список признаков
            target: целевая переменная
            metadata: дополнительные метаданные
            
        Returns:
            model_id уникальный идентификатор модели
        """
        # Генерация ID
        model_id = f"{task}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Сохранение модели
        model_filename = f"{model_id}.pkl"
        model_path = self.models_dir / model_filename
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Запись в индекс
        record = {
            "model_id": model_id,
            "model_type": model_type,
            "task": task,
            "metrics": metrics,
            "features": features,
            "target": target,
            "created_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "metadata": metadata or {},
        }
        
        self._index[model_id] = record
        self._save_index()
        
        return model_id
    
    def get_model(self, model_id: str):
        """Загрузка модели по ID."""
        if model_id not in self._index:
            raise KeyError(f"Model {model_id} not found in registry")
        
        record = self._index[model_id]
        with open(record["model_path"], "rb") as f:
            return pickle.load(f)
    
    def get_record(self, model_id: str) -> Dict:
        """Получение записи о модели."""
        if model_id not in self._index:
            raise KeyError(f"Model {model_id} not found")
        return self._index[model_id]
    
    def find_best(
        self,
        task: str,
        metric: str = "accuracy",
        ascending: bool = False,
    ) -> Optional[Dict]:
        """
        Поиск лучшей модели по задаче и метрике.
        
        Args:
            task: тип задачи (binary_classification, regression, ...)
            metric: название метрики для сравнения
            ascending: True если меньше = лучше (для RMSE)
            
        Returns:
            Запись о лучшей модели или None
        """
        candidates = [
            (mid, rec) for mid, rec in self._index.items()
            if rec["task"] == task and metric in rec.get("metrics", {})
        ]
        
        if not candidates:
            return None
        
        candidates.sort(
            key=lambda x: x[1]["metrics"].get(metric, 0),
            reverse=not ascending,
        )
        
        return candidates[0][1]
    
    def list_models(
        self,
        task: Optional[str] = None,
        model_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Список моделей с фильтрацией.
        """
        result = []
        for mid, rec in self._index.items():
            if task and rec["task"] != task:
                continue
            if model_type and rec["model_type"] != model_type:
                continue
            result.append(rec)
        
        # Сортировка по дате (новые сверху)
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result[:limit]
    
    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Сравнение нескольких моделей по метрикам.
        
        Returns:
            DataFrame с метриками моделей
        """
        rows = []
        for mid in model_ids:
            if mid in self._index:
                rec = self._index[mid]
                row = {
                    "model_id": mid,
                    "task": rec["task"],
                    "model_type": rec["model_type"],
                    **rec["metrics"],
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def delete_model(self, model_id: str):
        """Удаление модели из реестра."""
        if model_id not in self._index:
            raise KeyError(f"Model {model_id} not found")
        
        record = self._index[model_id]
        
        # Удалить файл модели
        model_path = Path(record["model_path"])
        if model_path.exists():
            model_path.unlink()
        
        # Удалить из индекса
        del self._index[model_id]
        self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика реестра."""
        tasks = {}
        types = {}
        for rec in self._index.values():
            task = rec["task"]
            mtype = rec["model_type"]
            tasks[task] = tasks.get(task, 0) + 1
            types[mtype] = types.get(mtype, 0) + 1
        
        return {
            "total_models": len(self._index),
            "by_task": tasks,
            "by_type": types,
            "storage_dir": str(self.storage_dir),
        }
