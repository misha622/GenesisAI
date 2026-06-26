"""
Dataset Finder - поиск датасетов в открытых источниках.

Ищет релевантные датасеты на Kaggle, Hugging Face, UCI ML Repository.
Анализирует задачу пользователя и подбирает подходящие данные.
Ничего не скачивает автоматически — только предлагает пользователю.
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DatasetInfo:
    """Информация о найденном датасете."""
    title: str
    description: str
    url: str
    source: str              # kaggle, huggingface, uci, openml
    task: str                # classification, regression, nlp, cv
    size: str                # 100MB, 1.2GB
    format: str              # CSV, JSON, Parquet, Images
    rows: Optional[int] = None
    columns: Optional[int] = None
    rating: float = 0.0
    downloads: int = 0
    license_info: str = ""
    tags: List[str] = field(default_factory=list)


class DatasetFinder:
    """
    Поисковик датасетов.
    
    Ищет в предустановленной базе датасетов + может
    подключаться к API Kaggle/HuggingFace (опционально).
    """
    
    # Встроенная база популярных датасетов по задачам
    DATASETS_DB = {
        # Binary Classification
        "churn": [
            DatasetInfo(
                title="Telco Customer Churn",
                description="Telecom customer churn dataset. 7043 rows, 21 features. Predict if customer will leave.",
                url="https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
                source="kaggle",
                task="binary_classification",
                size="956 KB",
                format="CSV",
                rows=7043,
                columns=21,
                rating=4.5,
                downloads=500000,
                license_info="Open Data",
                tags=["churn", "telecom", "customers", "binary"],
            ),
            DatasetInfo(
                title="Credit Card Fraud Detection",
                description="Anonymized credit card transactions. 284k rows. Highly imbalanced binary classification.",
                url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
                source="kaggle",
                task="binary_classification",
                size="143 MB",
                format="CSV",
                rows=284807,
                columns=31,
                rating=4.8,
                downloads=1000000,
                license_info="Open Data",
                tags=["fraud", "finance", "imbalanced", "binary"],
            ),
        ],
        "regression": [
            DatasetInfo(
                title="Boston Housing Prices",
                description="Classic housing price prediction. 506 rows, 14 features.",
                url="https://www.kaggle.com/datasets/vikrishnan/boston-house-prices",
                source="kaggle",
                task="regression",
                size="30 KB",
                format="CSV",
                rows=506,
                columns=14,
                rating=4.2,
                downloads=200000,
                license_info="Public Domain",
                tags=["housing", "prices", "real-estate", "regression"],
            ),
            DatasetInfo(
                title="California Housing Prices",
                description="Larger housing dataset. 20640 rows. Median house values for California districts.",
                url="https://www.kaggle.com/datasets/camnugent/california-housing-prices",
                source="kaggle",
                task="regression",
                size="1.2 MB",
                format="CSV",
                rows=20640,
                columns=10,
                rating=4.4,
                downloads=350000,
                license_info="CC0",
                tags=["housing", "california", "prices", "regression"],
            ),
        ],
        "nlp": [
            DatasetInfo(
                title="IMDB Movie Reviews",
                description="50K movie reviews for sentiment analysis. Binary positive/negative.",
                url="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
                source="kaggle",
                task="binary_classification",
                size="63 MB",
                format="CSV",
                rows=50000,
                columns=2,
                rating=4.6,
                downloads=800000,
                license_info="Open Data",
                tags=["nlp", "sentiment", "movies", "text"],
            ),
        ],
        "multiclass": [
            DatasetInfo(
                title="Iris Flower Dataset",
                description="Classic 3-class dataset. 150 samples, 4 features.",
                url="https://archive.ics.uci.edu/ml/datasets/iris",
                source="uci",
                task="multiclass_classification",
                size="4 KB",
                format="CSV",
                rows=150,
                columns=5,
                rating=4.9,
                downloads=10000000,
                license_info="Public Domain",
                tags=["iris", "flowers", "multiclass", "classic"],
            ),
        ],
        "image": [
            DatasetInfo(
                title="MNIST Handwritten Digits",
                description="70K grayscale images of handwritten digits (0-9). 28x28 pixels.",
                url="https://www.kaggle.com/datasets/hojjatk/mnist-dataset",
                source="kaggle",
                task="multiclass_classification",
                size="52 MB",
                format="Images",
                rows=70000,
                columns=784,
                rating=5.0,
                downloads=5000000,
                license_info="CC BY-SA 3.0",
                tags=["mnist", "digits", "image", "vision"],
            ),
        ],
    }
    
    def __init__(self, use_api: bool = False):
        self.use_api = use_api
    
    def search(
        self,
        query: str,
        task: Optional[str] = None,
        source: Optional[str] = None,
        max_results: int = 5,
    ) -> List[DatasetInfo]:
        """
        Поиск датасетов по запросу.
        
        Args:
            query: поисковый запрос (название задачи)
            task: фильтр по типу задачи
            source: фильтр по источнику
            max_results: максимум результатов
            
        Returns:
            список подходящих датасетов
        """
        results = []
        query_lower = query.lower()
        
        # Поиск по всем датасетам
        for task_key, datasets in self.DATASETS_DB.items():
            for ds in datasets:
                # Проверка по задаче
                if task and ds.task != task:
                    continue
                # Проверка по источнику
                if source and ds.source != source:
                    continue
                # Поиск по названию, описанию, тегам
                score = self._match_score(query_lower, ds)
                if score > 0:
                    results.append((score, ds))
        
        # Сортировка по релевантности
        results.sort(key=lambda x: (-x[0], -x[1].rating))
        
        return [r[1] for r in results[:max_results]]
    
    def _match_score(self, query: str, ds: DatasetInfo) -> float:
        """Оценка релевантности датасета запросу."""
        score = 0.0
        text = (ds.title + " " + ds.description + " " + " ".join(ds.tags)).lower()
        
        # Разбиваем запрос на слова
        words = re.findall(r'\w+', query)
        
        for word in words:
            if word in ds.title.lower():
                score += 3.0
            if word in " ".join(ds.tags).lower():
                score += 2.0
            if word in ds.description.lower():
                score += 1.0
        
        # Бонус за точное совпадение задачи
        task_map = {
            "классификац": "classification",
            "регресси": "regression",
            "отток": "churn",
            "churn": "churn",
            "цена": "regression",
            "price": "regression",
            "мошенничеств": "fraud",
            "fraud": "fraud",
            "текст": "nlp",
            "text": "nlp",
            "изображени": "image",
            "image": "image",
        }
        
        for key, val in task_map.items():
            if key in query and val in ds.tags:
                score += 5.0
        
        return score
    
    def get_sources_summary(self) -> Dict[str, int]:
        """Статистика по источникам."""
        sources = {}
        for datasets in self.DATASETS_DB.values():
            for ds in datasets:
                sources[ds.source] = sources.get(ds.source, 0) + 1
        return sources
    
    def install_dataset(self, dataset_url: str, target_dir: str = "./datasets") -> str:
        """
        Установка датасета (после подтверждения пользователем).
        
        В реальной системе использует kagglehub / huggingface_hub.
        Сейчас — заглушка с метаданными.
        
        Returns:
            путь к скачанному файлу
        """
        import os
        os.makedirs(target_dir, exist_ok=True)
        
        # Ищем датасет в базе по URL
        for datasets in self.DATASETS_DB.values():
            for ds in datasets:
                if ds.url == dataset_url:
                    # В реальности — скачивание
                    print(f"Would download: {ds.title} from {ds.source}")
                    return f"{target_dir}/{ds.title.lower().replace(' ', '_')}.csv"
        
        raise ValueError(f"Dataset not found: {dataset_url}")
    
    def format_for_chat(self, datasets: List[DatasetInfo]) -> str:
        """Форматирование результатов для чата."""
        if not datasets:
            return "Не нашёл подходящих датасетов. Попробуйте уточнить задачу."
        
        lines = [f"Нашёл {len(datasets)} подходящих датасетов:\n"]
        
        for i, ds in enumerate(datasets, 1):
            lines.append(
                f"{i}. **{ds.title}** ⭐{ds.rating}\n"
                f"   📋 {ds.description}\n"
                f"   📊 {ds.rows:,} rows × {ds.columns} cols · {ds.size} · {ds.format}\n"
                f"   🔗 {ds.url}\n"
                f"   🏷️ {', '.join(ds.tags)}\n"
            )
        
        lines.append("\nМогу установить любой из них. Какой выбираете? (напишите номер)")
        
        return "\n".join(lines)
