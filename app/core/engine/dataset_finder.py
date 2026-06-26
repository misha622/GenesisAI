"""
Dataset Finder - поиск через HuggingFace API + локальная база.
"""

import re, os, json, urllib.request, urllib.parse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

@dataclass
class DatasetInfo:
    title: str
    description: str
    url: str
    source: str
    task: str
    size: str
    format: str
    rows: Optional[int] = None
    columns: Optional[int] = None
    rating: float = 0.0
    downloads: int = 0
    tags: List[str] = field(default_factory=list)

class DatasetFinder:
    def __init__(self):
        self.all_datasets = []
        for datasets in self.DATASETS_DB.values():
            self.all_datasets.extend(datasets)

    def search(self, query: str, task: Optional[str] = None, max_results: int = 5) -> List[DatasetInfo]:
        results = []
        hf_results = self._search_huggingface(query, max_results)
        results.extend(hf_results)
        if len(results) < max_results:
            local_results = self._search_local(query, task, max_results - len(results))
            existing_urls = {r.url for r in results}
            results.extend([r for r in local_results if r.url not in existing_urls])
        return results[:max_results]

    def _search_huggingface(self, query: str, max_results: int) -> List[DatasetInfo]:
        results = []
        try:
            url = "https://huggingface.co/api/datasets?" + urllib.parse.urlencode({
                "search": query, "sort": "downloads", "direction": "-1", "limit": max_results
            })
            req = urllib.request.Request(url, headers={"User-Agent": "GenesisAI/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                for ds in data:
                    results.append(DatasetInfo(
                        title=ds.get("id", "Unknown"),
                        description=str(ds.get("description", ""))[:200],
                        url="https://huggingface.co/datasets/" + ds.get("id", ""),
                        source="huggingface",
                        task=self._guess_task(str(ds.get("tags", [])) + " " + ds.get("id", "")),
                        size="See on HF",
                        format="Parquet/JSON",
                        downloads=int(ds.get("downloads", 0) or 0),
                        rating=min(5.0, float(ds.get("likes", 0) or 0) / 10),
                        tags=list(ds.get("tags", []) or []),
                    ))
        except Exception as e:
            print(f"HuggingFace API: {e}")
        return results

    def _search_local(self, query: str, task: Optional[str], max_results: int) -> List[DatasetInfo]:
        query_lower = query.lower()
        scored = []
        for ds in self.all_datasets:
            if task and ds.task != task:
                continue
            score = self._match_score(query_lower, ds)
            if score > 0:
                scored.append((score, ds))
        scored.sort(key=lambda x: (-x[0], -x[1].rating))
        return [r[1] for r in scored[:max_results]]

    def _match_score(self, query: str, ds: DatasetInfo) -> float:
        score = 0.0
        text = (ds.title + " " + ds.description + " " + " ".join(ds.tags)).lower()
        for word in re.findall(r'\w+', query):
            if word in ds.title.lower(): score += 4.0
            if word in " ".join(ds.tags).lower(): score += 3.0
            if word in ds.description.lower(): score += 1.5
        return score

    def _guess_task(self, text: str) -> str:
        text = text.lower()
        if any(w in text for w in ["classification","classify"]): return "classification"
        if any(w in text for w in ["regression","predict","forecast"]): return "regression"
        if any(w in text for w in ["nlp","text","language","sentiment"]): return "nlp"
        if any(w in text for w in ["image","vision","detection","segmentation"]): return "image"
        return "other"

    def install_dataset(self, dataset_url: str, target_dir: str = "./datasets") -> str:
        os.makedirs(target_dir, exist_ok=True)
        fname = dataset_url.split("/")[-1][:50]
        path = f"{target_dir}/{fname}"
        with open(path, "w") as f:
            f.write(f"# Dataset: {dataset_url}\n# Please download manually\n")
        return path

    def format_for_chat(self, datasets: List[DatasetInfo]) -> str:
        if not datasets:
            return "Ничего не нашёл. Попробуйте другие ключевые слова или поищите на https://huggingface.co/datasets"
        lines = [f"Нашёл {len(datasets)} датасетов:\n"]
        for i, ds in enumerate(datasets, 1):
            stars = "⭐" * min(5, int(ds.rating))
            icon = {"huggingface": "🤗", "kaggle": "🟦"}.get(ds.source, "📦")
            lines.append(f"{i}. {icon} **{ds.title[:80]}** {stars}\n   📋 {ds.description[:150]}\n   📊 {ds.size} · {ds.format}\n   🔗 {ds.url}\n   🏷️ {', '.join(ds.tags[:5])}\n")
        lines.append("\nМогу установить любой. Напишите номер (например: «установи 1»)")
        return "\n".join(lines)

    DATASETS_DB = {
        "fruits": [
            DatasetInfo("Fruits 360 Dataset", "90K+ images of fruits/vegetables. 131 classes.",
                        "https://www.kaggle.com/datasets/moltean/fruits","kaggle","classification",
                        "580 MB","Images",90483,0,4.8,800000,["fruits","freshness","image"]),
        ],
        "security": [
            DatasetInfo("UNSW-NB15 Cybersecurity", "Network traffic. 9 attack types. 2.5M records.",
                        "https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15","kaggle",
                        "classification","100 MB","CSV",2540044,49,4.6,300000,["security","cyber","network"]),
        ],
    }
