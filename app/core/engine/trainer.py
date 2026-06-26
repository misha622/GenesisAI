import pandas as pd, numpy as np, pickle
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class ModelResult:
    def __init__(self):
        self.model_type = self.task = self.model_path = self.creation_time = ""
        self.metrics: Dict = {}
        self.feature_importance: Dict = {}

class DatasetTrainer:
    def __init__(self, model_dir="./models"):
        self.model_dir = Path(model_dir); self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result = ModelResult()

    def prepare_data(self, df, target_col):
        df = df.dropna(subset=[target_col]).copy()
        y = df[target_col]; X = df.drop(columns=[target_col])
        for c in X.select_dtypes(include=[np.number]).columns:
            X[c] = X[c].fillna(X[c].median())
        cat = X.select_dtypes(exclude=[np.number]).columns
        X = pd.get_dummies(X, columns=cat, drop_first=True)
        return X, y

    def train(self, df, target_col, task, model_type="xgboost", test_size=0.2, cv_folds=5):
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
        
        X, y = self.prepare_data(df, target_col)
        n_samples = len(X)
        
        # Предупреждения о размере датасета
        warnings_list = []
        if n_samples < 100:
            warnings_list.append(f"Малый датасет: {n_samples} строк. Рекомендуется > 1000.")
        if n_samples < 50:
            warnings_list.append(f"Очень малый датасет ({n_samples} строк). Результаты недостоверны.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Выбор модели
        if "classification" in task:
            if model_type=="logistic":
                from sklearn.linear_model import LogisticRegression
                m = LogisticRegression(max_iter=1000, random_state=42)
            elif model_type=="random_forest":
                from sklearn.ensemble import RandomForestClassifier
                m = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from xgboost import XGBClassifier
                m = XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
        else:
            if model_type=="linear":
                from sklearn.linear_model import LinearRegression
                m = LinearRegression()
            elif model_type=="random_forest":
                from sklearn.ensemble import RandomForestRegressor
                m = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                from xgboost import XGBRegressor
                m = XGBRegressor(random_state=42, verbosity=0)
        
        m.fit(X_train, y_train)
        y_pred_train = m.predict(X_train)
        y_pred_test = m.predict(X_test)
        
        # Метрики с кросс-валидацией
        min_cv = min(cv_folds, n_samples // 2) if n_samples >= 4 else 2
        
        if "classification" in task:
            try:
                cv_scores = cross_val_score(m, X, y, cv=min_cv, scoring='accuracy')
                cv_mean, cv_std = round(cv_scores.mean(), 4), round(cv_scores.std(), 4)
            except:
                cv_mean, cv_std = None, None
            
            self.result.metrics = {
                "accuracy_train": round(accuracy_score(y_train, y_pred_train), 4),
                "accuracy_test": round(accuracy_score(y_test, y_pred_test), 4),
                "f1_test": round(f1_score(y_test, y_pred_test, average="weighted"), 4),
                "cv_mean": cv_mean,
                "cv_std": cv_std,
            }
        else:
            try:
                cv_scores = cross_val_score(m, X, y, cv=min_cv, scoring='r2')
                cv_mean, cv_std = round(cv_scores.mean(), 4), round(cv_scores.std(), 4)
            except:
                cv_mean, cv_std = None, None
            
            self.result.metrics = {
                "r2_train": round(r2_score(y_train, y_pred_train), 4),
                "r2_test": round(r2_score(y_test, y_pred_test), 4),
                "rmse_test": round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 2),
                "cv_mean": cv_mean,
                "cv_std": cv_std,
            }
        
        if warnings_list:
            self.result.metrics["warnings"] = warnings_list
        
        # Важность признаков
        if hasattr(m, "feature_importances_"):
            imp = m.feature_importances_
            idx = np.argsort(imp)[::-1][:10]
            self.result.feature_importance = {X.columns[i]: round(imp[i], 4) for i in idx}
        
        # Сохранение модели
        fn = f"{task}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.pkl"
        self.result.model_path = str(self.model_dir / fn)
        with open(self.result.model_path, "wb") as f:
            pickle.dump(m, f)
        
        self.result.model_type = model_type
        self.result.task = task
        self.result.creation_time = datetime.now().isoformat()
        return self.result

    def get_summary(self):
        return {
            "model_type": self.result.model_type,
            "task": self.result.task,
            "metrics": self.result.metrics,
            "feature_importance": self.result.feature_importance,
            "model_path": self.result.model_path,
        }
