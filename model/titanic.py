"""Titanic survival model used by /api/titanic/predict."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


class TitanicModel:
    """Singleton Titanic model that trains once and serves many predictions."""

    _instance = None

    def __init__(self) -> None:
        self.model = None
        self.dt = None
        self.features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "alone"]
        self.target = "survived"
        self.encoder = OneHotEncoder(handle_unknown="ignore")
        self.titanic_data = self._load_dataset()

    def _load_dataset(self) -> pd.DataFrame:
        """Prefer local dataset, fallback to seaborn loader."""
        local_csv = Path(__file__).resolve().parent.parent / "datasets" / "titanic.csv"
        if local_csv.exists():
            return pd.read_csv(local_csv)
        return sns.load_dataset("titanic")

    @staticmethod
    def _scalar(value: Any, default: Any = None) -> Any:
        """Normalize list-style and scalar-style payload values to scalars."""
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            return value[0] if len(value) > 0 else default
        return default if value is None else value

    def _clean(self) -> None:
        drop_cols = ["alive", "who", "adult_male", "class", "embark_town", "deck"]
        existing = [c for c in drop_cols if c in self.titanic_data.columns]
        if existing:
            self.titanic_data.drop(existing, axis=1, inplace=True)

        self.titanic_data["sex"] = self.titanic_data["sex"].apply(lambda x: 1 if str(x).lower() == "male" else 0)
        self.titanic_data["alone"] = self.titanic_data["alone"].apply(lambda x: 1 if bool(x) else 0)

        self.titanic_data.dropna(subset=["embarked"], inplace=True)
        onehot = self.encoder.fit_transform(self.titanic_data[["embarked"]]).toarray()
        cols = ["embarked_" + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols, index=self.titanic_data.index)

        self.titanic_data = pd.concat([self.titanic_data, onehot_df], axis=1)
        self.titanic_data.drop(["embarked"], axis=1, inplace=True)
        self.features.extend(cols)
        self.titanic_data.dropna(inplace=True)

    def _train(self) -> None:
        X = self.titanic_data[self.features]
        y = self.titanic_data[self.target]

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

        self.dt = DecisionTreeClassifier(random_state=42)
        self.dt.fit(X, y)

    @classmethod
    def get_instance(cls) -> "TitanicModel":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._clean()
            cls._instance._train()
        return cls._instance

    def _normalize_passenger(self, passenger: Dict[str, Any]) -> Dict[str, Any]:
        embarked = str(self._scalar(passenger.get("embarked", "S"), "S")).strip().upper()
        if embarked not in {"C", "Q", "S"}:
            embarked = "S"

        sex = str(self._scalar(passenger.get("sex", "male"), "male")).strip().lower()
        if sex not in {"male", "female"}:
            sex = "male"

        alone_raw = self._scalar(passenger.get("alone", False), False)
        if isinstance(alone_raw, str):
            alone = alone_raw.strip().lower() in {"true", "1", "yes", "y"}
        else:
            alone = bool(alone_raw)

        def to_int(key: str, default: int, min_val: int = 0, max_val: int = 200) -> int:
            val = self._scalar(passenger.get(key, default), default)
            return max(min_val, min(max_val, int(float(val))))

        def to_float(key: str, default: float, min_val: float = 0.0, max_val: float = 10000.0) -> float:
            val = self._scalar(passenger.get(key, default), default)
            return max(min_val, min(max_val, float(val)))

        return {
            "pclass": to_int("pclass", 3, 1, 3),
            "sex": sex,
            "age": to_float("age", 30.0, 0.0, 100.0),
            "sibsp": to_int("sibsp", 0, 0, 20),
            "parch": to_int("parch", 0, 0, 20),
            "fare": to_float("fare", 30.0, 0.0, 10000.0),
            "embarked": embarked,
            "alone": alone,
        }

    def predict(self, passenger: Dict[str, Any]) -> Dict[str, float]:
        cleaned = self._normalize_passenger(passenger)

        passenger_df = pd.DataFrame(cleaned, index=[0])
        passenger_df["sex"] = passenger_df["sex"].apply(lambda x: 1 if x == "male" else 0)
        passenger_df["alone"] = passenger_df["alone"].apply(lambda x: 1 if x else 0)

        onehot = self.encoder.transform(passenger_df[["embarked"]]).toarray()
        cols = ["embarked_" + str(val) for val in self.encoder.categories_[0]]
        onehot_df = pd.DataFrame(onehot, columns=cols, index=passenger_df.index)

        passenger_df = pd.concat([passenger_df, onehot_df], axis=1)
        passenger_df.drop(["embarked"], axis=1, inplace=True)

        passenger_df = passenger_df.reindex(columns=self.features, fill_value=0)

        die, survive = np.squeeze(self.model.predict_proba(passenger_df))
        return {"die": float(die), "survive": float(survive)}

    def feature_weights(self) -> Dict[str, float]:
        importances = self.dt.feature_importances_
        return {feature: float(importance) for feature, importance in zip(self.features, importances)}


def initTitanic() -> None:
    """Initialize Titanic model in memory at startup."""
    TitanicModel.get_instance()
