"""Train a beginner-friendly house price prediction model."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "house_prices.csv"
MODEL_DIR = PROJECT_ROOT / "models"


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the CSV and remove rows without the value we want to predict."""
    data = pd.read_csv(path)
    return data.dropna(subset=["price_usd"]).copy()


def build_pipeline() -> Pipeline:
    numeric_features = ["size_sqft", "bedrooms", "bathrooms", "age_years"]
    categorical_features = ["city"]

    numeric_steps = Pipeline(
        [("fill_missing_numbers", SimpleImputer(strategy="median"))]
    )
    categorical_steps = Pipeline(
        [
            ("fill_missing_city", SimpleImputer(strategy="most_frequent")),
            ("encode_city", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    prepare_data = ColumnTransformer(
        [
            ("numbers", numeric_steps, numeric_features),
            ("city", categorical_steps, categorical_features),
        ]
    )

    return Pipeline([("prepare_data", prepare_data), ("model", LinearRegression())])


def main() -> None:
    data = load_and_clean_data(DATA_PATH)
    features = data.drop(columns="price_usd")
    target = data["price_usd"]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "mean_absolute_error_usd": round(mean_absolute_error(y_test, predictions), 2),
        "r2_score": round(r2_score(y_test, predictions), 3),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "house_price_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Model trained successfully.")
    print(json.dumps(metrics, indent=2))
    print("Saved model to models/house_price_model.joblib")


if __name__ == "__main__":
    main()
