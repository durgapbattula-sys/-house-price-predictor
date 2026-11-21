"""Make one house-price prediction with the trained model."""

from pathlib import Path
import argparse

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "house_price_model.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a house price.")
    parser.add_argument("--size", type=float, required=True, help="Square feet")
    parser.add_argument("--bedrooms", type=int, required=True)
    parser.add_argument("--bathrooms", type=float, required=True)
    parser.add_argument("--age", type=float, required=True, help="Age in years")
    parser.add_argument("--city", required=True)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python src/train.py")

    model = joblib.load(MODEL_PATH)
    home = pd.DataFrame(
        [{
            "size_sqft": args.size,
            "bedrooms": args.bedrooms,
            "bathrooms": args.bathrooms,
            "age_years": args.age,
            "city": args.city,
        }]
    )
    predicted_price = model.predict(home)[0]
    print(f"Estimated house price: ${predicted_price:,.0f}")


if __name__ == "__main__":
    main()
