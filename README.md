# House Price Prediction Pipeline

A beginner-friendly machine-learning project that predicts a home's price from its size, bedrooms, bathrooms, age, and city.

This is a strong first GitHub project for the first four days of an AI Engineer roadmap:

| Roadmap day | What this project demonstrates |
| --- | --- |
| Day 1 | Artificial intelligence and machine-learning fundamentals |
| Day 2 | Python scripts, functions, and command-line usage |
| Day 3 | Data cleaning with Pandas and numerical data handling |
| Day 4 | Training a machine-learning model to make a prediction |

## What the project does

1. Loads house data from a CSV file.
2. Removes rows where the target price is missing.
3. Fills missing input values using sensible defaults.
4. Converts city names into a format the model can use.
5. Trains a linear-regression model from past examples.
6. Evaluates the model and saves it for predictions.

The idea is simple: **past house examples → pattern → price prediction**.

## Project structure

```text
house-price-predictor/
├── data/house_prices.csv
├── src/train.py
├── src/predict.py
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-github-repository-url>
cd house-price-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Train the model

```bash
python src/train.py
```

This creates:

- `models/house_price_model.joblib` — the trained model
- `models/metrics.json` — model evaluation results

## Make a prediction

```bash
python src/predict.py --size 1500 --bedrooms 3 --bathrooms 2 --age 10 --city Chicago
```

## Push to GitHub

```bash
git init
git add .
git commit -m "Add house price prediction ML pipeline"
git branch -M main
git remote add origin <your-github-repository-url>
git push -u origin main
```

## Next improvements

- Replace the sample CSV with a larger real dataset.
- Add charts for the data and prediction errors.
- Add unit tests.
- Build a Streamlit or FastAPI interface.
- Deploy the model as an API.

## Project Presentation

This presentation explains the House Price Prediction
machine learning project, including its business problem,
technologies, architecture, training process and results.

[Download Project PowerPoint](docs/house_price_predictor_overview.pptx)
