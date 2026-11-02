import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# File paths
DATA_PATH = "data/train.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def train_model():

    # Check dataset
    if not os.path.exists(DATA_PATH):
        print("Dataset not found!")
        print("Please place train.csv inside the data folder.")
        return

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)

    # Remove unnecessary column
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # Remove rows where target is missing
    df = df.dropna(subset=["Loan_Status"])

    # Convert target values
    df["Loan_Status"] = df["Loan_Status"].map({
        "Y": 1,
        "N": 0
    })

    # Remove invalid target values, if any
    df = df.dropna(subset=["Loan_Status"])
    df["Loan_Status"] = df["Loan_Status"].astype(int)

    # Separate features and target
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    # Complete machine learning pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(
            precision_score(y_test, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_test, y_pred, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_test, y_pred, zero_division=0), 4
        ),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        )
    }

    # Create models folder
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    # Save evaluation metrics
    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    # Display results
    print("\nModel Training Completed!")
    print("--------------------------------")
    print("Accuracy :", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall   :", metrics["recall"])
    print("F1 Score :", metrics["f1_score"])

    print("\nModel saved at:", MODEL_PATH)
    print("Metrics saved at:", METRICS_PATH)


if __name__ == "__main__":
    train_model()