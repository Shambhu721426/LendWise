# LendWise - Loan Approval Prediction System

LendWise is a Machine Learning-based Loan Approval Prediction
System developed using Python and Streamlit.

The system predicts whether a loan application is likely to
be approved or not based on applicant information, loan details
and credit history.

## Features

- Applicant Details Dashboard
- Loan Details Input
- Credit History Analysis
- Loan Approval Prediction
- Approval Probability
- Model Performance Dashboard
- Accuracy, Precision, Recall and F1 Score

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- Streamlit
- Joblib

## Machine Learning Algorithm

Logistic Regression is used as the classification algorithm.

## Dataset

Kaggle Loan Prediction Problem Dataset:

https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Download the dataset and place the training file inside:

data/train.csv

## Project Structure

lendwise-simple/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── train.csv
│   └── README.txt
│
└── models/
    ├── loan_model.joblib
    └── metrics.json

## Installation

Step 1: Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

Step 2: Open the project folder

cd lendwise-simple

Step 3: Install dependencies

pip install -r requirements.txt

Step 4: Add the dataset

Place train.csv inside the data folder.

Step 5: Train the Machine Learning model

python train.py

Step 6: Run the Streamlit application

streamlit run app.py

The application will open in your browser.

Default local address:

http://localhost:8501

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

The evaluation results are saved in:

models/metrics.json

## Disclaimer

LendWise is an educational college project.

Its predictions are not actual banking decisions and should
not be used for real loan approvals.