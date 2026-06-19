import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "Student_Attendance_Summary.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

df = pd.read_csv(DATA_FILE)

X = df[
[
    'Total_Sessions',
    'Sessions_Attended',
    'Average_Attendance_Minutes',
    'Consistency_Score',
    'Domain_Participation_Count',
    'Engagement_Score'
]
]

y = df['Certified']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

log_model = LogisticRegression()
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

log_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

joblib.dump(
    log_model,
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

joblib.dump(
    dt_model,
    os.path.join(MODEL_DIR, "decision_tree.pkl")
)

joblib.dump(
    rf_model,
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

print("All Models Saved Successfully")