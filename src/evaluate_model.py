import pandas as pd
import joblib
import os

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "Student_Attendance_Summary.csv"
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

models = {
    "Logistic Regression":
    joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "logistic_regression.pkl"
        )
    ),

    "Decision Tree":
    joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "decision_tree.pkl"
        )
    ),

    "Random Forest":
    joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "random_forest.pkl"
        )
    )
}

results = []

for name, model in models.items():

    predictions = model.predict(X_test)

    results.append([
        name,
        accuracy_score(y_test, predictions),
        precision_score(y_test, predictions),
        recall_score(y_test, predictions),
        f1_score(y_test, predictions)
    ])

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

best_model = results_df.sort_values(
    by='Accuracy',
    ascending=False
).iloc[0]

print("\nBest Model:")
print(best_model)

from sklearn.metrics import classification_report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions
    )
)

results_df.to_csv(
    os.path.join(
        BASE_DIR,
        "outputs",
        "model_evaluation_report.csv"
    ),
    index=False
)

# ==================================================
# BEST MODEL SUMMARY
# ==================================================

best_model = results_df.sort_values(
    by='Accuracy',
    ascending=False
).iloc[0]

best_model_file = os.path.join(
    BASE_DIR,
    "outputs",
    "best_model_summary.txt"
)

with open(
    best_model_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("BEST MODEL SUMMARY\n")
    f.write("=" * 40 + "\n\n")

    f.write(
        f"Model      : {best_model['Model']}\n"
    )

    f.write(
        f"Accuracy   : {best_model['Accuracy']:.4f}\n"
    )

    f.write(
        f"Precision  : {best_model['Precision']:.4f}\n"
    )

    f.write(
        f"Recall     : {best_model['Recall']:.4f}\n"
    )

    f.write(
        f"F1 Score   : {best_model['F1 Score']:.4f}\n"
    )

print("Best Model Summary Saved")

rf_model = models["Random Forest"]

predictions = rf_model.predict(X_test)

# ==================================================
# CLASSIFICATION REPORT
# ==================================================

report = classification_report(
    y_test,
    predictions
)

report_file = os.path.join(
    BASE_DIR,
    "outputs",
    "classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RANDOM FOREST CLASSIFICATION REPORT\n"
    )

    f.write("=" * 50 + "\n\n")

    f.write(report)

print("Classification Report Saved")

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    os.path.join(
        BASE_DIR,
        "visualizations",
        "confusion_matrix.png"
    )
)

plt.close()

print(results_df)
print("\nEvaluation Completed")