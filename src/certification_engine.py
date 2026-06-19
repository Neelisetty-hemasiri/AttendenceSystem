import pandas as pd
import joblib
import os

# ==================================================
# PROJECT ROOT DIRECTORY
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ==================================================
# FILE PATHS
# ==================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "Student_Attendance_Summary.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest.pkl"
)

CERTIFIED_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "certified_students.csv"
)

UNCERTIFIED_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "uncertified_students.csv"
)

REPORT_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "final_certification_report.csv"
)

# ==================================================
# LOAD DATASET
# ==================================================

print("Loading Student Attendance Dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print(f"Total Students : {len(df)}")

# ==================================================
# LOAD TRAINED MODEL
# ==================================================

print("\nLoading Random Forest Model...")

model = joblib.load(MODEL_PATH)

print("Random Forest Model Loaded Successfully!")

# ==================================================
# SELECT FEATURES
# ==================================================

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

# ==================================================
# PREDICT CERTIFICATION ELIGIBILITY
# ==================================================

print("\nPredicting Certification Eligibility...")

df['Predicted_Certified'] = model.predict(X)

print("Prediction Completed!")

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "student_predictions.csv"
)

df.to_csv(
    PREDICTION_PATH,
    index=False
)

# ==================================================
# SEPARATE CERTIFIED STUDENTS
# ==================================================

certified_students = df[
    df['Predicted_Certified'] == 1
]

# ==================================================
# SEPARATE UNCERTIFIED STUDENTS
# ==================================================

uncertified_students = df[
    df['Predicted_Certified'] == 0
]

# ==================================================
# SAVE CERTIFIED STUDENTS
# ==================================================

certified_students.to_csv(
    CERTIFIED_PATH,
    index=False
)

# ==================================================
# SAVE UNCERTIFIED STUDENTS
# ==================================================

uncertified_students.to_csv(
    UNCERTIFIED_PATH,
    index=False
)

# ==================================================
# CREATE FINAL REPORT
# ==================================================

total_students = len(df)

certified_count = len(certified_students)

uncertified_count = len(uncertified_students)

certification_rate = round(
    (certified_count / total_students) * 100,
    2
)

final_report = pd.DataFrame({
    "Total_Students": [total_students],
    "Certified_Students": [certified_count],
    "Uncertified_Students": [uncertified_count],
    "Certification_Rate (%)": [certification_rate]
})

# ==================================================
# SAVE REPORT
# ==================================================

final_report.to_csv(
    REPORT_PATH,
    index=False
)

# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n========== FINAL REPORT ==========")

print(final_report)

print("\nFiles Generated Successfully:")

print(f"\n{CERTIFIED_PATH}")
print(f"{UNCERTIFIED_PATH}")
print(f"{REPORT_PATH}")

print("\nML-Based Certification Process Completed!")