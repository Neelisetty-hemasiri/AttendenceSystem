import pandas as pd
import numpy as np
import os

# ==================================================
# PROJECT ROOT DIRECTORY
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "Attendance_Preprocessed_Final.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "Student_Attendance_Summary.csv"
)

# ==================================================
# LOAD DATASET
# ==================================================

print("Loading Attendance Dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset Loaded Successfully!")
print(f"Total Records: {len(df)}")

# ==================================================
# BASIC STUDENT SUMMARY
# ==================================================

print("\nCreating Student Summary...")

student_summary = (
    df.groupby('Student_ID')
      .agg(
          Student_Name=('Student_Name', 'first'),
          College_Name=('College_Name', 'first'),

          Total_Sessions=(
              'Session_ID',
              'count'
          ),

          Sessions_Attended=(
              'Attendance_Status',
              lambda x: (x == 'Present').sum()
          ),

          Average_Attendance_Minutes=(
              'Attendance_Minutes',
              'mean'
          )
      )
      .reset_index()
)

# ==================================================
# ATTENDANCE PERCENTAGE
# ==================================================

student_summary['Attendance_Percentage'] = (
    student_summary['Sessions_Attended']
    /
    student_summary['Total_Sessions']
) * 100

# ==================================================
# ATTENDANCE CONSISTENCY
# ==================================================

print("Calculating Consistency Score...")

attendance_std = (
    df.groupby('Student_ID')
      ['Attendance_Minutes']
      .std()
      .fillna(0)
      .reset_index()
)

attendance_std.columns = [
    'Student_ID',
    'Attendance_Std'
]

student_summary = student_summary.merge(
    attendance_std,
    on='Student_ID',
    how='left'
)

student_summary['Consistency_Score'] = (
    100 -
    student_summary['Attendance_Std']
)

student_summary['Consistency_Score'] = (
    student_summary['Consistency_Score']
    .clip(lower=0)
)

# ==================================================
# DOMAIN PARTICIPATION COUNT
# ==================================================

print("Calculating Domain Participation...")

domain_count = (
    df.groupby('Student_ID')
      ['Domain']
      .nunique()
      .reset_index()
)

domain_count.columns = [
    'Student_ID',
    'Domain_Participation_Count'
]

student_summary = student_summary.merge(
    domain_count,
    on='Student_ID',
    how='left'
)

# ==================================================
# ENGAGEMENT SCORE
# ==================================================

print("Calculating Engagement Score...")

student_summary['Engagement_Score'] = (
    student_summary['Attendance_Percentage'] * 0.5 +
    student_summary['Consistency_Score'] * 0.3 +
    student_summary['Domain_Participation_Count'] * 5
)

# ==================================================
# CERTIFICATION LABEL
# ==================================================

student_summary['Certified'] = np.where(
    student_summary['Attendance_Percentage'] >= 80,
    1,
    0
)

# ==================================================
# ROUND VALUES
# ==================================================

student_summary['Average_Attendance_Minutes'] = (
    student_summary['Average_Attendance_Minutes']
    .round(2)
)

student_summary['Attendance_Percentage'] = (
    student_summary['Attendance_Percentage']
    .round(2)
)

student_summary['Attendance_Std'] = (
    student_summary['Attendance_Std']
    .round(2)
)

student_summary['Consistency_Score'] = (
    student_summary['Consistency_Score']
    .round(2)
)

student_summary['Engagement_Score'] = (
    student_summary['Engagement_Score']
    .round(2)
)

# ==================================================
# SAVE OUTPUT
# ==================================================

student_summary.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\nFeature Engineering Completed Successfully!")

print(f"\nTotal Students: {len(student_summary)}")

print("\nGenerated Features:")

print("""
1. Total_Sessions
2. Sessions_Attended
3. Average_Attendance_Minutes
4. Attendance_Percentage
5. Attendance_Std
6. Consistency_Score
7. Domain_Participation_Count
8. Engagement_Score
9. Certified
""")

print(f"\nSaved To:")
print(OUTPUT_FILE)

print("\nSample Data:")

print(student_summary.head())