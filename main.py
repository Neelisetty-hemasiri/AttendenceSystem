import subprocess
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "project_execution_log.txt"
)

with open(LOG_FILE, "w", encoding="utf-8") as log:

    start_time = datetime.now()

    header = (
        "=" * 50 +
        "\nML Attendance Certification System\n" +
        "=" * 50 +
        f"\nExecution Started: {start_time}\n\n"
    )

    print(header)
    log.write(header)

    steps = [
        ("Feature Engineering", "src/feature_engineering.py"),
        ("Model Training", "src/train_model.py"),
        ("Model Evaluation", "src/evaluate_model.py"),
        ("Certification Generation", "src/certification_engine.py")
    ]

    for step_name, script in steps:

        separator = f"\n{'='*50}\n{step_name}\n{'='*50}\n"

        print(separator)
        log.write(separator)

        result = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True
        )

        print(result.stdout)
        log.write(result.stdout)

        if result.stderr:
            print(result.stderr)
            log.write("\nERROR:\n")
            log.write(result.stderr)

    end_time = datetime.now()

    footer = (
        f"\nExecution Completed: {end_time}\n"
        f"Total Duration: {end_time - start_time}\n"
    )

    print(footer)
    log.write(footer)

print(f"\nExecution log saved to:\n{LOG_FILE}")