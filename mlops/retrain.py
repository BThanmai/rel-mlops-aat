import subprocess
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(ROOT, "src", "train.py")


def retrain():
    print("Retraining model...")
    result = subprocess.run(["python", TRAIN_SCRIPT], cwd=ROOT, capture_output=False)
    if result.returncode == 0:
        print("Retrain complete.")
    else:
        print(f"Retrain failed with exit code {result.returncode}")


if __name__ == "__main__":
    retrain()
