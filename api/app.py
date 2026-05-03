from flask import Flask, request, jsonify, render_template, make_response
import numpy as np
import os
import sys
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(ROOT, "src", "train.py")
MODEL_PATH = os.path.join(ROOT, "models", "q_table.npy")
PRICES = np.linspace(50, 250, 21)

import mlflow
mlflow.set_tracking_uri(f"file://{ROOT}/mlruns")

app = Flask(__name__, template_folder=os.path.join(ROOT, "templates"))

q_table = np.load(MODEL_PATH)
model_lock = threading.Lock()

last_retrain_time = None
app_start_time = time.time()
RETRAIN_INTERVAL = 120  # seconds


def retrain_job():
    global q_table, last_retrain_time
    print("[Scheduler] Retrain starting...")
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT, "--episodes", "1000"],
        cwd=ROOT,
        capture_output=False,
        text=True,
    )
    if result.returncode == 0:
        with model_lock:
            q_table = np.load(MODEL_PATH)
        last_retrain_time = time.time()
        print("[Scheduler] Retrain done, model reloaded.")
    else:
        print(f"[Scheduler] Retrain failed (exit {result.returncode})")


from apscheduler.schedulers.background import BackgroundScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(retrain_job, "interval", seconds=RETRAIN_INTERVAL)


@app.route("/")
def home():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/predict_price", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "demand" not in data or "time" not in data:
            return jsonify({"error": "Send JSON with demand (1-100) and time (0 or 1)"}), 400

        demand_raw = int(data["demand"])
        time_val = int(data["time"])

        if not (1 <= demand_raw <= 100):
            return jsonify({"error": "demand must be between 1 and 100"}), 400
        if time_val not in (0, 1):
            return jsonify({"error": "time must be 0 (off-peak) or 1 (peak)"}), 400

        demand_bin = min(demand_raw // 10, 9)
        state = (demand_bin, time_val)

        with model_lock:
            action = int(np.argmax(q_table[state]))
            q_vals = q_table[state].tolist()

        price = round(float(PRICES[action]), 2)

        return jsonify({
            "price": price,
            "explanation": {
                "state": {"demand_level": demand_bin, "time": "peak" if time_val == 1 else "off-peak"},
                "action_taken": action,
                "price_chosen": price,
                "q_values": q_vals,
            }
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reload", methods=["POST"])
def reload_model():
    global q_table
    with model_lock:
        q_table = np.load(MODEL_PATH)
    return jsonify({"status": "model reloaded"})


@app.route("/api/mlops")
def mlops_data():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("dynamic_pricing")
        if not experiment:
            return jsonify({"runs": [], "next_retrain_in": None})

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time ASC"],
        )

        result = []
        for i, run in enumerate(runs):
            result.append({
                "version": i + 1,
                "run_id": run.info.run_id,
                "timestamp": run.info.start_time,
                "status": run.info.status,
                "final_reward": run.data.metrics.get("final_avg_reward", 0),
                "episodes": run.data.params.get("episodes", "?"),
            })

        if last_retrain_time is not None:
            remaining = RETRAIN_INTERVAL - (time.time() - last_retrain_time)
        else:
            remaining = RETRAIN_INTERVAL - (time.time() - app_start_time)
        next_retrain = max(0, int(remaining))

        return jsonify({"runs": result, "next_retrain_in": next_retrain})

    except Exception as e:
        return jsonify({"error": str(e), "runs": []})


@app.route("/api/mlops/run/<run_id>")
def run_metrics(run_id):
    try:
        client = mlflow.tracking.MlflowClient()
        rewards = client.get_metric_history(run_id, "avg_reward")
        epsilons = client.get_metric_history(run_id, "epsilon")
        return jsonify({
            "rewards": [{"step": m.step, "value": round(m.value, 2)} for m in rewards],
            "epsilons": [{"step": m.step, "value": round(m.value, 4)} for m in epsilons],
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    scheduler.start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
