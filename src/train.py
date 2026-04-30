import numpy as np
import sys
import os
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlflow

from env import PricingEnv
from agent import QLearningAgent

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=5000)
args = parser.parse_args()
episodes = args.episodes

mlflow.set_tracking_uri(f"file://{ROOT}/mlruns")
mlflow.set_experiment("dynamic_pricing")

env = PricingEnv()
agent = QLearningAgent()

checkpoint = max(1, episodes // 10)  # log 10 points per run

with mlflow.start_run() as run:
    run_id = run.info.run_id

    mlflow.log_params({
        "lr": agent.lr,
        "gamma": agent.gamma,
        "epsilon_decay": agent.epsilon_decay,
        "epsilon_min": agent.epsilon_min,
        "episodes": episodes,
    })

    recent_rewards = []

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        for _ in range(50):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            ep_reward += reward
        recent_rewards.append(ep_reward)

        if (ep + 1) % checkpoint == 0:
            avg = float(np.mean(recent_rewards[-checkpoint:]))
            mlflow.log_metric("avg_reward", avg, step=ep + 1)
            mlflow.log_metric("epsilon", agent.epsilon, step=ep + 1)
            print(f"Episode {ep+1}/{episodes}  avg_reward={avg:.1f}  epsilon={agent.epsilon:.3f}")

    final_avg = float(np.mean(recent_rewards[-checkpoint:]))
    mlflow.log_metric("final_avg_reward", final_avg)

    model_path = os.path.join(ROOT, "models", "q_table.npy")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    np.save(model_path, agent.q_table)

    # Save versioned archive copy, keep only 10 most recent
    archive_dir = os.path.join(ROOT, "models", "archive")
    os.makedirs(archive_dir, exist_ok=True)
    np.save(os.path.join(archive_dir, f"q_table_{run_id[:8]}.npy"), agent.q_table)

    # Keep only the 10 best runs by reward; delete the rest from MLflow and archive
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("dynamic_pricing")
    if experiment:
        all_runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["metrics.final_avg_reward DESC"],
        )
        top10_ids = {r.info.run_id[:8] for r in all_runs[:10]}
        for r in all_runs[10:]:
            client.delete_run(r.info.run_id)
        for fname in os.listdir(archive_dir):
            if fname.endswith(".npy") and fname.replace("q_table_", "").replace(".npy", "") not in top10_ids:
                os.remove(os.path.join(archive_dir, fname))

    mlflow.log_artifact(model_path)

    print(f"Training complete. Final avg reward: {final_avg:.2f}  Run: {run_id[:8]}")
