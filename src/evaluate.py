import numpy as np
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import PricingEnv, PRICES

model_path = os.path.join(ROOT, "models", "q_table.npy")
q_table = np.load(model_path)
env = PricingEnv()

total_reward = 0
for _ in range(100):
    state = env.reset()
    action = int(np.argmax(q_table[state]))
    _, reward, _ = env.step(action)
    total_reward += reward

print(f"Average Reward over 100 episodes: {total_reward / 100:.2f}")

# Show what price the AI picks for each state
print("\nLearned Pricing Table:")
print(f"{'Demand Bin':<12} {'Time':<10} {'Chosen Price':>12}")
print("-" * 36)
for d in range(10):
    for t in range(2):
        action = int(np.argmax(q_table[d, t]))
        price = PRICES[action]
        time_label = "peak" if t == 1 else "off-peak"
        print(f"  {d:<10} {time_label:<10} ₹{price:>8.0f}")
