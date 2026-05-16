import numpy as np
import pandas as pd
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from env import PRICES, revenue, units_sold


def generate_data(n=1000):
    rows = []
    for _ in range(n):
        time_label = np.random.choice(["peak", "off_peak"])
        time = 1 if time_label == "peak" else 0
        demand = np.random.randint(1, 100)

        # Pick a random price from the same action space as the agent
        price = float(np.random.choice(PRICES))

        sold = units_sold(price, demand, time)
        earned = round(revenue(price, demand, time), 2)

        rows.append([time_label, demand, round(price, 2), round(sold, 2), earned])

    df = pd.DataFrame(rows, columns=["time", "demand", "price", "units_sold", "revenue"])

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/data.csv", index=False)
    print(f"Generated {n} rows → data/data.csv")
    print(df.describe())


if __name__ == "__main__":
    generate_data()
