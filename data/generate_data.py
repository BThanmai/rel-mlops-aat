import numpy as np
import pandas as pd
import os

MIN_PRICE = 50
MAX_PRICE = 250
PRICES = np.linspace(MIN_PRICE, MAX_PRICE, 21)  # same as env.py


def generate_data(n=1000):
    rows = []
    for _ in range(n):
        time_label = np.random.choice(["peak", "off_peak"])
        time = 1 if time_label == "peak" else 0
        demand = np.random.randint(1, 100)

        # Pick a random price from the same action space as the agent
        price = float(np.random.choice(PRICES))

        # Mirror env.py demand model exactly
        base_demand = demand * (1.4 if time == 1 else 0.7)
        price_ratio = (price - MIN_PRICE) / (MAX_PRICE - MIN_PRICE)
        units_sold = max(0.0, base_demand * (1.0 - price_ratio))
        revenue = round(price * units_sold, 2)

        rows.append([time_label, demand, round(price, 2), round(units_sold, 2), revenue])

    df = pd.DataFrame(rows, columns=["time", "demand", "price", "units_sold", "revenue"])

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/data.csv", index=False)
    print(f"Generated {n} rows → data/data.csv")
    print(df.describe())


if __name__ == "__main__":
    generate_data()
