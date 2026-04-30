import numpy as np

MIN_PRICE = 50
MAX_PRICE = 250
NUM_ACTIONS = 21  # prices: 50, 60, 70, ..., 250

PRICES = np.linspace(MIN_PRICE, MAX_PRICE, NUM_ACTIONS)


class PricingEnv:
    def __init__(self):
        self.demand = 50
        self.time = 0

    def reset(self):
        self.demand = np.random.randint(1, 100)
        self.time = np.random.choice([0, 1])  # 0=off_peak, 1=peak
        demand_bin = min(self.demand // 10, 9)
        return (demand_bin, self.time)

    def step(self, action):
        price = PRICES[action]

        # Peak hours boost demand, off-peak reduces it
        base_demand = self.demand * (1.4 if self.time == 1 else 0.7)

        # Linear elasticity: higher price reduces units sold
        price_ratio = (price - MIN_PRICE) / (MAX_PRICE - MIN_PRICE)
        units_sold = base_demand * (1.0 - price_ratio)
        units_sold = max(0.0, units_sold)

        reward = price * units_sold

        next_state = self.reset()
        return next_state, reward, False
