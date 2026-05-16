import numpy as np

MIN_PRICE = 50
MAX_PRICE = 250
NUM_ACTIONS = 21  # prices: 50, 60, 70, ..., 250
OFF_PEAK_MULTIPLIER = 0.7
PEAK_MULTIPLIER = 1.4

PRICES = np.linspace(MIN_PRICE, MAX_PRICE, NUM_ACTIONS)


def demand_bin(demand):
    return min(int(demand) // 10, 9)


def representative_demand(demand_level):
    if int(demand_level) == 0:
        return 5
    return min(95, int(demand_level) * 10 + 5)


def market_price(demand, time):
    """Price where the current demand context maximizes revenue."""
    time_premium = 25 if int(time) == 1 else 0
    return min(MAX_PRICE, 100 + 1.1 * int(demand) + time_premium)


def effective_demand(demand, time):
    multiplier = PEAK_MULTIPLIER if int(time) == 1 else OFF_PEAK_MULTIPLIER
    return int(demand) * multiplier


def units_sold(price, demand, time):
    base_demand = effective_demand(demand, time)
    reservation_price = market_price(demand, time)

    # Linear price response with optimum revenue at market_price(...).
    price_factor = 1.0 - (float(price) / (2.0 * reservation_price))
    return max(0.0, base_demand * price_factor)


def revenue(price, demand, time):
    return float(price) * units_sold(price, demand, time)


class PricingEnv:
    def __init__(self):
        self.demand = 50
        self.time = 0

    def reset(self):
        self.demand = np.random.randint(1, 100)
        self.time = np.random.choice([0, 1])  # 0=off_peak, 1=peak
        return (demand_bin(self.demand), self.time)

    def step(self, action):
        price = PRICES[action]
        state_demand = representative_demand(demand_bin(self.demand))
        reward = revenue(price, state_demand, self.time)

        next_state = self.reset()
        return next_state, reward, False
