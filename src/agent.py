import numpy as np
from env import NUM_ACTIONS

DEMAND_BINS = 10
TIME_STATES = 2


class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((DEMAND_BINS, TIME_STATES, NUM_ACTIONS))
        self.lr = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9999  # slower decay → proper exploration across all 5000 episodes

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (
            reward + self.gamma * best_next - self.q_table[state][action]
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
