import random
from collections import defaultdict

import numpy as np


class Agent:

    def __init__(self, algorithm='Q_learning', start_epsilon=1, epsilon_decay=0.9, epsilon_cut=0.1, alpha=0.01, gamma=1,
                 nA=6):
        
        algos = {
            'Q_learning': self.step_Q_learning,
            'exp_sarsa': self.step_exp_sarsa
        }

        self.step = algos[algorithm]
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon, self.epsilon_decay, self.epsilon_cut, self.alpha, self.gamma, self.nA = start_epsilon, epsilon_decay, epsilon_cut, alpha, gamma, nA

    def select_action(self, state):
        r = random.random()
        if r > self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return random.randint(0, 5)

    def get_probs(self, Q_s, epsilon, nA):
        policy_s = np.ones(nA) * epsilon / nA
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - epsilon + (epsilon / nA)
        return policy_s

    def step_exp_sarsa(self, state, action, reward, next_state, done):
        if not done:
            probs = self.get_probs(self.Q[next_state], self.epsilon, self.nA)

            self.Q[state][action] += self.alpha * (
                        reward + self.gamma * np.dot(probs, self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            self.epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon_cut is not None:
                self.epsilon = max(self.epsilon, self.epsilon_cut)

    def step_Q_learning(self, state, action, reward, next_state, done):
        if not done:
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            self.epsilon = self.epsilon * self.epsilon_decay
            if self.epsilon_cut is not None:
                self.epsilon = max(self.epsilon, self.epsilon_cut)
