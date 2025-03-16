#Use Case: Decision making in uncertain environments.
#Example: Teaching an agent to play a game like Tic-Tac-Toe or navigating a maze.

import numpy as np
import random

# Define the environment (simplified for Tic-Tac-Toe)
# States, actions, and rewards are predefined here as an example
actions = [0, 1, 2]  # Example actions: move in three directions
q_table = np.zeros((3, 3))  # Initialize Q-table

# Q-Learning parameters
learning_rate = 0.1
discount_factor = 0.9
episodes = 1000

# Q-Learning algorithm
for episode in range(episodes):
    state = random.choice([0, 1, 2])  # Random start state
    for _ in range(100):  # 100 steps per episode
        action = random.choice(actions)  # Random action for simplicity
        reward = random.randint(-1, 1)  # Random reward
        next_state = random.choice([0, 1, 2])  # Next state
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state]))
        state = next_state  # Move to the next state

print("Q-table after training:")
print(q_table)
