import numpy as np
import random

# Environment setup
n_states = 5          # States: 0 (start) to 4 (goal)
actions = [0, 1]      # 0: Left, 1: Right
q_table = np.zeros((n_states, len(actions)))  # Q-value table

# Hyperperameters
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 200

# Rewards
def get_reward(state):
    return 1 if state == n_states - 1 else 0

# Environment step
def step(state, action):
    if action == 1:
        next_state = min(state + 1, n_states - 1)
    else:
        next_state = max(state -1, 0)
    reward = get_reward(next_state)
    done = next_state == n_states - 1
    return next_state, reward, done

# Training loop
for episode in range(episodes):
    state = 0 # Start at state 0
    done = False

    while not done:
        # ε-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done = step(state, action)

        # Q-learning update rule
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

# Show learned Q-values
print("Learned Q-table:")
print(q_table)

# Test the agent
state = 0
print("\nTest run:")
while state != n_states - 1:
    action = np.argmax(q_table[state])
    print(f"State: {state}, Action: {'Right' if action else 'Left'}")
    state, _, _ = step(state, action)
print(f"Reached goal at state {state}!")
