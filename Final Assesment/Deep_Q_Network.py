import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Q-Network using Pytorch
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*sample)
        return (np.array(state), np.array(action),
                np.array(reward), np.array(next_state),
                np.array(done))
    
    def __len__(self):
        return len(self.buffer)
    
# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criteria = nn.MSELoss()

        self.epsilon = 1.0 # exploration rate
        self.epsilon_decay = 0.995
        self. epsion_min = 0.01
        self.gamma = 0.99 # discount factor

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.model(state).argmax().item()
        
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        state, action, reward, next_state, done = self.memory.sample(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.model(state)
        next_q_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        max_next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * max_next_q_value * (1 - done)

        loss = self.criteria(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

#Training loop 
def train_dqn(episodes=500):
    env = gym.make("CartPole-v1")
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n)
    
    for ep in range(episodes):
        state = env.reset()[0]
        total_reward = 0

        for t in range(200):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            if done:
                break

        agent.update_target()
        print(f"Episode {ep + 1}, Total reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    env.close()

if __name__ == "__main__":
    train_dqn()
