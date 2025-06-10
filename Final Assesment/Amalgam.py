import torch
import gym
import random
import tkinter as tk
from tkinter import messagebox
from torchvision import models, transforms
from PIL import Image
import pyttsx3
import numpy as np

# Dummy environment (Grid world)
class SimpleEnv:
    def __init__(self):
        self.state = (0, 0)
        self.goal = (4, 4)
        self.objects = {(2, 2): "obstacle", (3, 3): "target"}
        self.grid_size = 5

    def reset (self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 0: x = min(x + 1, self.grid_size - 1)
        if action == 1: x = max(x - 1, 0)
        if action == 2: y = min(y + 1, self.grid_size - 1)
        if action == 3: y = max(y - 1, 0)
        self.state = (x, y)
        reward = -1
        done = False
        obj = self.objects.get(self.state, None)
        if obj == "target":
            reward = 10
            done = True
        elif obj == "obstacle":
            reward = -10
        return self.state, reward, done, obj

# RL Agent (DQN-like logic)
class Agent:
    def __init__(self):
        self.q_table = {}

    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]
        return np.argmax(self.q_table[state]) if random.random() > 0.2 else random.randint(0, 3)
    
    def update( self, state, action, reward, next_state):
        self.q_table.setdefault(state, [0]*4)
        self.q_table.setdefault(next_state, [0]*4)
        lr = 0.1
        gamma = 0.9
        self.q_table[state][action] += lr * (reward + gamma * max(self.q_table[next_state]) - self.q_table[state][action])

# Voice Explanation
engine = pyttsx3.init()
def speak(text):
    print(f"[Agent Says]: {text}")
    engine.say(text)
    engine.runAndWait()

# Gui and Feedback
class App:
    def __init__(self, master, env, agent):
        self.master = master
        self.env = env
        self.agent = agent
        self. state = self. env.reset()
        self.label = tk.Label(master, text="Initializing...")
        self.label.pack()
        self.feedback = None

        self.btn_good = tk.Button(master, text="👍 Good", command=lambda: self.send_feedback(1))
        self.btn_bad = tk.Button(master, text="👎 Bad", command=lambda: self.send_feedback(-1))
        self.btn_next = tk.Button(master, text="Next Step", command=self.step)
        self.btn_next.pack()
        self.btn_good.pack()
        self.btn_bad.pack()
    
    def send_feedback(self, val):
        self.feedback = val
        speak("Thanks for the feedback!")

    def step(self):
        action = self.agent.get_action(self.state)
        next_state, reward, done, obj = self.env.step(action)

        explanation = f"I moved to {next_state}."
        if obj:
            explanation += f" I see a {obj} here."

        if self.feedback:
            reward += self.feedback
            self.feedback = None

        self .agent.update(self.state, action, reward, next_state)
        self.state = next_state
        self.label.config(text=f"Agent at {self.state} | {explanation}")
        speak(explanation)

        if done:
            speak("Goal reached! Resetting...")
            self.env.reset()

# Run Everything
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Autonomous Agent Interface")
    env = SimpleEnv()
    agent = Agent()
    app = App(root, env, agent)
    root.mainloop()