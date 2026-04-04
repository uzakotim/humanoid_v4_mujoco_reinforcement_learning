import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os
import time

# --- Parameters ---
env_name = "Humanoid-v5"
model_path = "humanoid_policy.pth"

# Create a temporary environment to get dimensions
temp_env = gym.make(env_name)
obs_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.shape[0]
temp_env.close()

# --- Policy Network (must match training script) ---
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
           nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        x = self.fc(x)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

def visualize():
    # Set up environment with rendering enabled
    env = gym.make(env_name, render_mode="human")
    
    policy = Policy()
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        policy.load_state_dict(torch.load(model_path))
    else:
        print(f"Warning: {model_path} not found. Using random policy.")

    policy.eval()

    try:
        for episode in range(20):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32)
                    mean, std = policy(state_tensor)
                    # For visualization, we usually use the mean to show best behavior
                    action = mean.numpy()
                
                # Clip action
                action_clipped = np.clip(action, env.action_space.low, env.action_space.high)
                
                state, reward, terminated, truncated, _ = env.step(action_clipped)
                done = terminated or truncated
                total_reward += reward
                
                # Add a small delay to prevent the simulation from running too fast
                time.sleep(0.01)
                # but some versions require explicit env.render() 
                # In gymnasium v5, render_mode="human" takes care of it.
            
            print(f"Episode {episode+1} finished. Total reward: {total_reward}")
            
    finally:
        env.close()

if __name__ == "__main__":
    visualize()
