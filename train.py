import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os

# --- Hyperparameters ---
learning_rate = 3e-4
gamma = 0.99
clip_epsilon = 0.2
ppo_epochs = 10
max_steps = 1000
env_name = "Humanoid-v5"
def evaluate_policy(eval_policy, eval_env, steps):
    eval_policy.eval()
    state, _ = eval_env.reset()
    total_reward = 0
    for _ in range(steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            mean, _ = eval_policy(state_tensor)
            action = mean.detach().numpy()
        action_clipped = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)
        state, reward, terminated, truncated, _ = eval_env.step(action_clipped)
        total_reward += reward
        if terminated or truncated:
            break
    eval_policy.train()
    return total_reward
# --- Custom standing environment wrapper ---
class StandHumanoidWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        torso_height = obs[0]  # z-coordinate/height of torso
        
        # 1. Standing Reward (Encourage staying upright)
        # torso_height is ~1.0-1.2 when standing
        standing_reward = max(0, torso_height - 0.5)
        
        # 2. Control Penalty (Encourage efficiency and reduce jitter)
        # Penalizes large action magnitudes
        control_penalty = 0.01 * np.sum(np.square(action))
        
        # 3. Position Penalty (Encourage staying on the spot)
        # Extracts x and y position from info dictionary to discourage drifting
        x = info.get("x_position", 0.0)
        y = info.get("y_position", 0.0)
        position_penalty = 0.1 * (x**2 + y**2)
        
        # Total reward combines these components
        reward = standing_reward - control_penalty - position_penalty
        
        return obs, reward, terminated, truncated, info

env = StandHumanoidWrapper(gym.make(env_name))
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# --- Policy Network ---
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
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

policy = Policy()
model_path = "humanoid_policy.pth"
if os.path.exists(model_path):
    print(f"Loading existing policy from {model_path}...")
    policy.load_state_dict(torch.load(model_path))
    print("Evaluating loaded policy for baseline...")
    best_reward = evaluate_policy(policy, env, max_steps)
    print(f"Loaded model baseline reward: {best_reward:.2f}")
else:
    best_reward = -float('inf')

optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# --- Helper ---
def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)



# --- Training loop ---
for episode in range(50000):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    states = []
    actions = []

    for t in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        mean, std = policy(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, float(env.action_space.low[0]), float(env.action_space.high[0]))
        
        next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().numpy())
        done = terminated or truncated
        
        log_probs.append(dist.log_prob(action).sum())
        rewards.append(reward)
        states.append(state_tensor)
        actions.append(action)
        
        state = next_state
        if done:
            break

    episode_reward = sum(rewards)
    returns = compute_returns(rewards, gamma)
    log_probs = torch.stack(log_probs).detach()
    states = torch.stack(states)
    actions = torch.stack(actions).detach()

    # --- PPO update ---
    for _ in range(ppo_epochs):
        mean, std = policy(states)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=1)
        ratio = (new_log_probs - log_probs).exp()
        loss = -torch.min(ratio * returns, torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * returns).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if episode % 5 == 0:
        print(f"Episode {episode}, total standing reward: {episode_reward}")
    
    # --- Save best model ---
    if episode_reward > best_reward:
        if os.path.exists(model_path):
            # Run an episode with the existing model to get a current baseline
            temp_policy = Policy()
            temp_policy.load_state_dict(torch.load(model_path))
            baseline_reward = evaluate_policy(temp_policy, env, max_steps)
            
            if episode_reward > baseline_reward:
                best_reward = episode_reward
                torch.save(policy.state_dict(), model_path)
                print(f"New best model saved! (Episode: {episode}, New: {episode_reward:.2f} > Baseline: {baseline_reward:.2f})")
            else:
                best_reward = baseline_reward
                print(f"New score {episode_reward:.2f} didn't beat baseline {baseline_reward:.2f}. Not saving.")
        else:
            best_reward = episode_reward
            torch.save(policy.state_dict(), model_path)
            print(f"Initial model saved! (Episode: {episode}, Reward: {episode_reward:.2f})")

env.close()