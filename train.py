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
num_envs = 8  # Number of parallel simulations

# --- Custom standing environment wrapper ---
class StandHumanoidWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        torso_height = obs[0]  # z-coordinate/height of torso
        
        # 1. Standing Reward (Encourage staying upright)
        standing_reward = max(0, torso_height - 1.0)
        
        # 2. Control Penalty (Encourage efficiency)
        control_penalty = 0.01 * np.sum(np.square(action))
        
        # 3. Position Penalty (Encourage staying on the spot)
        x = info.get("x_position", 0.0)
        y = info.get("y_position", 0.0)
        position_penalty = 0.1 * (x**2 + y**2)
        
        reward = standing_reward - control_penalty - position_penalty
        return obs, reward, terminated, truncated, info

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = StandHumanoidWrapper(env)
        # Note: seed is set for each env instance
        env.action_space.seed(seed)
        return env
    return thunk

def evaluate_policy(eval_policy, eval_env, steps):
    eval_policy.eval()
    state, _ = eval_env.reset()
    total_reward = 0
    device = next(eval_policy.parameters()).device
    for _ in range(steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            mean, _ = eval_policy(state_tensor)
            action = mean.cpu().detach().numpy()
        action_clipped = np.clip(action, eval_env.action_space.low, eval_env.action_space.high)
        state, reward, terminated, truncated, _ = eval_env.step(action_clipped)
        total_reward += reward
        if terminated or truncated:
            break
    eval_policy.train()
    return total_reward


# --- Policy Network ---
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim):
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
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.mean = nn.Linear(512, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        x = self.fc(x)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

# --- Helper ---
def compute_returns_vectorized(rewards, dones, gamma, n_envs):
    device = rewards.device
    returns = torch.zeros_like(rewards)
    R = torch.zeros(n_envs, device=device)
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * (1 - dones[t])
        returns[t] = R
    return returns

# --- Main Execution ---
if __name__ == "__main__":
    # Create vectorized environments for training
    envs = gym.vector.AsyncVectorEnv([make_env(env_name, i) for i in range(num_envs)])
    # Create a single environment for evaluation
    eval_env = StandHumanoidWrapper(gym.make(env_name))

    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    print(obs_dim, action_dim)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    policy = Policy(obs_dim, action_dim).to(device)
    model_path = "humanoid_policy.pth"
    
    if os.path.exists(model_path):
        print(f"Loading existing policy from {model_path}...")
        policy.load_state_dict(torch.load(model_path))
        best_reward = evaluate_policy(policy, eval_env, max_steps)
        print(f"Loaded model baseline reward: {best_reward:.2f}")
    else:
        best_reward = -float('inf')

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # --- Training loop ---
    next_obs, _ = envs.reset()
    for update in range(1, 50000):
        all_states = []
        all_actions = []
        all_log_probs = []
        all_rewards = []
        all_dones = []

        # Rollout phase: Collect data from all environments
        for t in range(max_steps):
            state_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                mean, std = policy(state_tensor)
                dist = Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Step environments
            action_np = action.cpu().numpy()
            action_clipped = np.clip(action_np, envs.single_action_space.low, envs.single_action_space.high)
            
            next_obs, reward, terminated, truncated, _ = envs.step(action_clipped)
            done = np.logical_or(terminated, truncated)

            all_states.append(state_tensor)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
            all_dones.append(torch.tensor(done, dtype=torch.float32).to(device))

        # Convert to tensors and flatten for PPO update
        states = torch.cat(all_states)
        actions = torch.cat(all_actions)
        log_probs = torch.cat(all_log_probs).detach()
        rewards = torch.stack(all_rewards)
        dones = torch.stack(all_dones)
        
        returns = compute_returns_vectorized(rewards, dones, gamma, num_envs).flatten()

        # --- PPO update ---
        for _ in range(ppo_epochs):
            mean, std = policy(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = (new_log_probs - log_probs).exp()
            
            # Simple policy gradient with clipped objective
            loss = -torch.min(ratio * returns, torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * returns).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if update % 5 == 0:
            avg_reward = evaluate_policy(policy, eval_env, max_steps)
            print(f"Update {update}, Mean Evaluation Reward: {avg_reward:.2f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(policy.state_dict(), model_path)
                print(f"New best model saved! (Update: {update}, Reward: {avg_reward:.2f})")

    envs.close()
    eval_env.close()