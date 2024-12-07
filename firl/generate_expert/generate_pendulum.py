import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import torch

def generate_expert_trajectories(env_name="Pendulum-v1", num_trajectories=10, max_steps=1000, save_path="Pendulum-v1.pt"):
    # Create the environment
    env = gym.make(env_name)

    # Train a PPO agent or load a pre-trained agent
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100_000)  # Train the agent
    # If you have a pre-trained model, load it using:
    # model = PPO.load("path_to_your_model")

    # Collect expert trajectories
    trajectories = []
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        trajectory = []
        for _ in range(max_steps):
            # Get action and probability distribution
            action, _ = model.predict(obs, deterministic=False)
            action_prob = model.policy.get_distribution(obs).probs.detach().numpy()
            trajectory.append((obs, action_prob))
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        trajectories.append(trajectory)

    # Convert to desired format
    # (N trajectories, 1000 steps, each step's state and action probabilities)
    expert_data = np.zeros((num_trajectories, max_steps, env.observation_space.shape[0] + model.policy.action_space.shape[0]))
    for i, traj in enumerate(trajectories):
        for j, (state, action_prob) in enumerate(traj):
            if j < max_steps:
                expert_data[i, j, :env.observation_space.shape[0]] = state
                expert_data[i, j, env.observation_space.shape[0]:] = action_prob

    # Save data
    torch.save(expert_data, save_path)
    print(f"Expert trajectories saved to {save_path}")

    return expert_data

# Generate and save expert trajectories
expert_trajectories = generate_expert_trajectories()

data = torch.load("Pendulum-v1.pt")
print(data.shape)  # Should print (num_trajectories, max_steps, D)
