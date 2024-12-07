import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import torch

def generate_expert_trajectories(env_name="MountainCar-v0", num_trajectories=10, max_steps=1000, save_path="expert_trajectories.pt"):
    """
    Generate expert trajectories for the MountainCar-v0 environment.
    :param env_name: Name of the environment.
    :param num_trajectories: Number of trajectories to generate.
    :param max_steps: Maximum steps per trajectory.
    :param save_path: Path to save the trajectory data.
    :return: Generated expert trajectory data.
    """
    # Create the environment
    env = gym.make(env_name)

    # Train a PPO agent or load a pre-trained agent
    model = PPO("MlpPolicy", env, verbose=1)
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
            action_probs = model.policy.get_distribution(obs).distribution.probs.detach().numpy()
            trajectory.append((obs, action_probs))
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        trajectories.append(trajectory)

    # Convert to desired format: (N trajectories, max_steps, state_dim + action_prob_dim)
    state_dim = env.observation_space.shape[0]
    action_prob_dim = env.action_space.n
    expert_data = np.zeros((num_trajectories, max_steps, state_dim + action_prob_dim))

    for i, traj in enumerate(trajectories):
        for j, (state, action_probs) in enumerate(traj):
            if j < max_steps:
                expert_data[i, j, :state_dim] = state
                expert_data[i, j, state_dim:] = action_probs

    # Save data
    torch.save(expert_data, save_path)
    print(f"Expert trajectories saved to {save_path}")

    return expert_data

# Generate and save expert trajectories
expert_trajectories = generate_expert_trajectories(env_name="MountainCar-v0")

data = torch.load("MountainCar-v0.pt")
print(data.shape)  # Should print (num_trajectories, max_steps, D)
