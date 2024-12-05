import gym
import numpy as np
import torch

# 动态规划计算最优策略
def value_iteration(env, gamma=0.99, theta=1e-8):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)  # 初始化值函数
    policy = np.zeros(n_states, dtype=int)  # 初始化策略

    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            action_values = np.zeros(n_actions)

            # 计算每个动作的值
            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state])

            # 更新值函数
            V[s] = max(action_values)
            policy[s] = np.argmax(action_values)
            delta = max(delta, abs(v - V[s]))

        # 收敛条件
        if delta < theta:
            break

    return policy

# 按最优策略采样轨迹
def generate_expert_trajectories(env, policy, num_trajectories=10, max_steps=1000):
    n_states = env.observation_space.n
    trajectories = np.zeros((num_trajectories, max_steps, n_states), dtype=np.float32)

    for traj_no in range(num_trajectories):
        state = env.reset()  # 初始化状态
        for step in range(max_steps):
            # 用独热编码表示当前状态的概率
            state_probs = np.zeros(n_states, dtype=np.float32)
            state_probs[state] = 1.0  # 当前状态的概率为 1，其余为 0
            trajectories[traj_no, step, :] = state_probs

            # 根据最优策略选择动作
            action = policy[state]
            next_state, _, done, _ = env.step(action)
            state = next_state

            if done:  # 如果到达终止状态，填充剩余步数为 0 向量
                for remaining_step in range(step + 1, max_steps):
                    trajectories[traj_no, remaining_step, :] = np.zeros(n_states, dtype=np.float32)
                break

    return trajectories

# 初始化环境
env_name = "FrozenLake-v0"
env = gym.make(env_name, is_slippery=True)

# 计算最优策略
optimal_policy = value_iteration(env)

# 生成专家轨迹数据
num_trajectories = 16  # 生成 10 条轨迹
max_steps = 1000       # 每条轨迹最多 1000 步
expert_trajectories = generate_expert_trajectories(env, optimal_policy, num_trajectories, max_steps)

# 打印结果
print("Generated expert trajectories shape:", expert_trajectories.shape)

# 保存数据
torch.save(torch.tensor(expert_trajectories), f"{env_name}.pt")
print(f"Expert trajectories saved to {env_name}.pt")
