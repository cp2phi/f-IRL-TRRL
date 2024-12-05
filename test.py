import gym

# 打印已注册的环境
print([env_spec.id for env_spec in gym.envs.registry.all()])
