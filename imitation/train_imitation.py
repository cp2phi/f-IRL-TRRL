import tempfile

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms.bc import BC
from imitation.algorithms.sqil import SQIL
from stable_baselines3.common.vec_env import DummyVecEnv
from ruamel.yaml import YAML
import sys, os, time
from imitation.policies.serialize import load_policy
import datetime
import dateutil.tz
import json
import torch.utils.tensorboard as tb
from utils import system, collect, logger, eval
from imitation.data import rollout
from stable_baselines3.ppo import MlpPolicy
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.dagger import SimpleDAggerTrainer
import logging
from gym.wrappers import TimeLimit

logging.basicConfig(level=logging.ERROR)

def calculate_record(policy, agent, iteration):
    env.seed(seed)
    reward, _ = evaluate_policy(policy, env, 1)

    obs = transitions.obs
    acts = transitions.acts

    obs_th = torch.as_tensor(obs, device=device)
    acts_th = torch.as_tensor(acts, device=device)

    agent.policy.to(device)
    expert.policy.to(device)

    input_values, input_log_prob, input_entropy = agent.policy.evaluate_actions(obs_th, acts_th)
    target_values, target_log_prob, target_entropy = expert.policy.evaluate_actions(obs_th, acts_th)

    distance = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob))

    distance = float(distance)
    reward = float(reward)
    # distance = torch.nn.functional.kl_div(input_values,target_values)

    # record to TensorBoard and logger
    writer.add_scalar(env_name + "/distance", distance, iteration)
    writer.add_scalar(env_name + "/reward", reward, iteration)

    logger.record_tabular("iteration", iteration)
    logger.record_tabular("distance", distance)
    logger.record_tabular("reward", reward)
    logger.dump_tabular()

    print(f"[{algorithm}] Iteration {iteration}: KL={distance}, Reward={reward}")

    return None


def GAIL_train():
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=64,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        gen_train_timesteps=300,
        # init_tensorboard=True,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    print(f"Training with {algorithm}...")
    for iteration in range(n_itrs):
        gail_trainer.train(300)
        calculate_record(gail_trainer.gen_algo, gail_trainer.gen_algo, iteration)

    return None


def AIRL_train():
    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=64,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
        gen_train_timesteps=300,
        allow_variable_horizon=True,
    )

    print(f"Training with {algorithm}...")
    for iteration in range(n_itrs):
        airl_trainer.train(300)
        calculate_record(airl_trainer.gen_algo, airl_trainer.gen_algo, iteration)

    return None


def BC_train():
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size= 64,
        demonstrations=transitions,
        rng=rng,
        device='cpu'
    )

    print(f"Training with {algorithm}...")
    for iteration in range(n_itrs):
        bc_trainer.train(n_epochs=300)
        calculate_record(bc_trainer.policy, bc_trainer, iteration)

    return None

def Dagger_train():
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=64,
        demonstrations=transitions,
        rng=rng,
        device='cpu'
    )

    print(f"Training with {algorithm}...")
    for iteration in range(n_itrs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=env,
                scratch_dir=tmpdir,
                expert_policy=expert,
                bc_trainer=bc_trainer,
                rng=rng)
            dagger_trainer.train(300)
        calculate_record(dagger_trainer.policy, dagger_trainer, iteration)

    return None


if __name__ == "__main__":
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']
    n_itrs = v['irl']['n_itrs']
    algorithm = v['obj']
    rng = np.random.default_rng(seed)

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    print("device:", device)
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid = os.getpid()

    # assumptions
    assert algorithm in ['GAIL', 'AIRL', 'BC', 'SQIL', 'Dagger', 'BIRL']

    # logs
    exp_id = f"logs//{env_name}/exp-imitatioon-{num_expert_trajs}/{algorithm}"  # task/obj/date structure
    if not os.path.exists(exp_id):
        os.makedirs(exp_id)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)
    print(f"Logging to directory: {log_folder}")

    os.system(f'cp imitation/train_imitation.py {log_folder}')
    os.system(f'cp {sys.argv[1]} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'))
    os.makedirs(os.path.join(log_folder, 'model'))
    os.makedirs(os.path.join(log_folder, env_name + "_" + algorithm))

    # environment
    env_fn = lambda: gym.make(env_name)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    env = make_vec_env(
        env_name,
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    )

    # tensorboard
    global writer
    writer = tb.SummaryWriter(log_folder + '/' + env_name + "_" + algorithm, flush_secs=1)

    #expert transitions
    expert = PPO.load(f"./imitation/imitation_expert/{env_name}")

    # rollouts = rollout.rollout(
    #     expert,
    #     env,
    #     rollout.make_sample_until(min_timesteps=16*128, min_episodes=16),
    #     rng=rng,
    # )
    # transitions = rollout.flatten_trajectories(rollouts)
    #
    # torch.save(transitions,f"./imitation/imitation_expert/transitions_{env_name}.npy")
    # torch.save(rollouts,f"./imitation/imitation_expert/rollouts_{env_name}.npy")

    transitions = torch.load(f"./imitation/imitation_expert/transitions_{env_name}.npy")
    rollouts = torch.load(f"./imitation/imitation_expert/rollouts_{env_name}.npy")
    print("transitions",len(transitions))
    print("rollouts",len(rollouts))
    # PPO
    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0004,
        gamma=0.95,
        n_epochs=5,
        clip_range=0.1,
        vf_coef=0.1,
        seed=seed,
        verbose=0,
        device='cpu'
    )

    # reward net
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )

    # train
    if algorithm == "GAIL":
        GAIL_train()
    elif algorithm == "AIRL":
        AIRL_train()
    elif algorithm == "BC":
        BC_train()
    elif algorithm == "Dagger":
        Dagger_train()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    writer.close()
