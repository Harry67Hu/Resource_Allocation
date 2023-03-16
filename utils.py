import numpy as np
# from mape.multiagent.environment import MultiAgentEnv
# import mape.multiagent.scenarios as scenarios
from allocation_tasks.multiagent.environment import MultiAgentEnv
import allocation_tasks.multiagent.scenarios as scenarios
import gym_vecenv


def normalize_obs(obs, mean, std):
    if mean is not None:
        return np.divide((obs - mean), std)
    else:
        return obs


def make_env(env_id, seed, rank, num_agents):
    def _thunk():
        env = make_multiagent_env(env_id, num_agents)
        env.seed(seed + rank) # seed not implemented
        return env
    return _thunk


def make_multiagent_env(env_id, num_agents):
    scenario = scenarios.load(env_id+".py").Scenario(num_agents=num_agents)
    world = scenario.make_world()

    env = MultiAgentEnv(world=world, 
                        reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation,
                        info_callback=scenario.info if hasattr(scenario, 'info') else None,
                        discrete_action=True,
                        done_callback=scenario.done,
                        cam_range=600,
                        )
    return env


def make_parallel_envs(args):
    # make parallel environments
    envs = [make_env(args.env_name, args.seed, i, args.num_agents) for i in range(args.num_processes)]
    if args.num_processes > 1:
        envs = gym_vecenv.SubprocVecEnv(envs)
    else:
        envs = gym_vecenv.DummyVecEnv(envs)
    # 此处都没有做标准化/归一化处理！
    envs = gym_vecenv.MultiAgentVecNormalize(envs, ob=True, ret=False)
    return envs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
