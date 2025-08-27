import argparse
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch as th
from tqdm import tqdm

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.algorithms.adversarial.airl import AIRL


# -------------------------
# Env utilities (Gym/Gymnasium compat + dtype)
# -------------------------
class ResetCompat(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

class StepCompat(gym.Wrapper):
    """Ensures step returns (obs, reward, terminated, truncated, info)."""
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = result
        return obs, reward, terminated, truncated, info

def to_float32(obs):
    if isinstance(obs, dict):
        return {k: to_float32(v) for k, v in obs.items()}
    if isinstance(obs, (list, tuple)):
        return type(obs)(to_float32(x) for x in obs)
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    return obs

class FloatObs(gym.ObservationWrapper):
    def observation(self, obs):
        return to_float32(obs)

def make_env(env_id: str, seed: int):
    env = gym.make(env_id, disable_env_checker=True)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = ResetCompat(env)
    env = StepCompat(env)
    env = FloatObs(env)
    return env


# -------------------------
# IO utilities
# -------------------------
def load_expert_pickle(path: str) -> List[Dict[str, Any]]:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def convert_to_imitation_trajectories(raw_trajs: List[Dict[str, Any]]) -> List[Trajectory]:
    """Converts dict-based dataset to imitation Trajectory objects.
    Expected keys per traj: 'obs' [T x obs_dim], 'next_obs' [T x obs_dim],
                            'acts' [T x act_dim], 'dones' [T]
    AIRL expects len(obs) == len(acts) + 1, so we pad final obs if needed.
    """
    out: List[Trajectory] = []
    for traj in raw_trajs:
        obs = np.asarray(traj["obs"])
        next_obs = np.asarray(traj["next_obs"])
        acts = np.asarray(traj["acts"])
        dones = np.asarray(traj["dones"])
        if obs.shape[0] == acts.shape[0]:
            obs = np.vstack([obs, next_obs[-1:]])
        out.append(
            Trajectory(
                obs=obs.astype(np.float32),
                acts=acts.astype(np.float32),
                terminal=bool(dones[-1]),
                infos=None,
            )
        )
    return out


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    env_id: str = "HalfCheetah-v4"
    expert_path: str = "ppo_expert_halfcheetah.pkl"
    save_dir: str = "outputs/airl_halfcheetah"
    seed: int = 21
    n_rounds: int = 50
    steps_per_round: int = 5000
    demo_batch_size: int = 1024
    disc_updates_per_round: int = 4
    normalize_obs: bool = True
    normalize_reward: bool = False


# -------------------------
# Main training
# -------------------------
def main(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Reproducibility
    th.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    # VecEnv
    venv = DummyVecEnv([lambda: make_env(cfg.env_id, cfg.seed)])
    venv = VecNormalize(venv, norm_obs=cfg.normalize_obs, norm_reward=cfg.normalize_reward)

    # Load expert data
    raw = load_expert_pickle(cfg.expert_path)
    demos = convert_to_imitation_trajectories(raw)

    # Reward net + generator
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_action=True,
        use_next_state=False,
    )
    gen = PPO("MlpPolicy", venv, verbose=1, seed=cfg.seed)

    # AIRL trainer
    trainer = AIRL(
        demonstrations=demos,
        demo_batch_size=cfg.demo_batch_size,
        venv=venv,
        gen_algo=gen,
        reward_net=reward_net,
        n_disc_updates_per_round=cfg.disc_updates_per_round,
        allow_variable_horizon=True,
    )

    # Training loop with checkpoints
    for r in tqdm(range(1, cfg.n_rounds + 1), desc="AIRL rounds"):
        # Collect & update
        trainer.train_gen(cfg.steps_per_round)
        trainer.train_disc()
        gen.learn(total_timesteps=cfg.steps_per_round, reset_num_timesteps=False)

        if r % 10 == 0 or r == cfg.n_rounds:
            gen.save(os.path.join(cfg.save_dir, f"policy_round_{r}.zip"))

    # Final save
    gen.save(os.path.join(cfg.save_dir, "policy_final.zip"))
    print(f"✅ Finished. Models saved to: {cfg.save_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default=TrainConfig.env_id)
    p.add_argument("--expert-path", type=str, default=TrainConfig.expert_path)
    p.add_argument("--save-dir", type=str, default=TrainConfig.save_dir)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--n-rounds", type=int, default=TrainConfig.n_rounds)
    p.add_argument("--steps-per-round", type=int, default=TrainConfig.steps_per_round)
    p.add_argument("--demo-batch-size", type=int, default=TrainConfig.demo_batch_size)
    p.add_argument("--disc-updates-per-round", type=int, default=TrainConfig.disc_updates_per_round)
    args = p.parse_args()

    cfg = TrainConfig(
        env_id=args.env_id,
        expert_path=args.expert_path,
        save_dir=args.save_dir,
        seed=args.seed,
        n_rounds=args.n_rounds,
        steps_per_round=args.steps_per_round,
        demo_batch_size=args.demo_batch_size,
        disc_updates_per_round=args.disc_updates_per_round,
    )
    main(cfg)
