import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_samm_irl import make_env, SAMMRewardWrapper, reward_fn  # if you prefer, copy reward logic here

def make_eval_env(env_id: str, seed: int, eps: float):
    return DummyVecEnv([lambda: make_env(env_id, seed, eps)])

def evaluate(policy_path: str, env_id: str, seed: int, eps: float, episodes: int = 10):
    venv = make_eval_env(env_id, seed, eps)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
    model = PPO.load(policy_path, env=venv)
    scores = []
    for _ in range(episodes):
        obs = venv.reset()
        done = False
        total = 0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = venv.step(act)
            done = bool(term[0] or trunc[0])
            total += float(r[0])
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--env-id", default="HalfCheetah-v4")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--eps", type=float, default=0.2)
    args = ap.parse_args()
    mean, std = evaluate(args.policy, args.env_id, args.seed, args.eps)
    print(f"Return (mean ± std over 10 eps): {mean:.2f} ± {std:.2f}")
