import argparse, numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Minimal wrappers (mirrors the training ones)
class ResetCompat(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs); return obs, info
class StepCompat(gym.Wrapper):
    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4:
            obs, r, done, info = res
            return obs, r, done, False, info
        return res
class FloatObs(gym.ObservationWrapper):
    def __init__(self, env): super().__init__(env)
    def observation(self, obs):
        import numpy as np
        return obs.astype(np.float32) if isinstance(obs, np.ndarray) else obs
class StatePerturbationWrapper(gym.ObservationWrapper):
    def __init__(self, env, radius=0.2): super().__init__(env); self.radius=radius; self._skip=True
    def reset(self, **kw): obs, info = self.env.reset(**kw); self._skip=True; return obs, info
    def observation(self, obs):
        import numpy as np
        if self._skip: self._skip=False; return obs
        return obs + np.random.uniform(-self.radius, self.radius, obs.shape)

def make_env(env_id: str, seed: int, eps: float):
    env = gym.make(env_id, disable_env_checker=True)
    env.reset(seed=seed); env.action_space.seed(seed); env.observation_space.seed(seed)
    env = StepCompat(env); env = ResetCompat(env); env = StatePerturbationWrapper(env, eps); env = FloatObs(env)
    return env

def evaluate(policy_path: str, env_id: str, seed: int, eps: float, episodes: int):
    venv = DummyVecEnv([lambda: make_env(env_id, seed, eps)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, training=False)
    model = PPO.load(policy_path, env=venv)
    scores = []
    for _ in range(episodes):
        obs = venv.reset(); done=False; total=0.0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = venv.step(act)
            done = bool(term[0] or trunc[0]); total += float(r[0])
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Unified evaluator for AIRL/SAMM-IRL PPO policies")
    ap.add_argument("--policy", required=True, help="Path to .zip (e.g., .../policy_final.zip)")
    ap.add_argument("--env-id", default="HalfCheetah-v4")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--eps", type=float, default=0.2, help="Observation perturbation radius")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    mean, std = evaluate(args.policy, args.env_id, args.seed, args.eps, args.episodes)
    print(f"Return over {args.episodes} eps @ eps={args.eps}: {mean:.2f} ± {std:.2f}")
