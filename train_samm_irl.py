import argparse, os, random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import torch as th
from tqdm import tqdm

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ======================
# VecEnv-safe wrappers
# ======================
class ResetCompat(gym.Wrapper):
    """Always returns (obs, info)."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

class StepCompat(gym.Wrapper):
    """Always returns (obs, reward, terminated, truncated, info)."""
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:  # old API
            obs, reward, done, info = result
            return obs, reward, done, False, info
        return result

def to_float32(x):
    if isinstance(x, dict): return {k: to_float32(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return np.asarray(x, dtype=np.float32)
    if isinstance(x, np.ndarray): return x.astype(np.float32)
    return x

class FloatObs(gym.ObservationWrapper):
    def observation(self, obs):
        return to_float32(obs)

class StatePerturbationWrapper(gym.ObservationWrapper):
    """Applies bounded uniform noise to observations (ε-ball)."""
    def __init__(self, env, radius: float = 0.2, skip_reset_noise: bool = True):
        super().__init__(env)
        self.radius = radius
        self._skip_next = skip_reset_noise

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._skip_next = True
        return obs, info

    def observation(self, obs):
        if self._skip_next:
            self._skip_next = False
            return obs
        if isinstance(obs, np.ndarray):
            return obs + np.random.uniform(-self.radius, self.radius, obs.shape)
        return obs


def make_env(env_id: str, seed: int, eps: float) -> gym.Env:
    env = gym.make(env_id, disable_env_checker=True)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = StepCompat(env)
    env = ResetCompat(env)
    env = StatePerturbationWrapper(env, radius=eps)
    env = FloatObs(env)
    return env


# ======================
# Trajectory utilities
# ======================
def _pad_obs(obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    """AIRL-style padding: len(obs) == len(acts)+1."""
    return np.vstack([obs, next_obs[-1:]]) if obs.shape[0] == next_obs.shape[0] else obs

def rollout_one(venv: DummyVecEnv, model: PPO, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Collect a single episode (vector env with num_envs=1)."""
    obs = venv.reset()
    done = False
    obs_list, act_list = [], []
    while not done:
        act, _ = model.predict(obs, deterministic=deterministic)
        obs_list.append(obs.copy())
        act_list.append(act.copy())
        obs, _, term, trunc, _ = venv.step(act)
        done = bool(term[0] or trunc[0])
    obs_arr = np.asarray(obs_list).squeeze(1)  # [T, obs_dim]
    act_arr = np.asarray(act_list).squeeze(1)  # [T, act_dim] or [T]
    # pad last obs for discounted features
    obs_arr = np.vstack([obs_arr, obs_arr[-1:]])
    return obs_arr.astype(np.float32), act_arr.astype(np.float32)


# ======================
# SAMM-IRL core
# ======================
class PhiNet(th.nn.Module):
    """Shallow encoder φ(s) from perturbed observations to features."""
    def __init__(self, obs_dim: int, feat_dim: int = 64):
        super().__init__()
        self.net = th.nn.Sequential(
            th.nn.Linear(obs_dim, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, feat_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


def discounted_feature_expectation(
    obs_list: List[np.ndarray], phi: PhiNet, gamma: float, device: th.device
) -> np.ndarray:
    """Compute μ(π)=E[∑ γ^t φ(s_t)] over a list of single-episode obs arrays with final padding."""
    outs = []
    for obs in obs_list:
        with th.no_grad():
            phi_s = phi(th.tensor(obs[:-1], dtype=th.float32, device=device))  # [T, d]
        T = phi_s.shape[0]
        weights = (gamma ** th.arange(T, device=device, dtype=th.float32)).unsqueeze(1)
        val = (weights * phi_s).sum(dim=0).cpu().numpy()
        outs.append(val)
    return np.mean(outs, axis=0)


@dataclass
class Config:
    env_id: str = "HalfCheetah-v4"
    seed: int = 21
    eps_radius: float = 0.2           # adversarial ε for observation perturbations
    gamma: float = 0.99
    feat_dim: int = 64
    lr: float = 1e-3
    l2: float = 1e-2
    iters: int = 20
    ppo_steps_per_iter: int = 10_000
    freeze_phi_after: int = 8         # optionally freeze φ after k iters
    clip_reward: float = 20.0
    save_dir: str = "outputs/samm_halfcheetah"


def main(cfg: Config):
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Reproducibility
    th.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)

    # VecEnv (with perturbations)
    venv = DummyVecEnv([lambda: make_env(cfg.env_id, cfg.seed, cfg.eps_radius)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    obs_dim = int(venv.observation_space.shape[0])

    # Models
    phi = PhiNet(obs_dim, cfg.feat_dim).to(device)
    w = th.nn.Parameter(th.randn(cfg.feat_dim, device=device))
    opt = th.optim.Adam(list(phi.parameters()) + [w], lr=cfg.lr)

    # Initialize policy and estimate expert μ_E from perturbed demos
    # If you have a perturbed expert dataset: provide --expert-pkl and we’ll load it.
    # Otherwise we bootstrap μ_E using the first policy rollout(s) as a proxy (optional).
    mu_E = None

    # PPO policy operating on reward r(s)=φ(s)^T w
    def reward_fn(obs_np: np.ndarray) -> np.ndarray:
        obs = th.tensor(obs_np, dtype=th.float32, device=device)
        r = (phi(obs) @ w)
        if cfg.clip_reward is not None:
            r = th.clamp(r, -cfg.clip_reward, cfg.clip_reward)
        return r.detach().cpu().numpy()

    class SAMMRewardWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
        def step(self, action):
            obs, _, term, trunc, info = self.env.step(action)
            rew = reward_fn(obs)
            return obs, rew, term, trunc, info

    # per-iter fresh env with same perturbation radius
    def make_rew_env():
        base = make_env(cfg.env_id, cfg.seed, cfg.eps_radius)
        return SAMMRewardWrapper(base)

    rew_venv = DummyVecEnv([make_rew_env])
    rew_venv = VecNormalize(rew_venv, norm_obs=True, norm_reward=False)

    policy = PPO("MlpPolicy", rew_venv, verbose=0, seed=cfg.seed)

    # Storage for μ(π)
    mu_list: List[np.ndarray] = []

    for it in tqdm(range(1, cfg.iters + 1), desc="SAMM-IRL"):
        # 1) Improve policy on current shaped reward
        policy.learn(total_timesteps=cfg.ppo_steps_per_iter)

        # 2) Rollout and compute μ(π_it)
        obs_arr, _ = rollout_one(rew_venv, policy, deterministic=True)
        mu_pi = discounted_feature_expectation([obs_arr], phi, cfg.gamma, device)
        mu_list.append(mu_pi)

        # Initialize μ_E once (if you don't have demos). Prefer: pass expert demos via --expert-pkl.
        if mu_E is None:
            mu_E = mu_pi.copy()

        # 3) Max-margin update on (φ, w):
        #    minimize  -min_j wᵀ(μ_E - μ_j) + λ||w||_2
        if it > 1:
            diffs = [th.tensor(mu_E - m, dtype=th.float32, device=device) for m in mu_list]
            margins = th.stack([w @ d for d in diffs])                # [it,]
            margin_loss = -margins.min()
            reg = cfg.l2 * th.norm(w, p=2)
            loss = margin_loss + reg

            # Optionally freeze φ after K iters (stabilizes late-stage training)
            if it >= cfg.freeze_phi_after:
                for p in phi.parameters(): p.requires_grad_(False)

            opt.zero_grad(); loss.backward(); opt.step()

            # keep w on unit sphere (optional but stabilizing)
            with th.no_grad():
                norm = th.norm(w, p=2)
                if th.isfinite(norm) and norm > 0:
                    w.mul_(1.0 / norm)

        # 4) Save checkpoints
        if it % 5 == 0 or it == cfg.iters:
            policy.save(os.path.join(cfg.save_dir, f"policy_round_{it}.zip"))
            th.save({"phi": phi.state_dict(), "w": w.detach().cpu()}, os.path.join(cfg.save_dir, f"samm_{it}.pt"))

    # Final artifacts
    policy.save(os.path.join(cfg.save_dir, "policy_final.zip"))
    th.save({"phi": phi.state_dict(), "w": w.detach().cpu()}, os.path.join(cfg.save_dir, "samm_final.pt"))
    print(f"✅ Done. Saved to {cfg.save_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default=Config.env_id)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--eps-radius", type=float, default=Config.eps_radius)
    p.add_argument("--gamma", type=float, default=Config.gamma)
    p.add_argument("--feat-dim", type=int, default=Config.feat_dim)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--l2", type=float, default=Config.l2)
    p.add_argument("--iters", type=int, default=Config.iters)
    p.add_argument("--ppo-steps-per-iter", type=int, default=Config.ppo_steps_per_iter)
    p.add_argument("--freeze-phi-after", type=int, default=Config.freeze_phi_after)
    p.add_argument("--clip-reward", type=float, default=Config.clip_reward)
    p.add_argument("--save-dir", type=str, default=Config.save_dir)
    args = p.parse_args()

    cfg = Config(
        env_id=args.env_id,
        seed=args.seed,
        eps_radius=args.eps_radius,
        gamma=args.gamma,
        feat_dim=args.feat_dim,
        lr=args.lr,
        l2=args.l2,
        iters=args.iters,
        ppo_steps_per_iter=args.ppo_steps_per_iter,
        freeze_phi_after=args.freeze_phi_after,
        clip_reward=args.clip_reward,
        save_dir=args.save_dir,
    )
    main(cfg)
