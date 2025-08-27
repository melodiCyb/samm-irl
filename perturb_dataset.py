import argparse
import pickle
import numpy as np
from typing import List, Dict, Any

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def perturb_array(x: np.ndarray, radius: float) -> np.ndarray:
    noise = np.random.uniform(low=-radius, high=radius, size=x.shape)
    return x + noise

def main(in_path: str, out_path: str, radius: float):
    data: List[Dict[str, Any]] = load_pickle(in_path)
    out: List[Dict[str, Any]] = []
    for traj in data:
        new_traj = {}
        for k, v in traj.items():
            if k in ("obs", "next_obs"):
                new_traj[k] = perturb_array(np.asarray(v), radius)
            else:
                new_traj[k] = v
        out.append(new_traj)
    save_pickle(out, out_path)
    print(f"✅ Saved perturbed trajectories to: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--radius", type=float, default=0.2)
    args = ap.parse_args()
    main(args.in_path, args.out_path, args.radius)
