# samm-irl
Supplementary Material and Source Code
``` # Install

pip install -r requirements.txt

# Create perturbed expert dataset from a clean one (Optional) 
python perturb_dataset.py --in ppo_expert_halfcheetah.pkl \
                          --out ppo_expert_halfcheetah_eps0.2.pkl \
                          --radius 0.2

# Train AIRL from (clean or perturbed) demos
python train_airl.py --env-id HalfCheetah-v4 \
                     --expert-path ppo_expert_halfcheetah_eps0.2.pkl \
                     --save-dir outputs/airl_halfcheetah_eps02 \
                     --seed 21 --n-rounds 50 --steps-per-round 5000
# Train SAMM-IRL

python train_samm_irl.py \
  --env-id HalfCheetah-v4 \
  --eps-radius 0.2 \
  --iters 20 \
  --ppo-steps-per-iter 10000 \
  --save-dir outputs/samm_halfcheetah_eps02 \
  --seed 21


# Evaluate AIRL
python evaluate.py \
  --policy outputs/airl_halfcheetah_eps02/policy_final.zip \
  --env-id HalfCheetah-v4 \
  --eps 0.2 \
  --seed 21 \
  --episodes 10

# Evaluate SAMM-IRL
python evaluate.py \
  --policy outputs/samm_halfcheetah_eps02/policy_final.zip \
  --env-id HalfCheetah-v4 \
  --eps 0.2 \
  --seed 21 \
  --episodes 10
```
