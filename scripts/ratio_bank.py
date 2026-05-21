import pickle

with open("outputs/mlp_diffusion/test_rygb/ratio_bank.pkl", "rb") as f:
    ratio_bank = pickle.load(f)

for k, v in ratio_bank.items():
    print(f"N={k}, num_ratios={len(v)}")
    print(" first 3 ratios:", v[:3])
