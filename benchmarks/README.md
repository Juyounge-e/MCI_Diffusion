# Conditional generation benchmarks

All baselines use the same contract:

- training CSV: `src/data/national_all.csv`
- target: `lat`, `lon`
- model condition: `pdr_mean`, `N`
- q-bin output: `lat,lon,N,pdr`
- no road projection in benchmark models

Benchmark training and sampling do not use the diffusion `ratio_bank`. Every model receives only `(pdr_mean, N)` as its condition. If the simulator later requires R/Y/G/B ratios, attach them in a separate preprocessing step shared by all model outputs.

## Models

- `mlp`: deterministic `condition -> coordinate` sanity baseline. It is not one-to-many for a fixed condition.
- `cvae`: stochastic latent-variable model. Repeated conditions use independent Gaussian latent vectors.
- `cgan`: continuous-condition GAN. Repeated conditions use independent noise vectors.
- `mdn`: full-covariance bivariate Gaussian mixture. It directly models `p(lat, lon | pdr_mean, N)`.

## Train

All wrappers accept the same arguments. Examples:

```bash
python benchmarks/mlp/train_mlp.py --out outputs/benchmarks/mlp
python benchmarks/cvae/train_cvae.py --out outputs/benchmarks/cvae
python benchmarks/cgan/train_cgan.py --out outputs/benchmarks/cgan
python benchmarks/mdn/train_mdn.py --out outputs/benchmarks/mdn
```

For an equal optimization-step budget, add (for example) `--max_steps 200000`. A quick smoke run can use `--max_train 2048 --epochs 1`.

## Single/range sample

```bash
python benchmarks/cvae/sample_cvae.py \
  --ckpt outputs/benchmarks/cvae/best_model.pt \
  --out outputs/benchmarks/cvae/samples.csv \
  --sample_num 100 --N 30 --uniform 0.06 0.08
```

Use `--temperature 1.0` for the primary comparison. MLP ignores temperature.

## Common q-bin sampling

```bash
python benchmarks/qbin_sampling.py \
  --model cvae \
  --ckpt outputs/benchmarks/cvae/best_model.pt \
  --training_csv src/data/national_all.csv \
  --out_dir outputs/benchmarks/cvae/qbin_n30 \
  --N 30 --n_min 27 --n_max 33 --num_bins 10 --total_samples 300 --seed 0
```

The same command works with `--model mlp`, `cgan`, or `mdn`. Each q-bin samples PDR uniformly from that bin's observed `pdr_min` to `pdr_max`, matching `notebooks/q-bin_sampling.py`. A deterministic `seed + bin_index` makes the requested PDR values reproducible across models.

After simulation, pass `q*.csv` and the corresponding simulator/metadata files to `eval/compare_gen_simul_pdr.py`.
