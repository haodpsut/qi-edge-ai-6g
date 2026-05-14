# qi-edge-ai-6g — Code

Companion code for paper05.

## Setup (RTX 4090 server, conda-only)

```bash
conda env create -f environment.yml
conda activate qi-edge-ai-6g
```

## Dataset

NF-BoT-IoT-v2 (Sarhan et al., NetFlow v2). Place CSV at:

```
data/raw/NF-BoT-IoT-v2.csv
```

(or set `--data-path` flag). The loader sub-samples to 1M rows for the
case-study benchmark; full set takes too long for 5 seeds × 5 models on
a magazine-paper budget.

## Models (all parameter-matched to ~3.6K)

| Tag         | File                       | Description                          |
|-------------|----------------------------|--------------------------------------|
| `mlp`       | `models/mlp.py`            | 3-layer MLP baseline                 |
| `cnn1d`     | `models/cnn1d.py`          | 2-block 1D-CNN baseline              |
| `kan`       | `models/efficient_kan.py`  | Efficient KAN (B-spline)             |
| `fkan`      | `models/fourier_kan.py`    | Fourier-basis KAN                    |
| `cmlp`      | `models/complex_mlp.py`    | Complex-valued MLP (unitary-init)    |

## Run

```bash
# Smoke test
python train.py --model mlp --seed 42 --epochs 2 --quick

# Full single run
python train.py --model kan --seed 42

# Full benchmark (5 models × 5 seeds)
bash run_all.sh
```

Output: `results/{model}_{seed}.json` with accuracy, F1, params, FLOPs,
inference latency.

## Push results back

```bash
git add results/
git commit -m "benchmark results from RTX 4090"
git push
```