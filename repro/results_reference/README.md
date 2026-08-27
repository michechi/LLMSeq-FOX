# Reference results

This directory is intentionally empty in the public repo. After running

```bash
bash scripts/reproduce_main.sh         # full protocol (3+ days on 1 GPU)
# or
bash scripts/reproduce_main.sh --tiny  # smoke test (~30 min)
```

JSON files will land in `${RESULTS_DIR:-./results}/`. The schema each file follows is:

```json
{
  "args": { "number_to_use": "6", "model_name": "...", "seed": 9550, ... },
  "results": [
    { "model": "...", "fraction": 0.01, "train_samples": 4000,
      "test_auc": 0.81, "test_f1": 0.74, "test_precision": 0.78, "test_recall": 0.71, ... }
  ]
}
```

`src/analysis/plot_main_figure.py` discovers any JSON under `--results_dir` matching this schema.
