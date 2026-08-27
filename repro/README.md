# Sequential Learning or Shortcut Exploitation? Reproducibility code

Code and data accompanying the NeurIPS 2026 submission *Sequential Learning or Shortcut Exploitation? A Diagnostic Framework for Sequence Models*.

The paper introduces a synthetic letter-sequence framework with three task variants (Tricky Deterministic, Tricky Random, Parity), evaluates the model families in Table 2 of the paper (XGBoost, LSTM, Transformer encoder, BiLSTM-Transformer hybrid, BERT-base, RoBERTa-large, Llama-3.2-1B, Llama-3.1-8B, Llama-3.1-70B [parity only], Qwen3-4B-Think, Qwen2.5-14B), and audits whether sequence models recover the latent rule or solve the task via lag-aware pair counts.

The same shortcut-audit ladder is then applied to MIMIC-IV CKD->ESRD diagnosis-sequence prediction (paper Section 5 + Appendix F.x): cohort construction, k-gram diagnostic, order-invariant baselines (LogReg, XGBoost), and ordered-vs-shuffled training of Transformer / BiLSTM / BERT.

## Quickstart

```bash
git clone <this-repo>
cd <this-repo>
pip install -e .
cp .env.example .env       # set HF_TOKEN if you plan to fine-tune gated LLMs
bash scripts/reproduce_main.sh --tiny    # smoke run, < 30 min on 1 CPU
```

For the full reproduction protocol, see [REPRODUCING.md](REPRODUCING.md).

## Hardware expectations

| Component | Hardware | Wall-clock |
|---|---|---|
| XGBoost / LogReg baselines | 1 CPU, 16 GB RAM | < 30 min per dataset |
| LSTM / Transformer (~100K-400K params) | 1 GPU, 16 GB | ~1-2 h per dataset |
| BERT / RoBERTa fine-tune (LoRA) | 1 GPU, 24 GB | order of hours per dataset |
| Llama-3.1-8B fine-tune (LoRA, 4-bit) | 1 GPU H100 / A100 80GB | order of hours per dataset |
| Llama-3.1-70B fine-tune (Parity only, App. results) | 1 GPU H100 / A100 80GB (4-bit) | overnight |

Wall-clock figures are approximate; paper figures average 3 seeds (9550, 9551, 9552).

## Repo layout

```
repro/
├── data/simulation/tested/   # 15 CSV: train/val/test splits for the 3 tasks
├── data/mimic/               # MIMIC-IV CKD->ESRD audit (no raw data shipped)
├── src/
│   ├── common.py             # paths via env vars
│   ├── generators/           # data generation scripts (CLI)
│   ├── models/               # architecture definitions
│   ├── data/                 # data loaders
│   ├── experiments/          # 5 main training scripts + parity decomp + few-shot
│   ├── analysis/             # mechanism-ID audit, attention, plotting
│   ├── ablations/            # appendix experiments (A, B, C, F, G)
│   ├── mimic/                # MIMIC-IV CKD->ESRD audit (cohort, k-gram, shuffle, neural)
│   └── utils/
├── configs/                  # default hyperparameters
├── scripts/                  # bash launchers + SLURM templates
└── results_reference/        # pre-computed JSON results for sanity check
```

## Data

The three core datasets are stored as plain CSV in `data/simulation/tested/`:

- **Tricky Deterministic** (tag `_6`): N=400K train, lag=7, key set size 6, label noise pi=0.
- **Tricky Random** (tag `_9`): same as above with pi=0.3 label noise.
- **Parity** (tag `test_just_pair`): N=400K train, label = parity of key-letter count.

Total ~62 MB. To regenerate from seed:

```bash
bash scripts/regenerate_data.sh
```

The Naive sanity check dataset (tag `_alph`, App. A.2) is also included.

## Reproducing main results

See [REPRODUCING.md](REPRODUCING.md) for the full claim -> script -> command table.

## MIMIC-IV CKD->ESRD audit

Section 5 and Appendix F.x of the paper apply the same shortcut-audit ladder to MIMIC-IV. Raw MIMIC-IV is gated (PhysioNet credentialing) and not shipped. After dropping the three required CSVs (`diagnoses_icd.csv`, `admissions.csv`, `patients.csv`) into `data/mimic/raw/hosp/` you can run

```bash
bash scripts/reproduce_mimic.sh                    # cohort + descriptive + k-gram + shuffle + splits
bash scripts/slurm/submit_all_mimic.sh             # 12 ordered/shuffled neural jobs (HPC)
```

See [data/mimic/README.md](data/mimic/README.md) for the expected layout and the `MIMIC_*` env-var overrides.

## Known discrepancies between paper and code

- **Parity hidden subset.** The paper Section 4.1 describes the Parity hidden subset as `{W,D,Q,J,X,U}`, but the dataset shipped in `data/simulation/tested/X_*_test_just_pair.csv` and the generator code in `src/generators/test_simulation_det.py` use `{W,D,Q,J,X,N}`. The two choices have the same combinatorial structure (any size-6 subset of a 26-letter alphabet), so the empirical findings on hidden-subset parity are unchanged. The mechanism-ID audit in `src/analysis/mechanism_id/report.md` and the parity-decomposition results are computed with the `{N}` variant.

- **DL_baselines.py vs DL_TR_baselines.py.** The Transformer / RNNTransformer / LSTM rows of Table 2 are produced by `DL_TR_baselines.py`. `DL_baselines.py` is an earlier script restricted to LSTM/BiLSTM/GRU and accepts `--number_to_use` only as `int` (so `test_just_pair` is not supported there). For the paper's main results use `DL_TR_baselines.py`.

## Citation

```bibtex
@inproceedings{anonymous2026sequential,
  title={Sequential Learning or Shortcut Exploitation? A Diagnostic Framework for Sequence Models},
  author={Anonymous},
  booktitle={Submitted to NeurIPS 2026},
  year={2026}
}
```

## License

MIT (see [LICENSE](LICENSE)).
