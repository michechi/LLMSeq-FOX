# Mechanism-ID analysis

Adversarial audit of Tricky / Parity benchmarks and matched baselines.

Layout:

```
analysis/mechanism_id/
├── README.md                     — this file (reproduction commands)
├── report.md                     — final memo (written after Steps 1–5 done)
├── scripts/
│   ├── common.py                 — shared utilities (data, rules, features)
│   ├── phase1_label_audit.py     — Q1: which rule matches stored labels
│   ├── phase2_baseline_ladder.py — Q2: feature families A..D on std eval
│   ├── phase3_matched_histogram.py — Q2: full-26 histogram-matched eval
│   ├── phase4_heldout_rule.py    — Q2: cross-rule generalisation
│   ├── phase5_parity.py          — Q3: parity decomposition
│   ├── phase6_oracle.py          — Q4: oracle AUC/F1 on retained dist
│   └── slurm/run_phase.sh        — single SLURM wrapper
├── results/                       — CSVs + SLURM logs
└── plots/                        — figures
```

## Environment

the HPC cluster. All compute **must** be submitted via SLURM — never
run on the login node. The arm64 apptainer container requires arm64
compute nodes, so jobs go to `--partition=accel` (H200 Grace-Hopper).
Billing weights on `accel` are 0 (free GPU), so requesting 1 GPU is fine
even for CPU-only work.

## Reproduction

From the repo root:

```bash
# Phase 1 — label audit (Q1)
PHASE=1 EXTRA="--rows 50000" \
  sbatch -J mechid_p1 --time=0-01:00:00 --mem=32GB --cpus-per-task=8 \
  analysis/mechanism_id/scripts/slurm/run_phase.sh

# Phase 2 — baseline ladder on standard eval (Q2)
PHASE=2 EXTRA="--rows_train 200000 --rows_eval 50000 \
  --families A1_count26 A2_count_key B_residue C1_lagpair \
  C1k_lagpair_key D1k_lagtrigram_key" \
  sbatch -J mechid_p2a --time=0-03:00:00 --mem=64GB --cpus-per-task=16 \
  analysis/mechanism_id/scripts/slurm/run_phase.sh

# Phase 2b — sparse position-aware lag-pair (memory-heavy)
PHASE=2 EXTRA="--rows_train 200000 --rows_eval 50000 \
  --families C2_lagpair_pos" \
  sbatch -J mechid_p2b --time=0-04:00:00 --mem=96GB --cpus-per-task=16 \
  analysis/mechanism_id/scripts/slurm/run_phase.sh

# Phase 3 — full-26 histogram-matched evaluation (Q2)
PHASE=3 EXTRA="--rows_train 200000 --rows_eval 50000" \
  sbatch -J mechid_p3 --time=0-06:00:00 --mem=96GB --cpus-per-task=16 \
  analysis/mechanism_id/scripts/slurm/run_phase.sh

# Phase 4 — held-out-rule generalisation (Q2)
PHASE=4 EXTRA="--n_rules 6 --n_per_rule 20000" \
  sbatch -J mechid_p4 --time=0-02:00:00 --mem=32GB --cpus-per-task=8 \
  analysis/mechanism_id/scripts/slurm/run_phase.sh

# Phase 5 — parity decomposition (Q3)
PHASE=5 EXTRA="--sweep_rows 30000" \
  sbatch -J mechid_p5 --time=0-02:00:00 --mem=48GB --cpus-per-task=8 \
  analysis/mechanism_id/scripts/slurm/run_phase.sh

# Phase 6 — oracle AUC/F1 audit (Q4)
PHASE=6 EXTRA="--rows_eval 50000 --rows_train 100000" \
  sbatch -J mechid_p6 --time=0-01:00:00 --mem=32GB --cpus-per-task=8 \
  analysis/mechanism_id/scripts/slurm/run_phase.sh

# Phase 1b — infer Tricky rule from LogReg weights (needs Phase 1 done)
PHASE=1b EXTRA="--rows 200000" \
  sbatch -J mechid_p1b --time=0-01:00:00 --mem=32GB --cpus-per-task=8 \
  analysis/mechanism_id/scripts/slurm/run_phase.sh

# After all phase CSVs exist:
PHASE=plots   sbatch -J mechid_plots --time=0-00:15:00 --mem=16GB --cpus-per-task=4 analysis/mechanism_id/scripts/slurm/run_phase.sh
PHASE=summary sbatch -J mechid_sum   --time=0-00:15:00 --mem=16GB --cpus-per-task=2 analysis/mechanism_id/scripts/slurm/run_phase.sh
```

Monitor with `squeue --me` or `tail -f results/slurm-<jobid>-*.out`.

Datasets:

| Task       | csv tag            | N_train | ρ (stored, int label) |
|------------|--------------------|---------|-----------------------|
| Tricky 1   | `6`                | 400,000 | 0.356                 |
| Tricky 2   | `9`                | 400,000 | 0.416                 |
| Parity     | `test_just_pair`   | 800,000 | 0.498                 |

## Key findings (see `report.md` §1)

1. **Parity** rule is exactly `sum(count(S)) % 2 == 0` with
   S = {W,D,Q,J,X,N}. 100 % label agreement. Equivalent to paper
   Def.4 since |S| = 6 is even.
2. **Tricky** uses S = {W,D,Q,J,X,U} (the paper’s S), NOT the N-based
   S from the parity generator. The two generators used different
   letter sets.
3. **Tricky Det (`_6`) and Tricky Rnd (`_9`) are DIFFERENT rules.**
   `_9` follows paper Def.1 (pair-at-λ with monotone κ) under
   π = 0.3 noise and matches the paper’s reported AUC\* = 0.67 and
   F1\* = 0.58 exactly. `_6` follows a weaker rule: "pair-at-λ, both
   in S, any κ". Phase 1b (`results/phase1b_rule_check.txt`) reads the
   LogReg weights directly and confirms κ is ignored on `_6`.
4. A **linear LogReg on 676-dim aggregated lag-pair counts** reaches
   AUC 0.997 on `_6` and AUC 0.671 on `_9` — saturating both tasks.
   No neural model in the paper beats this, yet the paper's
   published baseline is contiguous k-grams that don't know λ.
5. Held-out-rule (`results/phase4_heldout_rule.csv`): C1 lag-pair
   features give AUC 1.0 on-rule, AUC 0.5 off-rule. The task is
   per-rule linear weighting, nothing more.
6. Parity (`results/phase5_parity.csv`): once K is revealed as a
   20-bit membership vector, a 20→64→1 MLP reaches AUC 0.997. The
   bottleneck is hidden-subset ID, not parity computation.

Figures live in `plots/`: `fig_ladder_standard.png`,
`fig_heldout_rule.png`, `fig_parity_decomp.png`.
The aggregated summary CSV is at `results/summary_all_phases.csv`.
The full memo is at `report.md`.
