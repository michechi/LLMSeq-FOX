# Ordered Compliance hole audit

This report is generated only from valid completed-run markers. Missing cells and configurations are left missing; no result is inferred from a checkpoint filename or partial prediction shard.

## Completion inventory

- Completed: 8
- Failed or incomplete: 0
- Missing: 188
- Merged hole prediction rows: 20800000
- Merged strict-pair prediction rows: 80000

The row-level inventory is in `configuration_inventory.csv`.

## Main per-seed results

| Model | pi | Seed | Observed AUC | Latent AUC | Position AUROC | Repair AUC | Top-1 valid | Fixed flip | Strict accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bert_lora | 0.000 | 9950 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.989 |
| bert_lora | 0.300 | 9950 | 0.650 | 0.935 | 0.859 | 0.961 | 0.965 | 0.032 | 0.501 |
| llama_lora | 0.000 | 9950 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| llama_lora | 0.300 | 9950 | 0.665 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.737 |

Seed-mean, seed-SD, and available-seed-count columns are retained alongside every main metric in `model_summary.csv`. Evaluation confidence intervals are retained from the clustered bootstrap whenever the evaluator emitted them.
The time-constrained pretrained-model sweep intentionally uses only pi=0.0 and pi=0.3 with model seed 9950; seed-SD and intermediate-noise trajectories are therefore unavailable for BERT-LoRA and Llama-LoRA.

## Shortcut and random controls

_No completed rows._

## Paper endpoint checks

| Model | pi | Seed | Observed AUC | Paper AUC | Absolute deviation |
| --- | --- | --- | --- | --- | --- |
| llama_lora | 0.000 | 9950 | 1.000 | 0.846 | 0.154 |
| llama_lora | 0.300 | 9950 | 0.665 | 0.587 | 0.078 |

Runs outside tolerance are flagged, not discarded.

## Paired adjacent-noise differences

_No completed rows._

Differences appear only when the same model seed is complete at both adjacent noise levels.

## Interpretation

Standard AUC measures ordinary held-out prediction only. Hole localization, valid-letter ranking, fixed-position stability, and strict matched-pair accuracy answer distinct evaluation-only questions. Strong strict-pair performance shows use of information beyond the controlled count, aggregated lag-pair, occupancy, run-length, unordered-chain, and global edge-count representations; it does not prove a transferable Ordered Compliance algorithm. The mechanism parameters remain fixed.

## Exact recorded commands

```bash
/cluster/software/EL9/easybuild/software/Python/3.12.3-GCCcore-13.3.0/bin/python3 -m src.oc_completion.ordered_report --results-root /fp/homes01/u01/ec-michelec/LLMSeq-FOX/results/oc_hole_audit --checkpoint-root /cluster/work/projects/ec12/michechi/checkpoints/oc_hole_audit --data-root /fp/homes01/u01/ec-michelec/LLMSeq-FOX/data/simulation/oc_hole_audit
```

```bash
/fp/homes01/u01/ec-michelec/LLMSeq-FOX/repro/src/oc_completion/ordered_data.py --data-root /fp/homes01/u01/ec-michelec/LLMSeq-FOX/data/simulation/oc_hole_audit --results-root /fp/homes01/u01/ec-michelec/LLMSeq-FOX/results/oc_hole_audit --data-seed 9550 --noise-seed 9650 --hole-seed 9750 --n-workers 8
```

```bash
/fp/homes01/u01/ec-michelec/LLMSeq-FOX/repro/src/oc_completion/strict_pairs.py --pair_seed 9700 --data_dir /fp/homes01/u01/ec-michelec/LLMSeq-FOX/data/simulation/oc_hole_audit --out_dir /fp/homes01/u01/ec-michelec/LLMSeq-FOX/results/oc_hole_audit
```

## Generated artifacts

Aggregate CSVs, merged prediction parquets, LaTeX tables, and the nine SVG figures are colocated with this report. Placeholder SVGs explicitly say when no completed rows are available.
