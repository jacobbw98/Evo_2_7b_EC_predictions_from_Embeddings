# Evo2 EC-Number Prediction Codebase

Self-contained workflow: Evo2 (7B, bfloat16) protein language-model
embeddings → 4-level hierarchical EC-number prediction →
*Yarrowia lipolytica* consumption-route gene discovery, plus the
benchmark_2 / benchmark_4 model-training pipelines.

All scripts are standalone and import shared paths, loaders, and model
classes from the sibling `definitions.py`. Run any script from its own
folder (e.g. `cd notebooks/stage_2_benchmark2_training && python
train_deep_hierarchical_mlp.py`).

## Datasets

The benchmark datasets and raw sequences are not bundled in this
repository (they are too large). Download them from the project
Google Drive folder:
https://drive.google.com/drive/folders/1lDpdfMCbW5MSgWoo7ZeAlAUFWkpbegYs
and extract them into `data/`:

- `data/benchmark_2/` — EC-number train/valid sets (parquet + batch dirs)
- `data/benchmark_4/` — 22,280-train / 18,420-valid EC-number sets (32 Evo2 layers)
- `data/raw_sequences/` — *Y. lipolytica* candidate sequences (FASTA + annotated CSVs)

Model checkpoints (`models/`) need no download — the stage 2 and stage 5
training scripts produce them.

## Layout

```
Embeddings_Extraction/
├── data/                        (downloaded datasets + produced embeddings; see Datasets)
│   ├── benchmark_2/             EC-number train/valid sets (parquet + batch dirs)
│   ├── benchmark_4/             22,280-train / 18,420-valid EC-number sets (32 Evo2 layers)
│   ├── raw_sequences/           Y. lipolytica candidate sequences (FASTA + annotated CSVs)
│   └── predictions_and_results/ (produced parquets: CDS + Yarrowia predictions, blast outputs)
├── models/                      (trained checkpoints, produced by stages 2 and 5)
│   ├── benchmark_2/             flat MLP, hierarchical MLP, deep hierarchical MLP
│   └── benchmark_4/             XGBoost/MLP/RF per-layer + concat models + summaries
├── figures/                     (all produced plots, grouped by domain)
│   ├── benchmark_2/             b2 in-domain (benchmark_2 valid) accuracy/loss/error-bar plots
│   ├── benchmark_4/             per-layer and comparison plots (incl. confusion-matrix F1)
│   ├── confusion_matrixs/       deep-hierarchical confusion matrices (all 4 levels)
│   └── yarrowia/                Yarrowia CDS + cross-translation accuracy plots
└── notebooks/                   (stage folders; scripts + mirrored notebooks)
    ├── definitions.py           ← shared paths, loaders, model classes (imported by all)
    ├── stage_0_input_prep/      UniProt/BLAST validation of candidate genes
    ├── stage_1_extraction/      Evo2 embedding extraction (benchmark_2/4, Yarrowia, CDS, cross-trans)
    ├── stage_2_benchmark2_training/  train flat / hierarchical / deep hierarchical MLPs
    ├── stage_3_prediction_eval/ predict + evaluate (CDS, Yarrowia, cross-trans, error bars)
    ├── stage_4_discovery/       mevalonate-pathway candidate checking + plots
    ├── stage_5_benchmark4_training/  XGBoost/MLP/RF per-layer experiments + comparisons
    └── stage_6_plots/           all plots (permutations, cross-trans, F1/confusion, known-vs-predicted)
```

### Notebooks

`run_all_permutations.ipynb` (top of `notebooks/`) is the stage-5
orchestrator. Every other `.ipynb` is a direct mirror of its
same-named `.py` script — the code cells are identical (only the
prose/markdown cells differ), so editing either one is valid. Keep the
pair in sync when you change one.

## Run order

1. **Stage 0 — input prep** (CPU, internet for BLAST/UniProt):
   `blast_validate_candidates.py` validates the 104 candidate gene
   names in `candidates_for_blast.fasta` against UniProt + EBI BLAST
   (results bundled in `data/predictions_and_results/`).
2. **Stage 1 — embedding extraction** (GPU, ~hours each): extract Evo2
   embeddings for benchmark_2, benchmark_4, Yarrowia CDS, and
   cross-translated Yarrowia sequences (full genomic nucleotides with
   intron gaps back-translated from the UniProt protein sequences).
   On first run this stage extracts all embeddings (outputs are
   cached afterwards and skipped automatically — see below).
3. **Stage 2 — benchmark_2 training** (GPU): train the three models on
   the 1,160,000-row train / 1,048,500-row valid set. Checkpoints and
   summaries are cached and re-training is skipped automatically.
4. **Stage 3 — prediction & evaluation** (GPU): predict EC numbers for
   the Yarrowia CDS set (all three models) and the cross-translated
   Yarrowia sequences (flat/hierarchical L18 + deep hierarchical), with
   per-level accuracy/F1, top-k, and error-bar plots. This is the end
   of the main pipeline: the deep hierarchical MLP on Yarrowia is the
   final published result.
5. **Stage 4 — discovery** (CPU): check mevalonate-pathway candidates
   against the predictions and plot known-vs-predicted L1.
6. **Stage 5 — benchmark_4 training** (GPU/CPU): per-layer XGBoost,
   MLP (last/mean token), MLP layer-17, and Random Forest experiments
   on the 32-layer benchmark_4 sets, plus the 5-model concat
   comparison.
7. **Stage 6 — plots** (CPU): regenerate all figures from the model
   summaries.

All heavy scripts **skip automatically when their outputs already
exist** (models, summaries, embedding parquets) and exit with a
`[skip]` message. Re-run an explicit step with `--force`.

## Stage details

### Stage 0 — Input preparation (`stage_0_input_prep/`)

| Script | Role | Outputs |
|--------|------|---------|
| `blast_validate_candidates.py` | UniProt REST gene lookup + EBI BLAST (internet required) | `blast_validation_results.json`, `blast_validation_summary.csv` |
| `prepare_yarrowia_inputs.py` | build Yarrowia input FASTA from the bundled candidate set | — |

### Stage 1 — Embedding extraction (`stage_1_extraction/`)

All scripts: Evo2-7B (`Evo2/Evo2_7Bv2`) on CUDA, bf16, batched, with
checkpointing. Skips re-extraction when the output parquets exist.

| Script | Input | Output |
|--------|-------|--------|
| `extract_benchmark2_embeddings.py` | benchmark_2 FASTA (layer 18 + multilayer 13/17/18/20) | `Train/Valid_with_embeddings.parquet` + batch dirs |
| `extract_benchmark4_embeddings.py` | benchmark_4 FASTA (layer 18 + all 32 layers + mean-token) | `Train/Valid_with_embeddings.parquet`, `Train/Valid_all_layers_embeddings.parquet` |
| `extract_cds_multilayer_embeddings.py` | Yarrowia CDS FASTA (multilayer, introns removed) | `ec_cds_multilayer_embeddings.parquet` + batches |
| `extract_yarrowia_embeddings.py` | raw Yarrowia FASTA (layer 18 + multilayer) | `yarrowia_layer18_embeddings.parquet`, `yarrowia_multilayer_embeddings.parquet` |
| `extract_crosstrans_embeddings.py` | cross-translated sequences from `genes_with_promoters.csv` + `yarrowia_uniprot.csv` (real exon nucleotides; intron gaps back-translated from the protein AAs; layer 18 + multilayer) | `yarrowia_crosstrans_layer18_embeddings.parquet`, `yarrowia_crosstrans_multilayer_embeddings.parquet` |

### Stage 2 — benchmark_2 training (`stage_2_benchmark2_training/`)

| Script | Models | Key settings |
|--------|--------|--------------|
| `train_benchmark2_mlp.py` | flat `EC_MLP` (L18) + `HierarchicalEC_MLP` (L18) | 200 ep, AdamW 3e-4 + warmup 10 + cosine, FocalLoss γ=2.0 / per-level weights {0.5,1.0,1.5,3.0}, mixup 0.2, balanced weights, patience 30/25 |
| `train_deep_hierarchical_mlp.py` | `DeepHierarchicalEC_MLP` (L13/17/18/20, 16,384-dim) | 50 ep, AdamW 1e-3 + warmup 5 + cosine, per-level focal, mixup 0.2, torch.compile, 4640-class cosine head |

Checkpoints, loss-history `.npz`, and `results_summary.json` land in
`models/benchmark_2/{mlp_results, hierarchical_mlp_results,
deep_hierarchical_mlp_results}/`. Both scripts skip re-training when
their checkpoints + summaries exist.

### Stage 3 — prediction & evaluation (`stage_3_prediction_eval/`)

| Script | Predicts | Outputs |
|--------|----------|---------|
| `predict_yarrowia_ec.py` | Yarrowia raw seqs → flat MLP + hierarchical (L18) | `yarrowia_ec_predictions.parquet` |
| `predict_yarrowia_ec_deep.py` | Yarrowia raw seqs → deep hierarchical (16,384-dim) | `yarrowia_ec_deep_predictions.parquet`, per-level accuracy plots |
| `predict_cds_multilayer.py` | Yarrowia CDS set → all three models | `ec_cds_predictions.parquet` + per-model accuracy plots (→ `figures/yarrowia/cds_accuracy_by_level_*.png`) |
| `evaluate_crosstrans_layer18.py` | cross-trans seqs → flat + hierarchical (L18) | `yarrowia_crosstrans_layer18_predicted.parquet` + plot |
| `evaluate_crosstrans_multilayer.py` | cross-trans seqs → deep hierarchical | `yarrowia_crosstrans_multilayer_predicted.parquet` + plot |
| `evaluate_with_error_bars.py` | benchmark_2 valid re-evaluation, ±1 SD across 4 seeds | error-bar plot + metrics JSON (in-domain b2 figures: `figures/benchmark_2/accuracy_by_level_*.png`) |

### Stage 4 — discovery (`stage_4_discovery/`)

| Script | Role |
|--------|------|
| `check_mevalonate_predictions.py` | score mevalonate-pathway candidates with the L18 models |
| `check_mevalonate_predictions_multilayer.py` | same, with the deep hierarchical model |
| `find_consumption_route_genes.py` | filter candidates to consumption-route genes (plot) |
| `plot_known_vs_predicted_l1.py` | known-EC vs predicted L1 comparison (plot) |

### Stage 5 — benchmark_4 training (`stage_5_benchmark4_training/`)

| Script | Experiments | Notes |
|--------|-------------|-------|
| `train_benchmark4_permutations.py` | `xgb_block`, `xgb_mean`, `mlp_last`, `mlp_mean`, `mlp_layer17` (per-layer over 32 Evo2 layers) | per-layer models + `results_summary.json`; trained layers are skipped individually |
| `train_benchmark4_rf.py` | `rf_benchmark4` (9 RFs over L9/L24/L26 × block/mlp/post-norm) + `rf_combined` (12,288-dim concat) | CPU-only |
| `train_mlp_xgb_concat.py` | 5-model concat comparison (3 MLP + 2 XGB) | per-model checkpoints + `error_bar_metrics.json` |
| `evaluate_benchmark4_error_bars.py` | per-layer accuracy ± SD across all models | comparison plots + `summary_comparison_table.png` |

### Stage 6 — plots (`stage_6_plots/`)

| Script | Figures |
|--------|---------|
| `plot_all_permutations.py` | 4 per-layer bar charts + combined overlay + RF plot + summary table (→ `figures/benchmark_4/`, `figures/benchmark_2/`) |
| `plot_all_confusion_matrices.py` | deep-hierarchical confusion matrices, all 4 levels (→ `figures/confusion_matrixs/`) |
| `plot_crosstrans_accuracy.py` | cross-trans L18 + multilayer accuracy curves (→ `figures/yarrowia/`) |
| `plot_f1_and_confusion.py` | F1-by-layer + L1 confusion (→ `figures/benchmark_4/`, `figures/confusion_matrixs/`) |
| `plot_known_vs_predicted_l1.py` | known-vs-predicted L1 (→ `figures/yarrowia/`) |

## Summary tables

`data/summary_tables/` holds the four summary CSVs:

| File | Contents |
|------|----------|
| `table1_codon_table.csv` | Evo2 7B per-layer codon-usage statistics |
| `table2_model_training_summary.csv` | per-model training config, best epoch, losses, per-level accuracy + SD |
| `table3_embedding_extraction_times.csv` | wall-clock times for every extraction run |
| `table4_mevalonate_gois.csv` | mevalonate candidate genes: model predictions + UniProt/BLAST validation |

## Conventions

- **Device selection** — `get_device(require_cuda=True)` (in
  `definitions.py`) picks the GPU with the most free memory; pass
  `--cpu` / `--device cpu` for CPU.
- **Checkpointing** — extraction scripts checkpoint every 200 batches
  and resume from the last checkpoint; training scripts save the
  best-epoch checkpoint plus a JSON summary.
- **Memory** — `torch.set_num_threads` is clamped to the cgroup CPU
  quota (Slurm-aware); large embeddings are loaded as float32 batches
  and moved to GPU per-batch.
- **Plots** use the Agg backend (`matplotlib.use("Agg")`).
- **Labels** — 4-level hierarchical EC numbers; per-level macro/weighted
  F1 and accuracy are the standard metrics.
- **torch.compile** — the deep-hierarchical checkpoint was saved from a
  `torch.compile`d model, so state-dict keys carry an `_orig_mod.`
  prefix; loader scripts strip it before `load_state_dict`.
- **Internet** — only `blast_validate_candidates.py` needs internet
  (EBI BLAST REST + UniProt REST); every other script is fully local.
- **Cross-translation** — for the cross-translated Yarrowia arm, each
  gene's full genomic sequence (promoter, exons, UTR) keeps its real
  nucleotides; only the intron gaps — located by an amino-acid-guided
  codon walk over the UniProt protein — are replaced with an in-frame
  back-translation of the protein, with codons sampled by frequency
  from the gene's own exon codons. Output length equals the input
  length; reverse-strand genes are returned in genomic orientation.
- **In-domain vs transfer** — `figures/benchmark_2/accuracy_by_level_*.png` show
  in-domain benchmark_2 valid accuracy (Level-4 top-1 ≈ 60–61%);
  `figures/yarrowia/cds_accuracy_by_level_*.png` show cross-species transfer on
  the Yarrowia CDS set (Level-4 top-1 ≈ 45–47%). The transfer drop — largest at
  Level 4 with its 4,640 leaf classes — is expected, not a regression.

## Environment

- Conda env: `evo2` (Python 3.12, PyTorch 2.9+cu129, Evo2-7B, XGBoost, scikit-learn)
- GPU: single NVIDIA GPU with ≥80 GB VRAM (H200-class); `CUDA_VISIBLE_DEVICES` honored
- Evo2 checkpoint: `Evo2/Evo2_7Bv2` on HuggingFace (cached in `HF_HOME`)
