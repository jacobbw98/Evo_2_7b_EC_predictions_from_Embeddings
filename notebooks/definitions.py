"""Shared definitions for the Evo2 EC-prediction pipeline.

Single home for everything the stage scripts used to duplicate:

  - project paths (PROJECT_ROOT, DATA_DIR, MODELS_DIR, FIGURES_DIR, RAW_DIR,
    BENCH2_DIR, BENCH4_DIR, PRED_DIR)
  - device selection (get_device) and Evo2 7B loading (load_evo2_7b,
    including the weights_only=False torch.load patch Evo2's loader needs)
  - DNA helpers (VALID_DNA, clean_sequence, manual_tokenize, rev_comp,
    CODON_TABLE, translate_cds, translate_dna_to_aa, AA_TO_CODON,
    translate_aa_to_dna)
  - EC helpers (extract_ec_numbers, extract_ec_string, ec_at_level)
  - embedding batch IO (load_embeddings_from_batches, load_b2_layer_dicts,
    merge_embeddings)
  - model classes (EC_MLP, EC_MLP_Compact, HierarchicalEC_MLP,
    DeepHierarchicalEC_MLP, CosineClassifier, ResBlock, FocalLoss)

Every stage script bootstraps with:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
    from definitions import PROJECT_ROOT, ...

Note: model classes here are the canonical checkpoint-compatible versions.
All stage-2/3/5 checkpoints load against these definitions (verified by the
pipeline run).
"""

import gc
import glob
import os
import pickle
import re
from pathlib import Path

# Must be set before the first torch import (reduces CUDA fragmentation).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def _cgroup_cpu_count():
    """Effective CPU count from the cgroup quota (Slurm jobs cap this well
    below the node core count; torch's default of node-wide threads then
    oversubscribes and slows every CPU-side op)."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read().strip())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1

torch.set_num_threads(min(torch.get_num_threads(), _cgroup_cpu_count()))

# ============================================================
# Project paths
# ============================================================
# definitions.py lives in notebooks/; the project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
FIGURES_DIR  = PROJECT_ROOT / "figures"
RAW_DIR      = DATA_DIR / "raw_sequences"
BENCH2_DIR   = DATA_DIR / "benchmark_2"
BENCH4_DIR   = DATA_DIR / "benchmark_4"
PRED_DIR     = DATA_DIR / "predictions_and_results"

# ============================================================
# Device & Evo2 loading
# ============================================================
def get_device(require_cuda=False):
    """Return the compute device and print what will be used.

    require_cuda=True asserts CUDA is available (training/extraction scripts).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda:
        assert device.type == "cuda", "CUDA not available — required"
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / 1024**3:.1f} GB", flush=True)
    return device


def load_evo2_7b():
    """Load the pretrained Evo2 7B model.

    Evo2's loader calls torch.load without weights_only, which PyTorch 2.6+
    rejects; patch torch.load temporarily so missing weights_only defaults to
    False, then restore the original.
    """
    original_load = torch.load

    def unsafe_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = unsafe_load
    try:
        from evo2 import Evo2
        return Evo2("evo2_7b")
    finally:
        torch.load = original_load


# ============================================================
# DNA helpers
# ============================================================
VALID_DNA = set("ACGT")


def clean_sequence(seq):
    """Keep only A/C/G/T characters (uppercased)."""
    return "".join(c for c in seq.upper() if c in VALID_DNA)


def manual_tokenize(text):
    """Convert nucleotide characters to token IDs (ordinal encoding)."""
    return [ord(c) for c in text]


def rev_comp(seq):
    """Return the reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))


# 64-codon genetic code table (single source for all translation helpers).
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


def translate_cds(cds):
    """Translate a nucleotide CDS to amino acids (stops at first stop codon).

    Unknown codons map to '?'.
    """
    protein = []
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i:i + 3]
        aa = CODON_TABLE.get(codon, '?')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


def translate_dna_to_aa(dna_seq):
    """Translate a DNA sequence to amino acids (full length, no early stop).

    Unknown codons map to 'X'; a trailing stop codon is dropped.
    """
    dna_seq = str(dna_seq).upper()
    aa_parts = []
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i + 3]
        aa_parts.append(CODON_TABLE.get(codon, "X"))
    if aa_parts and aa_parts[-1] == '*':
        aa_parts.pop()
    return "".join(aa_parts)


# GC-optimized codon choice per amino acid (for back-translation).
AA_TO_CODON = {
    'A': 'GCC', 'C': 'TGC', 'D': 'GAC', 'E': 'GAG', 'F': 'TTC',
    'G': 'GGC', 'H': 'CAC', 'I': 'ATC', 'K': 'AAG', 'L': 'CTG',
    'M': 'ATG', 'N': 'AAC', 'P': 'CCC', 'Q': 'CAG', 'R': 'CGA',
    'S': 'TCC', 'T': 'ACC', 'V': 'GTG', 'W': 'TGG', 'Y': 'TAC',
    '*': 'TAA'
}


def translate_aa_to_dna(aa_seq):
    """Translate amino acid sequence to GC-optimized nucleotide sequence."""
    dna_parts = []
    for aa in str(aa_seq).upper():
        dna_parts.append(AA_TO_CODON.get(aa, "GCT"))  # Default to alanine GCT if unknown
    return "".join(dna_parts)


# ============================================================
# EC helpers
# ============================================================
def extract_ec_numbers(protein_name) -> list:
    """Extract all EC numbers from a UniProt protein-name string.

    Matches patterns like (EC 1.2.3.4), (EC 2.3.1.-), etc.
    Returns a deduplicated list (order preserved); empty list when the
    input is NaN or has no EC numbers.
    """
    if pd.isna(protein_name):
        return []
    matches = re.findall(r'EC\s+([\d]+\.[\d\-]+\.[\d\-]+\.[\d\-]+)', protein_name)
    seen = set()
    unique = []
    for ec in matches:
        if ec not in seen:
            seen.add(ec)
            unique.append(ec)
    return unique


def extract_ec_string(protein_name) -> str:
    """Like extract_ec_numbers but returns ';'.join(ecs), or 'UP' if none."""
    ecs = extract_ec_numbers(protein_name)
    return ';'.join(ecs) if ecs else 'UP'

def ec_at_level(ec_str, level):
    """Truncate an EC number to a specific hierarchy level.

    e.g. ec_at_level('2.3.1.86', 2) -> '2.3'
    """
    parts = str(ec_str).split(".")
    return ".".join(parts[:level])

def gpu_batches(X, y, bs, shuffle=True):
    """Yield (xb, yb) batches from GPU-resident tensors via index slicing.

    A DataLoader over GPU tensors falls back to per-sample Python
    __getitem__ (~18 ms/step at bs=2048, measured); direct HBM index
    gathers are ~2 ms/step, so the trainers iterate this generator
    instead of a DataLoader.

    ``y`` may be a single tensor (returned as a tensor) or a list of
    tensors (returned as a list, indexed element-wise).
    """
    n = X.size(0)
    ys = y if isinstance(y, (list, tuple)) else [y]
    if shuffle:
        perm = torch.randperm(n, device=X.device)
        idxs = (perm[s:s + bs] for s in range(0, n, bs))
    else:
        idxs = (torch.arange(s, min(s + bs, n), device=X.device)
                for s in range(0, n, bs))
    for idx in idxs:
        yield X[idx], (ys[0][idx] if len(ys) == 1 else [t[idx] for t in ys])


def length_tercile_groups(lengths):
    """Assign each row to a Short/Medium/Long sequence-length group.

    Rows are split into 3 equal-sized groups by length (ascending):
    0 = Short, 1 = Medium, 2 = Long. Same convention as the error-bar
    evaluation scripts (tercile split of the evaluation set).
    """
    lengths = np.asarray(lengths)
    n = len(lengths)
    t1, t2 = n // 3, 2 * n // 3
    order = np.argsort(lengths, kind="stable")
    groups = np.empty(n, dtype=np.int32)
    groups[order[:t1]] = 0
    groups[order[t1:t2]] = 1
    groups[order[t2:]] = 2
    return groups

# ============================================================
# Embedding batch IO
# ============================================================
def load_embeddings_from_batches(batch_dir, layer_key):
    """Load all embeddings from batch pickle files.

    Batch files are {batch_N.pkl: {AC_key: {layer_key: np.ndarray}}}.
    Returns dict {AC_key: np.ndarray} for the requested layer_key.
    """
    emb_dict = {}
    batch_files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.pkl")))
    print(f"  Loading {len(batch_files)} batch files from {os.path.basename(batch_dir)}/...",
          flush=True)
    for bf in tqdm(batch_files, desc="Loading batches"):
        with open(bf, "rb") as f:
            batch = pickle.load(f)
        for key, layers in batch.items():
            if layer_key in layers:
                emb_dict[key] = layers[layer_key]
    print(f"  Loaded {len(emb_dict)} embeddings", flush=True)
    return emb_dict


def load_b2_layer_dicts(split_name):
    """Load per-layer benchmark-2 embedding dicts for a split.

    Layer 18 lives in train/valid_embedding_batches; layers 13/17/20 in
    train/valid_embedding_batches_layer_{L}.
    Returns (layer_dicts {layer: {AC: (4096,) float32}}, available_layers).
    """
    layer_dicts = {}

    # Layer 18 is in the original batch dir
    if split_name == "train":
        dir18 = BENCH2_DIR / "train_embedding_batches"
    else:
        dir18 = BENCH2_DIR / "valid_embedding_batches"

    print(f"\n  Loading layer 18 from {os.path.basename(dir18)}/...")
    layer_dicts[18] = load_embeddings_from_batches(dir18, "blocks.18")

    # Try loading other layers
    available_layers = [18]
    for layer_idx in [13, 17, 20]:
        if split_name == "train":
            layer_dir = BENCH2_DIR / f"train_embedding_batches_layer_{layer_idx}"
        else:
            layer_dir = BENCH2_DIR / f"valid_embedding_batches_layer_{layer_idx}"

        if os.path.exists(layer_dir) and len(glob.glob(os.path.join(layer_dir, "batch_*.pkl"))) > 0:
            layer_key = f"blocks.{layer_idx}"
            print(f"  Loading layer {layer_idx} from {os.path.basename(layer_dir)}/...")
            layer_dicts[layer_idx] = load_embeddings_from_batches(layer_dir, layer_key)
            available_layers.append(layer_idx)
        else:
            print(f"  Layer {layer_idx}: not found, skipping")

    available_layers.sort()
    print(f"  Available layers: {available_layers}")
    return layer_dicts, available_layers


# ============================================================
# Memory-safe parquet merge (used by benchmark extraction scripts)
# ============================================================
def merge_embeddings(input_path, output_path, batch_dir, col_layer_map,
                     base_path=None, dim=4096):
    """Append 4096-d embedding columns to a parquet base from batch pickles.

    Loads batch pkls once, streams into float32 arrays, and writes
    large_list<float32> columns via pyarrow. (The old Series.apply(.tolist())
    merge materialized ~1.16M Python lists of 4096 floats, ~150 GB of objects,
    and was OOM-killed by the ~200 GB job cgroup.)

    col_layer_map: list of (column_name, layer_key) pairs.

    Idempotency: on re-run/resume the base may already contain columns appended
    by an earlier (possibly killed) run; stale copies are dropped first so we
    never write duplicate field names (ArrowInvalid on read).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    base = str(base_path) if base_path else str(input_path)
    table = pq.read_table(base)
    stale = {c for c, _ in col_layer_map if c in table.schema.names}
    if stale:
        keep_idx = [i for i, nm in enumerate(table.schema.names)
                    if nm not in stale]
        table = table.select(keep_idx)
        print(f"  Dropped {len(stale)} stale column(s) from base for re-append")
    n = table.num_rows
    ac_np = table.column("AC").to_numpy(zero_copy_only=False)
    uniq = np.unique(ac_np)
    pos = {u: k for k, u in enumerate(uniq)}
    row_uniq = np.fromiter((pos[a] for a in ac_np), dtype=np.int64, count=n)
    print(f"\nMerging batches (base: {os.path.basename(base)}, {n} rows, "
          f"{len(uniq)} unique ACs)...")
    embeddings_dict = {}
    for bf in tqdm(sorted(glob.glob(os.path.join(str(batch_dir), "batch_*.pkl"))),
                   desc="Loading batches"):
        with open(bf, "rb") as f:
            embeddings_dict.update(pickle.load(f))
    print(f"Total embeddings: {len(embeddings_dict)}")
    for col_name, layer_key in col_layer_map:
        emb_u = np.zeros((len(uniq), dim), dtype=np.float32)
        pop_u = np.zeros(len(uniq), dtype=bool)
        for key, embs in embeddings_dict.items():
            k = pos.get(key)
            if k is None or layer_key not in embs:
                continue
            emb_u[k] = np.asarray(embs[layer_key], dtype=np.float32)
            pop_u[k] = True
        # null-safe column: valid rows hold the 4096-d vector; null rows
        # are zero-length ranges (required by the parquet writer) plus a
        # null mask so read-back yields None, not []
        mask = pop_u[row_uniq]
        valid_uniq = row_uniq[mask]
        flat = pa.array(emb_u[valid_uniq].ravel(order="C"), type=pa.float32())
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(np.where(mask, dim, 0))
        col = pa.LargeListArray.from_arrays(
            pa.array(offsets, type=pa.int64()), flat,
            mask=pa.array(~mask, type=pa.bool_()))
        table = table.append_column(col_name, col)
        print(f"  {col_name}: {int(mask.sum())}/{n} rows populated "
              f"({dim}-d float32, large_list)")
        del mask, valid_uniq, flat, offsets, col, emb_u, pop_u
        gc.collect()
    del embeddings_dict
    pq.write_table(table, str(output_path))
    print(f"Saved: {output_path} (shape: {table.shape})")
    del table, ac_np, uniq, pos, row_uniq
    gc.collect()


# ============================================================
# Model classes (canonical, checkpoint-compatible)
# ============================================================
class EC_MLP(nn.Module):
    """Wide two-hidden-layer MLP for EC number classification (benchmark 2).

    Architecture: 4096 → 2048 → 1024 → num_classes
    Wider layers give more capacity for 4,640 EC classes.
    BatchNorm + Dropout for training stability and regularization.
    """
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        return self.net(x)


class EC_MLP_Compact(nn.Module):
    """Compact 3-layer MLP used for the benchmark-4 all-layers permutations."""
    def __init__(self, dim, nclass, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 1024), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(512, nclass)
        )
    def forward(self, x):
        return self.net(x)


class HierarchicalEC_MLP(nn.Module):
    """
    Hierarchical MLP with 4 cascaded heads for 4-level EC classification.

    Shared backbone + 4 cascaded heads, where each head (after the first)
    receives backbone features concatenated with the previous level's logits
    (detached). Trained by train_benchmark2_mlp.py (hierarchical experiment);
    checkpoints load with this exact architecture.
    """
    def __init__(self, input_dim, n_ec1, n_ec2, n_ec3, n_ec4, dropout=0.3):
        super().__init__()
        # Shared backbone: 4096 → 2048 → 1024
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        backbone_dim = 1024

        # Head 1: EC level 1 (7 classes) — no cascade input
        self.head1 = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_ec1),
        )

        # Head 2: EC level 2 — receives backbone + EC1 logits
        self.head2 = nn.Sequential(
            nn.Linear(backbone_dim + n_ec1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_ec2),
        )

        # Head 3: EC level 3 — receives backbone + EC2 logits
        self.head3 = nn.Sequential(
            nn.Linear(backbone_dim + n_ec2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_ec3),
        )

        # Head 4: EC level 4 — receives backbone + EC3 logits (wider head)
        self.head4 = nn.Sequential(
            nn.Linear(backbone_dim + n_ec3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_ec4),
        )

    def forward(self, x):
        feat = self.backbone(x)
        l1 = self.head1(feat)
        l2 = self.head2(torch.cat([feat, l1.detach()], dim=1))
        l3 = self.head3(torch.cat([feat, l2.detach()], dim=1))
        l4 = self.head4(torch.cat([feat, l3.detach()], dim=1))
        return l1, l2, l3, l4


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)


class CosineClassifier(nn.Module):
    """Cosine similarity classifier: normalizes both weights and features."""
    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x_norm, w_norm)


class DeepHierarchicalEC_MLP(nn.Module):
    """
    3-layer backbone with residual + 4 cascaded heads.
    Backbone dims scale proportionally to input_dim (~2-4x compression per step).
    EC4 head uses cosine classifier for better many-class separation.
    """
    def __init__(self, input_dim, n_ec1, n_ec2, n_ec3, n_ec4, dropout=0.3):
        super().__init__()
        # Scale hidden dims proportionally to input
        # Single layer (4096): 2048 → 1024 → 512
        # Multi layer (16384): 4096 → 2048 → 1024
        h1 = max(2048, input_dim // 2)    # ~2x compression
        h2 = max(1024, h1 // 2)           # ~2x compression
        h3 = max(512, h2 // 2)            # ~2x compression

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResBlock(h1, dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResBlock(h2, dropout),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        bb_dim = h3

        self.head1 = nn.Sequential(
            nn.Linear(bb_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_ec1))

        self.head2 = nn.Sequential(
            nn.Linear(bb_dim + n_ec1, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_ec2))

        self.head3 = nn.Sequential(
            nn.Linear(bb_dim + n_ec2, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_ec3))

        # EC4 uses cosine classifier for better many-class separation
        h4_proj_dim = max(512, bb_dim)
        self.head4_proj = nn.Sequential(
            nn.Linear(bb_dim + n_ec3, h4_proj_dim), nn.BatchNorm1d(h4_proj_dim),
            nn.ReLU(), nn.Dropout(dropout))
        self.head4_cls = CosineClassifier(h4_proj_dim, n_ec4, scale=30.0)

    def forward(self, x):
        feat = self.backbone(x)
        l1 = self.head1(feat)
        l2 = self.head2(torch.cat([feat, l1.detach()], dim=1))
        l3 = self.head3(torch.cat([feat, l2.detach()], dim=1))
        h4_feat = self.head4_proj(torch.cat([feat, l3.detach()], dim=1))
        l4 = self.head4_cls(h4_feat)
        return l1, l2, l3, l4


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification. Down-weights easy examples."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
