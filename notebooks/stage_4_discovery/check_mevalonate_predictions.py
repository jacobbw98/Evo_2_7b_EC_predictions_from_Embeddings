#!/usr/bin/env python3
"""
Check predicted EC numbers for unannotated (UP) Yarrowia proteins
against the Mevalonate pathway enzymes.
"""

import os
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import RAW_DIR, PRED_DIR, clean_sequence

import pandas as pd
import numpy as np

MEVALONATE_CSV = RAW_DIR / "mevalonate_pathway_EC_numbers.csv"
GENES_CSV = RAW_DIR / "genes_with_promoters.csv"

PARQUET_PATHS = {
    "Wide MLP (Layer 18)": PRED_DIR / "yarrowia_evo2_layer18_embeddings_predicted.parquet",
    "Deep Hierarchical MLP (Multilayer)": PRED_DIR / "yarrowia_evo2_multilayer_embeddings_predicted.parquet"
}

def main():
    # 1. Load Mevalonate Pathway EC numbers
    print(f"Loading Mevalonate pathway info from: {MEVALONATE_CSV}")
    mevalonate_ecs = {}
    with open(MEVALONATE_CSV, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ec = row["EC Number"].strip()
            mevalonate_ecs[ec] = {
                "Step": row["Step"],
                "Enzyme": row["Enzyme"],
                "Genes": row["Common Gene Name(s)"],
                "Reaction": row["Reaction"]
            }
    print(f"  Loaded {len(mevalonate_ecs)} unique EC numbers from Mevalonate pathway:\n" + 
          "\n".join([f"    - {ec}: {info['Enzyme']}" for ec, info in mevalonate_ecs.items()]))

    # 2. Build Sequence -> Gene Name lookup from genes_with_promoters.csv
    print(f"\nBuilding Sequence -> Gene Name lookup from: {GENES_CSV}")
    seq_to_gene = {}
    with open(GENES_CSV, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_seq = clean_sequence(row["sequence"])
            seq_to_gene[clean_seq] = row["gene_name"].strip()
    print(f"  Loaded {len(seq_to_gene)} gene sequence mappings")

    # 3. Analyze each predicted parquet file
    for name, path in PARQUET_PATHS.items():
        print(f"\n{'='*80}")
        print(f"Analyzing Predictions from: {name}")
        print(f"Path: {path}")
        print(f"{'='*80}")
        
        if not os.path.exists(path):
            print(f"  [Skipped] File does not exist yet. Please run the prediction script first.")
            continue
            
        df = pd.read_parquet(path)
        print(f"  Loaded {len(df)} rows")
        
        # Filter for Uncategorized Proteins (UP)
        up_df = df[df["EC_Numbers"] == "UP"].copy()
        print(f"  Found {len(up_df)} UP (uncategorized) genes")
        
        matches = []
        for idx, row in up_df.iterrows():
            clean_seq = clean_sequence(row["Nucleotide_Sequence"])
            gene_name = seq_to_gene.get(clean_seq, f"Row_{idx} (Unknown Gene)")
            
            # Check all top-3 predictions at Level 4 (EC4)
            for rank in [1, 2, 3]:
                pred_col = f"Predicted_EC_L4_top{rank}"
                prob_col = f"Prob_L4_top{rank}"
                
                # Support older parquet column layout if needed
                if pred_col not in row and rank == 1:
                    pred_col = "Predicted_EC"
                    prob_col = "Prob_top1" if "Prob_top1" in row else None
                
                if pred_col not in row:
                    continue
                
                pred_ec = str(row[pred_col]).strip()
                prob = row[prob_col] if (prob_col and prob_col in row) else np.nan
                
                if pred_ec in mevalonate_ecs:
                    matches.append({
                        "Gene_Name": gene_name,
                        "Matched_EC": pred_ec,
                        "Rank": rank,
                        "Prob": prob,
                        "Enzyme": mevalonate_ecs[pred_ec]["Enzyme"],
                        "Step": mevalonate_ecs[pred_ec]["Step"],
                        "Reaction": mevalonate_ecs[pred_ec]["Reaction"],
                        "Common_Genes": mevalonate_ecs[pred_ec]["Genes"]
                    })
        
        if len(matches) == 0:
            print("\n  No mevalonate pathway EC numbers were predicted for UP genes.")
        else:
            print(f"\n  Found {len(matches)} predictions matching Mevalonate pathway enzymes:")
            match_df = pd.DataFrame(matches)
            # Sort by Matched EC and probability descending
            match_df = match_df.sort_values(by=["Matched_EC", "Rank", "Prob"], ascending=[True, True, False])
            
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            for ec, group in match_df.groupby("Matched_EC"):
                print(f"\n  Matched EC: {ec} ({mevalonate_ecs[ec]['Enzyme']})")
                print(f"  Step {mevalonate_ecs[ec]['Step']} | Reaction: {mevalonate_ecs[ec]['Reaction']}")
                print(f"  Related Reference Genes: {mevalonate_ecs[ec]['Genes']}")
                print("-" * 100)
                print(group[["Gene_Name", "Rank", "Prob"]].to_string(index=False))
                print("-" * 100)

if __name__ == "__main__":
    main()
