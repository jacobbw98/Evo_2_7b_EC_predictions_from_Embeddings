#!/usr/bin/env python3
"""
Find unannotated intermediate consumption route genes in the Yarrowia lipolytica
mevalonate pathway by screening Deep Hierarchical MLP (multilayer) predictions,
then plot the known-vs-predicted EC class comparison from the same predictions.
"""

import os
import csv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import PRED_DIR, RAW_DIR, FIGURES_DIR, clean_sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GENES_CSV = RAW_DIR / "genes_with_promoters.csv"
PREDICTED_PARQUET = PRED_DIR / "yarrowia_evo2_multilayer_embeddings_predicted.parquet"
OUTPUT_CSV = PRED_DIR / "unannotated_mevalonate_consumption_candidates.csv"

# Target EC numbers consuming Acetyl-CoA, Acetoacetyl-CoA, HMG-CoA, or Mevalonate
TARGET_EC_MAP = {
    # --- Acetyl-CoA Consumers ---
    "2.3.1.9": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "Acetyl-CoA acetyltransferase (Thiolase)",
        "Reaction": "2 Acetyl-CoA -> Acetoacetyl-CoA + CoA",
        "Pathway Role": "Pathway Step 1 (thiolase / redundant copies check)"
    },
    "2.3.3.10": {
        "Intermediate": "Acetyl-CoA / Acetoacetyl-CoA",
        "Enzyme": "HMG-CoA synthase",
        "Reaction": "Acetoacetyl-CoA + Acetyl-CoA + H2O -> HMG-CoA + CoA",
        "Pathway Role": "Pathway Step 2 (synthase / redundant copies check)"
    },
    "2.3.3.1": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "Citrate synthase",
        "Reaction": "Acetyl-CoA + Oxaloacetate + H2O -> Citrate + CoA",
        "Pathway Role": "TCA Cycle Entry (competing route)"
    },
    "2.3.3.8": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "ATP-citrate synthase",
        "Reaction": "ADP + Pi + Citrate + CoA -> ATP + Acetyl-CoA + Oxaloacetate",
        "Pathway Role": "Cytosolic Acetyl-CoA generation/utilization"
    },
    "6.4.1.2": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "Acetyl-CoA carboxylase",
        "Reaction": "Acetyl-CoA + HCO3- + ATP -> Malonyl-CoA + ADP + Pi",
        "Pathway Role": "Fatty Acid Synthesis Entry (competing route)"
    },
    "2.3.1.85": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "Fatty acid synthase (malonyl transferase activity)",
        "Reaction": "Acetyl-CoA + ACP -> Acetyl-ACP + CoA",
        "Pathway Role": "Fatty Acid Synthesis (competing route)"
    },
    "2.3.1.86": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "Fatty acid synthase (palmitoyl transferase activity)",
        "Reaction": "Palmitoyl-CoA + ACP -> Palmitoyl-ACP + CoA",
        "Pathway Role": "Fatty Acid Synthesis (competing route)"
    },
    "2.3.1.12": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "Dihydrolipoyllysine-residue acetyltransferase",
        "Reaction": "Acetyl-CoA + dihydrolipoamide -> CoA + S-acetyldihydrolipoamide",
        "Pathway Role": "Pyruvate dehydrogenase complex"
    },
    "2.3.1.38": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "[Acyl-carrier-protein] S-acetyltransferase",
        "Reaction": "Acetyl-CoA + [acyl-carrier protein] -> CoA + acetyl-[acyl-carrier protein]",
        "Pathway Role": "Lipid metabolism"
    },
    "2.3.1.54": {
        "Intermediate": "Acetyl-CoA",
        "Enzyme": "Formate C-acetyltransferase",
        "Reaction": "Acetyl-CoA + Formate -> Pyruvate + CoA",
        "Pathway Role": "Pyruvate formate-lyase system"
    },
    
    # --- Acetoacetyl-CoA Consumers ---
    "3.1.2.11": {
        "Intermediate": "Acetoacetyl-CoA",
        "Enzyme": "Acetoacetyl-CoA hydrolase",
        "Reaction": "Acetoacetyl-CoA + H2O -> Acetoacetate + CoA",
        "Pathway Role": "Acetoacetyl-CoA Degradation (flux-wasting route)"
    },
    "1.1.1.35": {
        "Intermediate": "Acetoacetyl-CoA",
        "Enzyme": "3-hydroxyacyl-CoA dehydrogenase",
        "Reaction": "(S)-3-hydroxyacyl-CoA + NAD+ -> 3-oxoacyl-CoA + NADH + H+",
        "Pathway Role": "Beta-oxidation / degradation (competing route)"
    },
    "1.1.1.36": {
        "Intermediate": "Acetoacetyl-CoA",
        "Enzyme": "Acetoacetyl-CoA reductase",
        "Reaction": "(R)-3-hydroxybutanoyl-CoA + NADP+ -> Acetoacetyl-CoA + NADPH + H+",
        "Pathway Role": "PHA/PHB synthesis pathway (competing route)"
    },
    "2.8.3.9": {
        "Intermediate": "Acetoacetyl-CoA",
        "Enzyme": "Butyrate--acetoacetate CoA-transferase",
        "Reaction": "Acetoacetyl-CoA + Butanoate -> Acetoacetate + Butanoyl-CoA",
        "Pathway Role": "Butyrate metabolism (competing route)"
    },
    "2.8.3.15": {
        "Intermediate": "Acetoacetyl-CoA",
        "Enzyme": "Succinyl-CoA:acetoacetate CoA-transferase",
        "Reaction": "Acetoacetyl-CoA + Succinate -> Acetoacetate + Succinyl-CoA",
        "Pathway Role": "Ketolysis / degradation (competing route)"
    },
    
    # --- HMG-CoA Consumers ---
    "1.1.1.34": {
        "Intermediate": "HMG-CoA",
        "Enzyme": "HMG-CoA reductase (NADPH)",
        "Reaction": "Mevalonate + 2 NADP+ + CoA -> HMG-CoA + 2 NADPH + 2 H+",
        "Pathway Role": "Pathway Step 3 (NADPH HMGR / redundant copies check)"
    },
    "1.1.1.88": {
        "Intermediate": "HMG-CoA",
        "Enzyme": "HMG-CoA reductase (NADH)",
        "Reaction": "Mevalonate + 2 NAD+ + CoA -> HMG-CoA + 2 NADH + 2 H+",
        "Pathway Role": "Pathway Step 3 (NADH HMGR / redundant copies check)"
    },
    "4.1.3.4": {
        "Intermediate": "HMG-CoA",
        "Enzyme": "HMG-CoA lyase",
        "Reaction": "HMG-CoA -> Acetyl-CoA + Acetoacetate",
        "Pathway Role": "HMG-CoA Cleavage (competing degradation route)"
    },
    
    # --- Mevalonate Consumers ---
    "2.7.1.36": {
        "Intermediate": "Mevalonate",
        "Enzyme": "Mevalonate kinase",
        "Reaction": "ATP + Mevalonate -> ADP + Mevalonate 5-phosphate",
        "Pathway Role": "Pathway Step 4 (kinase / redundant copies check)"
    },
    "1.1.1.216": {
        "Intermediate": "Mevalonate",
        "Enzyme": "Mevalonate dehydrogenase (NADP+)",
        "Reaction": "Mevalonate + NADP+ -> Mevalonate-3-phosphate + NADPH + H+",
        "Pathway Role": "Mevalonate shunt / degradation (competing route)"
    },
    "1.1.1.294": {
        "Intermediate": "Mevalonate",
        "Enzyme": "Mevalonate 3-kinase / mevalonate 3-dehydrogenase",
        "Reaction": "Mevalonate + ATP -> Mevalonate 3-phosphate + ADP",
        "Pathway Role": "Mevalonate shunt / degradation (competing route)"
    }
}

PARQUET_PATH = PRED_DIR / 'yarrowia_evo2_multilayer_embeddings_predicted.parquet'
WORKSPACE_IMG_PATH = FIGURES_DIR / 'yarrowia' / 'ec_class_comparison.png'


def get_first_digit_known(ec_str):
    if not ec_str or pd.isna(ec_str) or ec_str == 'UP':
        return []
    digits = []
    parts = ec_str.replace(';', ',').split(',')
    for p in parts:
        p = p.strip()
        if p and p[0].isdigit():
            digits.append(p[0])
    return list(set(digits))


def run_discovery():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Scanning Yarrowia Predictions for Mevalonate Intermediate Consumption Route Genes ===")
    
    # 1. Check files
    if not os.path.exists(PREDICTED_PARQUET):
        raise FileNotFoundError(f"Predicted parquet file not found at: {PREDICTED_PARQUET}")
    if not os.path.exists(GENES_CSV):
        raise FileNotFoundError(f"Genes CSV file not found at: {GENES_CSV}")

    # 2. Load sequence to gene name lookup
    print(f"Loading gene names from: {GENES_CSV}")
    seq_to_gene = {}
    with open(GENES_CSV, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_seq = clean_sequence(row["sequence"])
            if clean_seq:
                seq_to_gene[clean_seq] = row["gene_name"].strip()
    print(f"  Loaded {len(seq_to_gene)} sequence-to-gene mappings.")

    # 3. Load Parquet predictions
    print(f"Loading predictions from: {PREDICTED_PARQUET}")
    df = pd.read_parquet(PREDICTED_PARQUET)
    print(f"  Loaded {len(df)} predictions.")

    # 4. Filter for Unannotated Proteins (UP)
    up_df = df[df["EC_Numbers"] == "UP"].copy()
    print(f"  Found {len(up_df)} unannotated (UP) genes.")

    # 5. Screen predictions for target EC numbers
    matches = []
    for idx, row in up_df.iterrows():
        clean_seq = clean_sequence(row["Nucleotide_Sequence"])
        gene_name = seq_to_gene.get(clean_seq, f"Row_{idx} (Unknown Gene)")
        
        # Check top-3 predictions
        for rank in [1, 2, 3]:
            pred_col = f"Predicted_EC_L4_top{rank}"
            prob_col = f"Prob_L4_top{rank}"
            
            if pred_col not in row or prob_col not in row:
                continue
                
            pred_ec = str(row[pred_col]).strip()
            prob = row[prob_col]
            
            # Check for exact match or prefix match in target map
            matched_ec_info = None
            if pred_ec in TARGET_EC_MAP:
                matched_ec_info = TARGET_EC_MAP[pred_ec]
            else:
                # Fallback check for prefix
                for key in TARGET_EC_MAP:
                    if pred_ec.startswith(key):
                        matched_ec_info = TARGET_EC_MAP[key]
                        break
            
            if matched_ec_info:
                matches.append({
                    "Gene_Name": gene_name,
                    "Predicted_EC": pred_ec,
                    "Rank": rank,
                    "Probability": prob,
                    "Intermediate": matched_ec_info["Intermediate"],
                    "Enzyme": matched_ec_info["Enzyme"],
                    "Reaction": matched_ec_info["Reaction"],
                    "Pathway_Role": matched_ec_info["Pathway Role"]
                })

    # 6. Save and print results
    if len(matches) == 0:
        print("\nNo candidate genes found matching target EC numbers.")
        return
        
    match_df = pd.DataFrame(matches)
    # Sort by intermediate, matched EC, rank, and probability
    match_df = match_df.sort_values(
        by=["Intermediate", "Predicted_EC", "Rank", "Probability"], 
        ascending=[True, True, True, False]
    )
    
    match_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFound {len(match_df)} prioritized candidates. Results saved to: {OUTPUT_CSV}")
    
    # Print candidates summary table
    print("\n" + "="*110)
    print(f"{'Gene Name':<15} | {'Intermediate':<25} | {'Predicted EC':<12} | {'Rank':<4} | {'Prob':<6} | {'Enzyme':<35}")
    print("="*110)
    for _, r in match_df.iterrows():
        print(f"{r['Gene_Name']:<15} | {r['Intermediate']:<25} | {r['Predicted_EC']:<12} | {r['Rank']:<4} | {r['Probability']:.4f} | {r['Enzyme']:<35}")
    print("="*110)


def run_ec_comparison_plot():
    print("Loading predictions parquet...")
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(f"Predictions parquet not found at: {PARQUET_PATH}")
        
    df = pd.read_parquet(PARQUET_PATH)
    
    # Process known ECs
    known_df = df[df['EC_Numbers'] != 'UP']
    known_digits = []
    for val in known_df['EC_Numbers']:
        known_digits.extend(get_first_digit_known(val))
    known_counts_series = pd.Series(known_digits).value_counts()
    
    # Process unannotated predictions
    unannotated_df = df[df['EC_Numbers'] == 'UP']
    unannotated_digits = []
    for val in unannotated_df['Predicted_EC_L4_top1']:
        val = str(val).strip()
        if val and val[0].isdigit():
            unannotated_digits.append(val[0])
    unannotated_counts_series = pd.Series(unannotated_digits).value_counts()
    
    # Setup classes 1 to 7
    classes = [str(i) for i in range(1, 8)]
    class_labels = [
        '1: Oxidoreductases',
        '2: Transferases',
        '3: Hydrolases',
        '4: Lyases',
        '5: Isomerases',
        '6: Ligases',
        '7: Translocases'
    ]
    
    known_counts = [known_counts_series.get(c, 0) for c in classes]
    unannotated_counts = [unannotated_counts_series.get(c, 0) for c in classes]
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Set modern style details
    ax.set_facecolor('#f8fafc') # Very light slate background
    fig.patch.set_facecolor('#ffffff')
    
    x = np.arange(len(classes))
    width = 0.35  # Bar width
    
    # Colors
    color_known = '#3b82f6'  # Modern blue
    color_pred = '#f97316'   # Modern orange
    
    rects1 = ax.bar(x - width/2, known_counts, width, label='Known EC Annotations', color=color_known, edgecolor='none', alpha=0.9)
    rects2 = ax.bar(x + width/2, unannotated_counts, width, label='Predicted ECs (Unannotated Genes)', color=color_pred, edgecolor='none', alpha=0.9)
    
    # Labels and Titles
    ax.set_ylabel('Gene Count (Amount)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Yarrowia lipolytica: Known vs. Predicted Functional EC Classes', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10, rotation=15, ha='right')
    
    # Legend
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0', fontsize=11, loc='upper right')
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e1')
    ax.set_axisbelow(True)
    
    # Clean spines
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#e2e8f0')
    
    # Add count labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color='#334155')
                            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    (FIGURES_DIR / "yarrowia").mkdir(parents=True, exist_ok=True)
    plt.savefig(WORKSPACE_IMG_PATH, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Plot saved successfully to: {WORKSPACE_IMG_PATH}")


def main():
    run_discovery()
    run_ec_comparison_plot()


if __name__ == "__main__":
    main()
