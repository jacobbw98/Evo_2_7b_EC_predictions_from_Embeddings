#!/usr/bin/env python3
"""
Stage 0 input preparation for the Evo2 EC-prediction pipeline.

Merges the two original input-prep scripts (same inputs, no shared model):

Part 1 (run_sequences_ec):
  Join nucleotide sequences from genes_with_promoters.csv with EC numbers
  from yarrowia_uniprot.csv, matching on gene locus name.

  Creates a CSV output with two columns:
    - Nucleotide_Sequence: the gene nucleotide sequence
    - EC_Numbers: semicolon-separated EC numbers

  Only entries that have both a nucleotide sequence AND at least one EC number
  are included in the output.

Part 2 (run_cds_with_ec):
  Extract coding sequences (CDS) with introns removed, paired with EC numbers.

  Cross-references:
    - genes_with_promoters.csv  (genomic nucleotide sequences, may contain introns)
    - yarrowia_uniprot.csv       (amino acid sequences + EC numbers from UniProt)

  The amino acid sequence is used to guide intron removal from the nucleotide sequence.
  The algorithm walks through the nucleotide sequence codon-by-codon, matching each codon
  to the expected amino acid. When a mismatch is found (intron boundary), it scans forward
  to find where the coding sequence resumes.

  Output: ec_cds_sequences.csv with columns:
    - gene_name: gene identifier
    - ec_number: EC number(s) from UniProt protein name, or "UP" (unknown protein)
    - cds_sequence: nucleotide coding sequence with introns removed
"""

import csv
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import (
    RAW_DIR,
    extract_ec_numbers,
    extract_ec_string,
    rev_comp,
    CODON_TABLE,
    translate_cds,
)

# ---------- Config ----------

GENES_CSV = RAW_DIR / "genes_with_promoters.csv"
UNIPROT_CSV = RAW_DIR / "yarrowia_uniprot.csv"
OUTPUT_CSV = RAW_DIR / "yarrowia_sequences_ec.csv"

# ── Standard codon table ──────────────────────────────────────────────────────
# (CODON_TABLE and rev_comp are now shared via definitions.py)

STOP_CODONS = {'TAA', 'TAG', 'TGA'}


def build_ec_lookup(uniprot_csv: str) -> dict[str, str]:
    """Build a dict mapping gene locus name -> semicolon-joined EC numbers.

    Checks both 'Gene Names (ordered locus)' and 'Gene Names (ORF)' columns.
    """
    lookup = {}
    with open(uniprot_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            protein_names = row.get("Protein names", "")
            ec_numbers = extract_ec_numbers(protein_names)
            if not ec_numbers:
                continue

            ec_str = ";".join(ec_numbers)

            # Try ordered locus name
            locus = row.get("Gene Names (ordered locus)", "").strip()
            if locus:
                lookup[locus] = ec_str

            # Also try ORF name (some entries use this instead)
            # ORF names may have underscores like YALI0_E32164g
            orf = row.get("Gene Names (ORF)", "").strip()
            if orf:
                lookup[orf] = ec_str
                # Also store without underscore for matching
                orf_no_underscore = orf.replace("_", "")
                lookup[orf_no_underscore] = ec_str

    return lookup


def _try_extract_cds(nt_seq, aa_seq, min_match=5):
    """
    Try to extract the CDS from a nucleotide sequence by matching codons
    to the expected amino acid sequence. Returns CDS string or None.
    
    When a codon mismatch occurs (intron boundary), scans forward to find
    where the coding sequence resumes, requiring `min_match` consecutive
    matching codons to avoid false positives.
    """
    for start in range(len(nt_seq) - 2):
        if nt_seq[start:start+3] != 'ATG':
            continue
        # First codon must be M
        if CODON_TABLE.get(nt_seq[start:start+3]) != aa_seq[0]:
            continue

        exon_parts = []
        nt_pos = start
        aa_pos = 0
        exon_start = start
        success = True

        while aa_pos < len(aa_seq):
            if nt_pos + 3 > len(nt_seq):
                success = False
                break

            codon = nt_seq[nt_pos:nt_pos+3]
            translated_aa = CODON_TABLE.get(codon, '?')

            if translated_aa == aa_seq[aa_pos]:
                # Codon matches — still in an exon
                aa_pos += 1
                nt_pos += 3
            else:
                # Mismatch — we've hit an intron boundary
                # Save the exon up to this point
                if nt_pos > exon_start:
                    exon_parts.append(nt_seq[exon_start:nt_pos])

                # Scan forward to find where coding resumes
                remaining_aa = len(aa_seq) - aa_pos
                required = min(min_match, remaining_aa)
                found = False

                for offset in range(1, len(nt_seq) - nt_pos):
                    test_pos = nt_pos + offset
                    if test_pos + required * 3 > len(nt_seq):
                        break

                    # Check if `required` consecutive codons match
                    match_count = 0
                    for k in range(required):
                        c = nt_seq[test_pos + k*3 : test_pos + k*3 + 3]
                        if CODON_TABLE.get(c, '?') == aa_seq[aa_pos + k]:
                            match_count += 1
                        else:
                            break

                    if match_count >= required:
                        nt_pos = test_pos
                        exon_start = test_pos
                        found = True
                        break

                if not found:
                    success = False
                    break

        if success and aa_pos == len(aa_seq):
            # Include stop codon if present
            stop = nt_seq[nt_pos:nt_pos+3]
            end_pos = nt_pos + 3 if stop in STOP_CODONS else nt_pos
            exon_parts.append(nt_seq[exon_start:end_pos])
            return ''.join(exon_parts)

    return None


def extract_cds(nt_seq, aa_seq):
    """
    Extract CDS from nucleotide sequence by matching to amino acid sequence.
    Tries both forward and reverse complement strands.
    Returns the CDS nucleotide string or None if extraction fails.
    """
    if not aa_seq or not nt_seq:
        return None

    # Skip if nucleotide is too short to encode the protein
    min_required = len(aa_seq) * 3
    if len(nt_seq) < min_required:
        return None

    # Try forward strand first, then reverse complement
    for seq in [nt_seq, rev_comp(nt_seq)]:
        result = _try_extract_cds(seq, aa_seq)
        if result is not None:
            return result

    # Retry with relaxed matching (3 consecutive codons instead of 5)
    # This handles genes with very short exons
    for seq in [nt_seq, rev_comp(nt_seq)]:
        result = _try_extract_cds(seq, aa_seq, min_match=3)
        if result is not None:
            return result

    return None


def run_sequences_ec():
    # Step 1: Build EC number lookup from UniProt CSV
    print("Building EC number lookup from yarrowia_uniprot.csv...")
    ec_lookup = build_ec_lookup(UNIPROT_CSV)
    print(f"  Found {len(ec_lookup)} gene-to-EC mappings")

    # Step 2: Read genes_with_promoters.csv and join with EC numbers
    print("Joining with nucleotide sequences from genes_with_promoters.csv...")
    total_genes = 0
    matched = 0
    no_ec = 0

    with open(GENES_CSV, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as fout:

        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["Nucleotide_Sequence", "EC_Numbers"])

        for row in reader:
            total_genes += 1
            gene_name = row.get("gene_name", "").strip()
            sequence = row.get("sequence", "").strip()

            if not sequence or not gene_name:
                continue

            # Try to find EC numbers for this gene
            ec_str = ec_lookup.get(gene_name)

            # If not found, try without underscore variant
            if ec_str is None:
                ec_str = ec_lookup.get(gene_name.replace("_", ""))

            if ec_str is None:
                ec_str = "UP"
                no_ec += 1
            else:
                matched += 1

            writer.writerow([sequence, ec_str])

    print(f"\nDone! Results written to: {OUTPUT_CSV}")
    print(f"  Total genes in FASTA/genes file: {total_genes}")
    print(f"  Matched with EC numbers:         {matched}")
    print(f"  No EC number found:              {no_ec}")


def run_cds_with_ec():
    # ── Load genes_with_promoters.csv ──────────────────────────────────────
    print("Loading genes_with_promoters.csv...")
    genes = OrderedDict()
    with open(RAW_DIR / "genes_with_promoters.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            gene_name = row[1]
            nt_seq = row[6].strip()
            genes[gene_name] = nt_seq
    print(f"  Loaded {len(genes)} genes")

    # ── Load yarrowia_uniprot.csv ──────────────────────────────────────────
    print("Loading yarrowia_uniprot.csv...")
    uniprot = {}  # gene_name -> {aa_seq, protein_name}
    with open(RAW_DIR / "yarrowia_uniprot.csv", 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            ordered_locus = row[1].strip()
            orf_name = row[2].strip().replace('_', '')  # normalize YALI0_A -> YALI0A
            protein_name = row[4].strip()
            aa_seq = row[6].strip()

            entry = {'aa_seq': aa_seq, 'protein_name': protein_name}

            # Index by both ordered locus and normalized ORF name
            if ordered_locus:
                uniprot[ordered_locus] = entry
            if orf_name:
                uniprot[orf_name] = entry
    print(f"  Loaded {len(uniprot)} UniProt entries (by gene name)")

    # ── Process each gene ──────────────────────────────────────────────────
    print("\nExtracting CDS and EC numbers...")
    results = []
    stats = {
        'total': 0,
        'has_uniprot': 0,
        'cds_extracted': 0,
        'cds_verified': 0,
        'cds_failed': 0,
        'no_uniprot': 0,
        'with_ec': 0,
        'without_ec': 0,
    }

    for gene_name, nt_seq in genes.items():
        stats['total'] += 1

        if gene_name not in uniprot:
            stats['no_uniprot'] += 1
            continue

        stats['has_uniprot'] += 1
        entry = uniprot[gene_name]
        aa_seq = entry['aa_seq']
        protein_name = entry['protein_name']

        # Extract EC number
        ec_number = extract_ec_string(protein_name)
        if ec_number != 'UP':
            stats['with_ec'] += 1
        else:
            stats['without_ec'] += 1

        # Extract CDS (introns removed)
        cds = extract_cds(nt_seq, aa_seq)

        if cds is None:
            stats['cds_failed'] += 1
            continue

        # Verify the CDS translates correctly
        verification = translate_cds(cds)
        if verification != aa_seq:
            stats['cds_failed'] += 1
            print(f"  WARNING: Verification failed for {gene_name} "
                  f"(translated {len(verification)} vs expected {len(aa_seq)} AAs)")
            continue

        stats['cds_extracted'] += 1
        stats['cds_verified'] += 1
        results.append({
            'gene_name': gene_name,
            'ec_number': ec_number,
            'cds_sequence': cds,
        })

    # ── Write output ───────────────────────────────────────────────────────
    output_file = RAW_DIR / "ec_cds_sequences.csv"
    print(f"\nWriting {len(results)} entries to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['gene_name', 'ec_number', 'cds_sequence'])
        writer.writeheader()
        writer.writerows(results)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total genes in promoter file:    {stats['total']}")
    print(f"  Matched to UniProt:              {stats['has_uniprot']}")
    print(f"  No UniProt match:                {stats['no_uniprot']}")
    print(f"  CDS successfully extracted:      {stats['cds_extracted']}")
    print(f"  CDS extraction failed:           {stats['cds_failed']}")
    print(f"  Genes with EC number:            {stats['with_ec']}")
    print(f"  Genes without EC (labeled UP):   {stats['without_ec']}")
    print(f"\n  Output written to: {output_file}")
    print("=" * 60)


def main():
    run_sequences_ec()
    run_cds_with_ec()


if __name__ == "__main__":
    main()
