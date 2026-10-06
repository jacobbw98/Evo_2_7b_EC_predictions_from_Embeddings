#!/usr/bin/env python3
"""
Phase 1: In Silico Sequence Homology Validation
Fast approach: UniProt REST API lookup for all candidate gene names in Y. lipolytica,
then EBI BLAST only for high-priority unknowns.
"""

import os
import csv
import json
import time
import urllib.request
import urllib.parse

# ============================================================
# Project paths — shared via definitions.py
# (script lives at <root>/notebooks/<stage>/, one level under notebooks/)
# ============================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # notebooks/
from definitions import PRED_DIR, RAW_DIR

FASTA_FILE = RAW_DIR / "candidates_for_blast.fasta"
OUTPUT_JSON = PRED_DIR / "blast_validation_results.json"
OUTPUT_CSV = PRED_DIR / "blast_validation_summary.csv"

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
EBI_BLAST_URL = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"


def parse_fasta(filepath):
    sequences = []
    header = None
    seq_parts = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    sequences.append((header, ''.join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        sequences.append((header, ''.join(seq_parts)))
    return sequences


def search_uniprot(gene_name):
    """Search UniProt for a gene in Yarrowia lipolytica (taxid 284591)."""
    clean = gene_name.replace('YALI0', 'YALI0_')
    
    for q in [
        f'(gene:{gene_name}) AND (organism_id:284591)',
        f'(gene:{clean}) AND (organism_id:284591)',
        f'(gene:{gene_name}) AND (organism_name:"Yarrowia lipolytica")',
        f'(gene:{clean}) AND (organism_name:"Yarrowia lipolytica")',
    ]:
        params = urllib.parse.urlencode({
            'query': q,
            'format': 'json',
            'fields': 'accession,gene_names,protein_name,ec,organism_name,go_f,xref_interpro,reviewed',
            'size': '5',
        })
        url = f"{UNIPROT_SEARCH_URL}?{params}"
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            results = data.get('results', [])
            if results:
                return results
        except Exception:
            continue
    return []


def extract_uniprot_info(entry):
    """Extract key fields from a UniProt JSON entry."""
    acc = entry.get('primaryAccession', '')
    reviewed = entry.get('entryType', '') == 'UniProtKB reviewed (Swiss-Prot)'
    
    prot_name = ''
    ec_list = []
    prot_desc = entry.get('proteinDescription', {})
    if prot_desc:
        for name_type in ['recommendedName', 'submissionNames']:
            names = prot_desc.get(name_type, {})
            if isinstance(names, list):
                names = names[0] if names else {}
            if names:
                full_name = names.get('fullName', {})
                if isinstance(full_name, dict):
                    prot_name = full_name.get('value', '')
                else:
                    prot_name = str(full_name)
                for ecn in names.get('ecNumbers', []):
                    ec_list.append(ecn.get('value', ''))
                if prot_name:
                    break
    
    gene_names = []
    for gn in entry.get('genes', []):
        if gn.get('geneName'):
            gene_names.append(gn['geneName'].get('value', ''))
    
    # GO molecular function terms
    go_terms = []
    for xr in entry.get('uniProtKBCrossReferences', []):
        if xr.get('database') == 'GO':
            props = {p['key']: p['value'] for p in xr.get('properties', [])}
            if props.get('GoTerm', '').startswith('F:'):
                go_terms.append(props['GoTerm'][2:])
    
    # InterPro domains
    interpro = []
    for xr in entry.get('uniProtKBCrossReferences', []):
        if xr.get('database') == 'InterPro':
            props = {p['key']: p['value'] for p in xr.get('properties', [])}
            interpro.append(f"{xr.get('id','')}:{props.get('EntryName','')}")
    
    return {
        'accession': acc,
        'reviewed': reviewed,
        'protein_name': prot_name,
        'ec_numbers': ec_list,
        'gene_names': gene_names,
        'go_functions': go_terms[:5],
        'interpro': interpro[:5],
    }


def ebi_blast_submit(sequence):
    params = urllib.parse.urlencode({
        'email': 'anonymous@example.com',
        'program': 'blastp',
        'stype': 'protein',
        'database': 'uniprotkb_swissprot',
        'sequence': sequence,
        'alignments': '5',
        'scores': '5',
        'exp': '1e-3',
    }).encode('utf-8')
    req = urllib.request.Request(f"{EBI_BLAST_URL}/run", data=params)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode('utf-8').strip()


def ebi_blast_poll(job_id, max_wait=300):
    time.sleep(8)
    start = time.time()
    while time.time() - start < max_wait:
        try:
            url = f"{EBI_BLAST_URL}/status/{job_id}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.read().decode('utf-8').strip()
            if status == 'FINISHED':
                return True
            elif status in ('FAILURE', 'ERROR', 'NOT_FOUND'):
                return False
        except Exception:
            pass
        time.sleep(10)
    return False


def ebi_blast_get(job_id):
    url = f"{EBI_BLAST_URL}/result/{job_id}/out"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode('utf-8')


def parse_ebi_output(text):
    hits = []
    in_hits = False
    for line in text.split('\n'):
        if 'Sequences producing significant alignments' in line:
            in_hits = True
            continue
        if in_hits and line.strip() == '':
            if hits:
                break
            continue
        if in_hits and line.strip():
            parts = line.rsplit(None, 2)
            if len(parts) >= 3:
                try:
                    evalue = float(parts[-1])
                    score = float(parts[-2])
                    hits.append({
                        'description': parts[0],
                        'score': score,
                        'evalue': evalue,
                    })
                except (ValueError, IndexError):
                    pass
    return hits


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("Phase 1: In Silico Homology Validation")
    print("=" * 100)
    
    # Load sequences
    fasta_seqs = {}
    for header, seq in parse_fasta(FASTA_FILE):
        gene = header.split()[0]
        pred_ec, prob = "", 0.0
        for p in header.split()[1:]:
            if p.startswith("predicted_EC="):
                pred_ec = p.split("=")[1]
            elif p.startswith("prob="):
                prob = float(p.split("=")[1])
        fasta_seqs[gene] = {'seq': seq, 'predicted_ec': pred_ec, 'prob': prob}
    
    print(f"Loaded {len(fasta_seqs)} candidates\n")
    
    # ========================================================================
    # PASS 1: UniProt gene lookup (fast, ~1 second each)
    # ========================================================================
    print("--- PASS 1: UniProt Gene Lookup ---")
    all_results = {}
    count = 0
    for gene, info in fasta_seqs.items():
        count += 1
        result = {
            'gene': gene,
            'predicted_ec': info['predicted_ec'],
            'prob': info['prob'],
            'seq_len': len(info['seq']),
            'uniprot': None,
            'blast_hits': [],
            'needs_blast': True,
        }
        
        try:
            entries = search_uniprot(gene)
            if entries:
                up_info = extract_uniprot_info(entries[0])
                result['uniprot'] = up_info
                
                # If UniProt has a real protein name (not just YALI0_...), we have an annotation
                name = up_info['protein_name']
                has_real_name = name and not name.startswith('YALI0') and 'ncharacterized' not in name.lower()
                has_ec = bool(up_info['ec_numbers'])
                
                if has_ec or has_real_name:
                    result['needs_blast'] = False  # Already have annotation
                
                status_str = f"{'✓' if has_ec else '○'} {up_info['accession']} | {name[:55]}"
                if has_ec:
                    status_str += f" | EC={';'.join(up_info['ec_numbers'])}"
                print(f"  [{count:3d}/104] {gene:<16} pred={info['predicted_ec']:<12} {status_str}")
            else:
                print(f"  [{count:3d}/104] {gene:<16} pred={info['predicted_ec']:<12} ✗ Not found in UniProt")
        except Exception as e:
            print(f"  [{count:3d}/104] {gene:<16} pred={info['predicted_ec']:<12} ERROR: {e}")
        
        all_results[gene] = result
        time.sleep(0.3)  # Rate limit
    
    # ========================================================================
    # PASS 2: EBI BLAST for unknowns (only high-priority: prob>=0.05 and no annotation)
    # ========================================================================
    need_blast = [g for g, r in all_results.items() 
                  if r['needs_blast'] and r['prob'] >= 0.04]
    
    print(f"\n--- PASS 2: EBI BLAST for {len(need_blast)} unannotated candidates ---")
    
    for i, gene in enumerate(need_blast):
        info = fasta_seqs[gene]
        result = all_results[gene]
        
        print(f"  [{i+1}/{len(need_blast)}] {gene} (EC={info['predicted_ec']}, prob={info['prob']:.4f}) ...", end='', flush=True)
        
        try:
            job_id = ebi_blast_submit(info['seq'])
            if ebi_blast_poll(job_id):
                raw = ebi_blast_get(job_id)
                hits = parse_ebi_output(raw)
                result['blast_hits'] = hits
                if hits:
                    print(f" {hits[0]['description'][:50]} | E={hits[0]['evalue']:.1e}")
                else:
                    print(" No hits")
            else:
                print(" BLAST failed/timeout")
        except Exception as e:
            print(f" ERROR: {e}")
        
        time.sleep(2)
    
    # ========================================================================
    # Generate summary
    # ========================================================================
    print("\n" + "=" * 100)
    print("GENERATING FINAL REPORT")
    print("=" * 100)
    
    rows = []
    for gene, r in all_results.items():
        pred_ec = r['predicted_ec']
        prob = r['prob']
        
        up_acc = ''
        up_name = ''
        up_ec = ''
        up_go = ''
        up_interpro = ''
        if r['uniprot']:
            up = r['uniprot']
            up_acc = up['accession']
            up_name = up['protein_name']
            up_ec = '; '.join(up['ec_numbers']) if up['ec_numbers'] else ''
            up_go = '; '.join(up['go_functions'][:3]) if up['go_functions'] else ''
            up_interpro = '; '.join(up['interpro'][:3]) if up['interpro'] else ''
        
        blast_desc = ''
        blast_evalue = ''
        if r['blast_hits']:
            bh = r['blast_hits'][0]
            blast_desc = bh['description'][:80]
            blast_evalue = f"{bh['evalue']:.1e}"
        
        # Classify
        if up_ec:
            ec_match = any(pred_ec.startswith(e.split('.')[0]) for e in up_ec.split('; ') if e)
            exact_match = pred_ec in up_ec
            if exact_match:
                status = 'EC_EXACT_MATCH'
            elif ec_match:
                status = 'EC_CLASS_MATCH'
            else:
                status = 'EC_MISMATCH'
        elif up_name and not up_name.startswith('YALI0') and 'ncharacterized' not in up_name.lower():
            status = 'NAMED_NO_EC'
        elif r['blast_hits']:
            status = 'BLAST_HOMOLOG'
        elif up_name:
            status = 'UNCHARACTERIZED'
        else:
            status = 'NOT_IN_UNIPROT'
        
        rows.append({
            'Gene_Name': gene,
            'Predicted_EC': pred_ec,
            'Prob': f"{prob:.4f}",
            'UniProt_Acc': up_acc,
            'UniProt_Name': up_name,
            'UniProt_EC': up_ec,
            'GO_Function': up_go,
            'InterPro': up_interpro,
            'BLAST_Hit': blast_desc,
            'BLAST_EValue': blast_evalue,
            'Status': status
        })
    
    # Save JSON
    json_data = []
    for gene, r in all_results.items():
        json_entry = {k: v for k, v in r.items() if k != 'seq'}
        json_data.append(json_entry)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Gene_Name', 'Predicted_EC', 'Prob',
            'UniProt_Acc', 'UniProt_Name', 'UniProt_EC',
            'GO_Function', 'InterPro',
            'BLAST_Hit', 'BLAST_EValue', 'Status'
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSON}")
    
    # Summary
    counts = {}
    for r in rows:
        s = r['Status']
        counts[s] = counts.get(s, 0) + 1
    
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY ({len(rows)} candidates)")
    print(f"{'='*70}")
    for s in ['EC_EXACT_MATCH', 'EC_CLASS_MATCH', 'EC_MISMATCH', 'NAMED_NO_EC', 'BLAST_HOMOLOG', 'UNCHARACTERIZED', 'NOT_IN_UNIPROT']:
        print(f"  {s:<25}: {counts.get(s, 0)}")
    print(f"{'='*70}")
    
    # Detailed tables
    print(f"\n{'='*100}")
    print("DETAILED RESULTS BY CATEGORY")
    print(f"{'='*100}")
    
    print(f"\n✓ EC Exact Matches (Evo2 prediction confirmed by UniProt):")
    for r in rows:
        if r['Status'] == 'EC_EXACT_MATCH':
            print(f"  {r['Gene_Name']:<16} predicted={r['Predicted_EC']:<12} UniProt_EC={r['UniProt_EC']:<15} {r['UniProt_Name'][:50]}")
    
    print(f"\n≈ EC Class Matches (same enzyme class, different sub-number):")
    for r in rows:
        if r['Status'] == 'EC_CLASS_MATCH':
            print(f"  {r['Gene_Name']:<16} predicted={r['Predicted_EC']:<12} UniProt_EC={r['UniProt_EC']:<15} {r['UniProt_Name'][:50]}")
    
    print(f"\n✗ EC Mismatches (prediction disagrees with UniProt):")
    for r in rows:
        if r['Status'] == 'EC_MISMATCH':
            print(f"  {r['Gene_Name']:<16} predicted={r['Predicted_EC']:<12} UniProt_EC={r['UniProt_EC']:<15} {r['UniProt_Name'][:50]}")
    
    print(f"\n? Named but no EC in UniProt (potentially novel annotation):")
    for r in rows:
        if r['Status'] == 'NAMED_NO_EC':
            print(f"  {r['Gene_Name']:<16} predicted={r['Predicted_EC']:<12} prob={r['Prob']}  {r['UniProt_Name'][:55]}")
    
    print(f"\n★ Uncharacterized / No annotation (highest novelty):")
    for r in rows:
        if r['Status'] in ('UNCHARACTERIZED', 'NOT_IN_UNIPROT', 'BLAST_HOMOLOG'):
            blast = f" BLAST: {r['BLAST_Hit'][:40]}" if r['BLAST_Hit'] else ""
            print(f"  {r['Gene_Name']:<16} predicted={r['Predicted_EC']:<12} prob={r['Prob']}{blast}")


if __name__ == "__main__":
    main()
