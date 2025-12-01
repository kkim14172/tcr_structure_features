"""Aggregate Rosetta residue energy breakdown into CDR-level scores.

Input: residue_energy_breakdown.out (Rosetta application output)
Output: CSV with summed energies per (tcr_id, chain, region) where region ∈ {cdr1,cdr2,cdr3,framework}
"""
from __future__ import annotations
import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# CDR ranges in IMGT numbering
CDR_RANGES = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}

# Energy terms to aggregate
ENERGY_TERMS = [
    "fa_atr", "fa_rep", "fa_sol", "fa_intra_rep", "fa_intra_sol_xover4",
    "lk_ball_wtd", "fa_elec", "pro_close", "hbond_sr_bb", "hbond_lr_bb",
    "hbond_bb_sc", "hbond_sc", "dslf_fa13", "omega", "fa_dun", "p_aa_pp",
    "yhh_planarity", "ref", "rama_prepro", "total"
]

PAIR_TARGETS = {
    "cdr1a-cdr1b", "cdr1a-cdr2b", "cdr1a-cdr3b",
    "cdr2a-cdr1b", "cdr2a-cdr2b", "cdr2a-cdr3b",
    "cdr3a-cdr1b", "cdr3a-cdr2b", "cdr3a-cdr3b",
}

ANARCI_SCRIPT = Path(__file__).resolve().parents[2] / "structure_feature" / "sequence" / "alignment" / "run_anarci.sh"


def _to_one_letter(residue) -> Optional[str]:
    try:
        return seq1(residue.get_resname(), custom_map={"ASX": "B", "GLX": "Z"})
    except Exception:
        return None


def _chain_sequence(chain) -> Tuple[str, List[int]]:
    seq_chars, resseqs = [], []
    for res in chain:
        if res.get_id()[0] != " ":
            continue
        aa = _to_one_letter(res)
        if aa is None:
            continue
        seq_chars.append(aa)
        resseqs.append(res.get_id()[1])
    return "".join(seq_chars), resseqs


def _parse_anarci_numbering(stdout: str) -> List[Tuple[int, str, str]]:
    pat = re.compile(r"^\s*[A-Za-z]\s+(\d+)([A-Za-z]?)\s+([A-Z-])")
    numbering: List[Tuple[int, str, str]] = []
    for line in stdout.splitlines():
        m = pat.match(line)
        if not m:
            continue
        numbering.append((int(m.group(1)), m.group(2), m.group(3)))
    return numbering


def _run_anarci(seq: str, scheme: str = "a", cache: Dict[str, Optional[List[Tuple[int, str, str]]]] = None) -> Optional[List[Tuple[int, str, str]]]:
    cache = cache if cache is not None else {}
    if seq in cache:
        return cache[seq]
    if not seq or not ANARCI_SCRIPT.exists():
        cache[seq] = None
        return None
    cmd = ["bash", str(ANARCI_SCRIPT), "-i", seq, "-s", scheme]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
    except Exception:
        cache[seq] = None
        return None
    numbering = _parse_anarci_numbering(res.stdout)
    cache[seq] = numbering if numbering else None
    return cache[seq]


def _cdr_label(base_num: int) -> str:
    for name, (lo, hi) in CDR_RANGES.items():
        if lo <= base_num <= hi:
            return name
    return "framework"


def _build_chain_maps(chain, use_anarci: bool, cache: Dict[str, Optional[List[Tuple[int, str, str]]]]) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Return mapping from resseq->cdr_label and resseq->seq_index."""
    seq, resseqs = _chain_sequence(chain)
    numbering = _run_anarci(seq, cache=cache) if use_anarci else None
    resseq_to_idx = {r: i for i, r in enumerate(resseqs)}
    if numbering and len(numbering) == len(resseqs):
        idx_to_cdr = {i: _cdr_label(base) for i, (base, _, _) in enumerate(numbering)}
    else:
        # fallback: approximate by positional windows
        idx_to_cdr = {}
        for i, _ in enumerate(resseqs):
            # crude mapping assuming 1-based position ~ IMGT (may be off if numbering differs)
            base = i + 1
            idx_to_cdr[i] = _cdr_label(base)
    resseq_to_cdr = {resseq: idx_to_cdr.get(idx, "framework") for resseq, idx in resseq_to_idx.items()}
    return resseq_to_cdr, resseq_to_idx


def extract_chain(label: str) -> Tuple[Optional[int], Optional[str]]:
    m = re.match(r"(-?\d+)([A-Za-z])", label)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def stream_aggregate(reb_path: Path, pdb_dir: Path, use_anarci: bool, max_poses: Optional[int] = None) -> pd.DataFrame:
    """
    Stream through the residue energy breakdown file to avoid large memory usage.
    Aggregates one-body energy terms for CDR loops (cdr1a, cdr2a, cdr3a, cdr1b, cdr2b, cdr3b)
    and pairwise energy terms for predefined cross-chain CDR pairs.
    One-body entries are identified by restype2 == "--" (Rosetta residue-energy breakdown "onebody").
    """
    cache: Dict[str, Optional[List[Tuple[int, str, str]]]] = {}
    chain_maps: Dict[str, Dict[str, Dict]] = {}
    sums: Dict[Tuple[str, str, str], float] = defaultdict(float)

    def region_label(pose_id: str, chain_id: Optional[str], resnum: Optional[int]) -> Optional[str]:
        if pose_id not in chain_maps or chain_id is None or chain_id not in chain_maps[pose_id]:
            return None
        base = chain_maps[pose_id][chain_id]["cdr_map"].get(resnum, "framework")
        if base not in {"cdr1", "cdr2", "cdr3"}:
            return None
        suffix = "a" if chain_id.upper() == "A" else ("b" if chain_id.upper() == "B" else None)
        if suffix is None:
            return None
        return f"{base}{suffix}"

    with reb_path.open("r") as f:
        header_line = None
        energy_cols = ENERGY_TERMS
        for line in f:
            if not line.startswith("SCORE:"):
                continue
            parts = line.split()
            if header_line is None:
                header_line = parts[1:]
                continue
            if parts[1] == header_line[0]:
                continue
            rec = dict(zip(header_line, parts[1:]))
            pose_id = rec.get("pose_id")
            if pose_id is None:
                continue

            if max_poses is not None and len(chain_maps) >= max_poses and pose_id not in chain_maps:
                continue
            if pose_id not in chain_maps:
                pdb_path = pdb_dir / pose_id
                if not pdb_path.exists():
                    chain_maps[pose_id] = {}
                else:
                    structure = PDBParser(QUIET=True).get_structure("x", pdb_path)[0]
                    cmap: Dict[str, Dict[str, Dict]] = {}
                    for chain in structure:
                        resseq_to_cdr, _ = _build_chain_maps(chain, use_anarci=use_anarci, cache=cache)
                        cmap[chain.id] = {"cdr_map": resseq_to_cdr}
                    chain_maps[pose_id] = cmap

            restype2 = rec.get("restype2")
            pdbid2 = rec.get("pdbid2")
            is_onebody = (restype2 == "onebody") or (pdbid2 == "--")
            if not is_onebody:
                # pairwise energies
                resnum1, chain1 = extract_chain(rec.get("pdbid1", ""))
                resnum2, chain2 = extract_chain(rec.get("pdbid2", ""))
                region1 = region_label(pose_id, chain1, resnum1)
                region2 = region_label(pose_id, chain2, resnum2)
                if not region1 or not region2:
                    continue
                pair_key = f"{region1}-{region2}"
                if pair_key not in PAIR_TARGETS:
                    pair_key = f"{region2}-{region1}"
                if pair_key not in PAIR_TARGETS:
                    continue
                for term in energy_cols:
                    try:
                        val = float(rec.get(term, 0.0))
                    except Exception:
                        val = 0.0
                    sums[(pose_id, pair_key, term)] += val
            else:
                # one-body energies
                resnum, chain_id = extract_chain(rec.get("pdbid1", ""))
                region = region_label(pose_id, chain_id, resnum)
                if not region:
                    continue
                for term in energy_cols:
                    try:
                        val = float(rec.get(term, 0.0))
                    except Exception:
                        val = 0.0
                    sums[(pose_id, region, term)] += val

    if not sums:
        return pd.DataFrame()

    rows = []
    for (pose_id, region, term), val in sums.items():
        rows.append({"tcr_id": pose_id, "region": region, "term": term, "value": val})
    df_energy = pd.DataFrame(rows)
    df_energy["col"] = df_energy["region"] + "__" + df_energy["term"]
    pivot = df_energy.pivot_table(index="tcr_id", columns="col", values="value", fill_value=0).reset_index()
    return pivot


def main():
    parser = argparse.ArgumentParser(description="Aggregate residue energy breakdown to CDR regions.")
    parser.add_argument("--reb-file", type=Path, default=Path("outputs/maura_hnncc/features/residue_energy_breakdown.out"))
    parser.add_argument("--pdb-dir", type=Path, default=Path("data/maura_hnncc/tcrbuilder2_out"))
    parser.add_argument("--output", type=Path, default=Path("outputs/maura_hnncc/features/residue_energy_cdr.csv"))
    parser.add_argument("--use-anarci", action="store_true", help="Use ANARCI numbering (docker-based); otherwise use positional fallback.")
    parser.add_argument("--max-poses", type=int, default=None, help="Optional limit on number of pose_ids to process (for testing)")
    args = parser.parse_args()

    df_cdr = stream_aggregate(args.reb_file, args.pdb_dir, use_anarci=args.use_anarci, max_poses=args.max_poses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_cdr.to_csv(args.output, index=False)
    print(f"Saved CDR energy table to {args.output} ({len(df_cdr)} rows)")


if __name__ == "__main__":
    main()
