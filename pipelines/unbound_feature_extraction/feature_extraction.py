"""Extract rich structural features from tcrbuilder2 PDBs for FEST classification.

Features span V-domain geometry, CDR loop geometry/composition, SASA, and global
shape descriptors. Everything is rotation/translation invariant and robust to
minor numbering differences by using simple CA-based calculations.
"""
from __future__ import annotations
import argparse
import subprocess
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import seq1
from Bio.PDB.SASA import ShrakeRupley
import yaml

# --- Constants and defaults -------------------------------------------------

DEFAULT_CDR_RANGES = {"cdr1": (25, 42), "cdr2": (58, 77), "cdr3": (107, 138)}

HYDROPHOBIC = set("AILMFWYV")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FYW")
POLAR = set("STNQCY")

# Local path to ANARCI runner (docker-based)
ANARCI_SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "run_anarci.sh"
CDR_BASE_RANGES = [(25, 42), (58, 77), (107, 138)]


# --- Geometry helpers -------------------------------------------------------

def _to_one_letter(residue) -> Optional[str]:
    try:
        return seq1(residue.get_resname(), custom_map={"ASX": "B", "GLX": "Z"})
    except Exception:
        return None


def _ca_coords(chain) -> np.ndarray:
    coords = [atom.coord for atom in chain.get_atoms() if atom.get_id() == "CA"]
    return np.array(coords) if coords else np.empty((0, 3))


def _radius_of_gyration(coords: np.ndarray) -> float:
    if coords.size == 0:
        return np.nan
    centroid = coords.mean(axis=0)
    diffs = coords - centroid
    return float(np.sqrt((diffs ** 2).sum(axis=1).mean()))


def _span(coords: np.ndarray) -> float:
    if coords.shape[0] < 2:
        return np.nan
    max_dist = 0.0
    for i in range(coords.shape[0]):
        deltas = coords[i + 1 :] - coords[i]
        if deltas.size == 0:
            continue
        dists = np.sqrt((deltas ** 2).sum(axis=1))
        max_dist = max(max_dist, float(dists.max()))
    return max_dist


def _min_inter_chain_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if coords_a.size == 0 or coords_b.size == 0:
        return np.nan
    min_dist = np.inf
    for a in coords_a:
        deltas = coords_b - a
        dists = np.sqrt((deltas ** 2).sum(axis=1))
        local_min = float(dists.min())
        if local_min < min_dist:
            min_dist = local_min
    return float(min_dist)


def _contact_fraction(coords_a: np.ndarray, coords_b: np.ndarray, cutoff: float = 8.0) -> float:
    if coords_a.size == 0 or coords_b.size == 0:
        return np.nan
    contacts = 0
    total = coords_a.shape[0] * coords_b.shape[0]
    for a in coords_a:
        deltas = coords_b - a
        dists = np.sqrt((deltas ** 2).sum(axis=1))
        contacts += int((dists < cutoff).sum())
    return contacts / total if total else np.nan


def _mean_bfactor(chain) -> float:
    bfs = [atom.get_bfactor() for atom in chain.get_atoms() if atom.get_id() == "CA"]
    return float(np.mean(bfs)) if bfs else np.nan


def _principal_axes(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return centroid and principal axes (3 unit vectors) via PCA."""
    if coords.shape[0] < 3:
        return np.full(3, np.nan), np.full((3, 3), np.nan)
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh  # principal directions
    return coords.mean(axis=0), axes


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
        return np.nan
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    cosang = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def _gyration_tensor(coords: np.ndarray) -> np.ndarray:
    if coords.shape[0] < 2:
        return np.full((3, 3), np.nan)
    centered = coords - coords.mean(axis=0)
    return centered.T @ centered / coords.shape[0]


def _asphericity(lams: np.ndarray) -> float:
    if np.any(np.isnan(lams)):
        return np.nan
    l1, l2, l3 = lams
    return float(l1 - 0.5 * (l2 + l3))


def _acylindricity(lams: np.ndarray) -> float:
    if np.any(np.isnan(lams)):
        return np.nan
    l1, l2, l3 = lams
    return float(l2 - l3)


def _eccentricity(lams: np.ndarray) -> float:
    if np.any(np.isnan(lams)) or lams[0] == 0:
        return np.nan
    return float(np.sqrt(1 - lams[-1] / lams[0]))


def _plane_normal(coords: np.ndarray) -> np.ndarray:
    _, axes = _principal_axes(coords)
    return axes[2] if not np.any(np.isnan(axes)) else np.full(3, np.nan)


def _distance_to_plane(coords: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    if coords.size == 0 or np.any(np.isnan(plane_point)) or np.any(np.isnan(plane_normal)):
        return np.full(coords.shape[0], np.nan)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    return ((coords - plane_point) @ plane_normal)


def _loop_residues(chain, res_range: Tuple[int, int]):
    return [res for res in chain if res.get_id()[1] >= res_range[0] and res.get_id()[1] <= res_range[1]]


def _loop_coords(loop_res):
    coords = [atom.coord for res in loop_res for atom in res if atom.get_id() == "CA"]
    return np.array(coords) if coords else np.empty((0, 3))


def _loop_sequence(loop_res):
    seq = [_to_one_letter(res) for res in loop_res]
    return [s for s in seq if s]


def _loop_orientation(loop_coords: np.ndarray) -> np.ndarray:
    if loop_coords.shape[0] < 2:
        return np.full(3, np.nan)
    vec = loop_coords[-1] - loop_coords[0]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.full(3, np.nan)


def _sasa_for_selection(residues: List) -> float:
    sasas = [getattr(residue, "sasa", np.nan) for residue in residues]
    sasas = [s for s in sasas if not np.isnan(s)]
    return float(np.mean(sasas)) if sasas else np.nan


def _is_cdr_base(base: int) -> bool:
    return any(start <= base <= end for start, end in CDR_BASE_RANGES)


def _sidechain_stretch(res_list: List) -> Tuple[float, float, float]:
    """
    Measure side-chain reach in the middle of a loop.
    Returns (max, mean, p95) of CA-to-tip distances across mid residues.
    """
    if not res_list:
        return np.nan, np.nan, np.nan
    # choose middle residue(s)
    if len(res_list) <= 3:
        idxs = list(range(len(res_list)))
    else:
        mid = len(res_list) // 2
        idxs = [mid - 1, mid, mid + 1] if len(res_list) % 2 == 1 else [mid - 2, mid - 1, mid, mid + 1]

    vals = []
    for idx in idxs:
        res = res_list[idx]
        if not res.has_id("CA"):
            continue
        ca = res["CA"].coord
        tips = [
            atom.coord
            for atom in res
            if atom.element != "H" and atom.get_id() not in ("N", "CA", "C", "O")
        ]
        if not tips:
            continue
        d = max(np.linalg.norm(t - ca) for t in tips)
        vals.append(d)
    if not vals:
        return np.nan, np.nan, np.nan
    return float(np.max(vals)), float(np.mean(vals)), float(np.percentile(vals, 95))


# --- ANARCI helpers ---------------------------------------------------------

def _chain_sequence_and_residues(chain) -> Tuple[str, List]:
    seq_chars, residues = [], []
    for res in chain:
        # Skip heteroatoms
        if res.get_id()[0] != " ":
            continue
        aa = _to_one_letter(res)
        if aa is None:
            continue
        seq_chars.append(aa)
        residues.append(res)
    return "".join(seq_chars), residues


def _parse_anarci_numbering(stdout: str) -> List[Tuple[int, str, str]]:
    """
    Return list of (base_position, insertion_code, aa) in sequence order.
    """
    pat = re.compile(r"^\s*[A-Za-z]\s+(\d+)([A-Za-z]?)\s+([A-Z-])")
    numbering: List[Tuple[int, str, str]] = []
    for line in stdout.splitlines():
        m = pat.match(line)
        if not m:
            continue
        numbering.append((int(m.group(1)), m.group(2), m.group(3)))
    return numbering


def _run_anarci(seq: str, scheme: str = "a") -> Optional[List[Tuple[int, str, str]]]:
    """
    Call the dockerized ANARCI CLI. Returns numbering list or None on failure.
    """
    if not seq:
        return None
    if not ANARCI_SCRIPT.exists():
        return None
    cmd = ["bash", str(ANARCI_SCRIPT), "-i", seq, "-s", scheme]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
    except Exception:
        return None
    numbering = _parse_anarci_numbering(res.stdout)
    if not numbering:
        return None
    return numbering


def _cdr_indices_from_numbering(numbering: List[Tuple[int, str, str]]) -> Dict[str, List[int]]:
    """
    Map CDR names to sequence indices (0-based) using IMGT-like definitions.
    """
    cdrs = {"cdr1": [], "cdr2": [], "cdr3": []}
    for idx, (base, _, _) in enumerate(numbering):
        if min(DEFAULT_CDR_RANGES["cdr1"]) <= base <= max(DEFAULT_CDR_RANGES["cdr1"]):
            cdrs["cdr1"].append(idx)
        if min(DEFAULT_CDR_RANGES["cdr2"]) <= base <= max(DEFAULT_CDR_RANGES["cdr2"]):
            cdrs["cdr2"].append(idx)
        if min(DEFAULT_CDR_RANGES["cdr3"]) <= base <= max(DEFAULT_CDR_RANGES["cdr3"]):
            cdrs["cdr3"].append(idx)
    return cdrs


def _loop_keys_from_numbering(numbering: Optional[List[Tuple[int, str, str]]]) -> Optional[Dict[str, List[str]]]:
    if not numbering:
        return None
    keys: Dict[str, List[str]] = {"cdr1": [], "cdr2": [], "cdr3": []}
    for base, ins, _ in numbering:
        key = f"{base}{ins}"
        if 27 <= base <= 38:
            keys["cdr1"].append(key)
        if 56 <= base <= 65:
            keys["cdr2"].append(key)
        if 105 <= base <= 117:
            keys["cdr3"].append(key)
    return keys


def _framework_keys_from_numbering(numbering: Optional[List[Tuple[int, str, str]]]) -> Optional[List[str]]:
    if not numbering:
        return None
    keys: List[str] = []
    for base, ins, _ in numbering:
        if _is_cdr_base(base):
            continue
        keys.append(f"{base}{ins}")
    return keys


def _anarci_numbering_and_residues(chain) -> Tuple[Optional[List[Tuple[int, str, str]]], List]:
    if chain is None:
        return None, []
    seq, residues = _chain_sequence_and_residues(chain)
    numbering = _run_anarci(seq)
    return numbering, residues


def _residue_map_by_numbering(residues: List, numbering: Optional[List[Tuple[int, str, str]]]) -> Dict[str, object]:
    if not numbering:
        return {}
    return {f"{base}{ins}": res for res, (base, ins, _) in zip(residues, numbering) if res.has_id("CA")}


def _apply_framework_superimposition(structure, sample_chain_info: Dict[str, Dict[str, object]], ref_info: Optional[Dict[str, object]]) -> float:
    """
    Superimpose framework residues (non-CDR) of alpha/beta onto a reference structure.
    Returns RMSD or NaN if not enough common atoms.
    """
    if not ref_info:
        return np.nan

    ref_atoms, mob_atoms = [], []
    for label in ("A", "B"):
        mob = sample_chain_info.get(label) or {}
        ref = ref_info["chains"].get(label) if "chains" in ref_info else None
        if not ref or not mob:
            continue
        fw_keys_mob = mob.get("framework_keys")
        fw_keys_ref = ref.get("framework_keys")
        if not fw_keys_mob or not fw_keys_ref:
            continue
        res_map_mob = mob.get("res_map", {})
        res_map_ref = ref.get("res_map", {})
        common_keys = [k for k in fw_keys_mob if (k in fw_keys_ref and k in res_map_mob and k in res_map_ref)]
        for k in common_keys:
            if res_map_ref[k].has_id("CA") and res_map_mob[k].has_id("CA"):
                ref_atoms.append(res_map_ref[k]["CA"])
                mob_atoms.append(res_map_mob[k]["CA"])

    if len(ref_atoms) < 3 or len(ref_atoms) != len(mob_atoms):
        return np.nan

    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    sup.apply(structure.get_atoms())
    return float(sup.rms)


# --- Feature calculators ----------------------------------------------------

def compute_struct_features(pdb_path: Path, ref_info: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("tcr", pdb_path)[0]
    chains = {chain.id: chain for chain in structure}

    chain_ids = list(chains)
    alpha = chains.get("A") or chains.get("a") or (chains.get(chain_ids[0]) if chain_ids else None)
    beta = chains.get("B") or chains.get("b") or (chains.get(chain_ids[1]) if len(chain_ids) > 1 else None)

    # ANARCI numbering, residue maps, and loop keys
    num_a, residues_a = _anarci_numbering_and_residues(alpha)
    num_b, residues_b = _anarci_numbering_and_residues(beta)
    res_map_a = _residue_map_by_numbering(residues_a, num_a)
    res_map_b = _residue_map_by_numbering(residues_b, num_b)
    loop_keys_a = _loop_keys_from_numbering(num_a)
    loop_keys_b = _loop_keys_from_numbering(num_b)
    fw_keys_a = _framework_keys_from_numbering(num_a)
    fw_keys_b = _framework_keys_from_numbering(num_b)

    sample_chain_info = {
        "A": {"chain": alpha, "framework_keys": fw_keys_a, "res_map": res_map_a},
        "B": {"chain": beta, "framework_keys": fw_keys_b, "res_map": res_map_b},
    }

    framework_rmsd = _apply_framework_superimposition(structure, sample_chain_info, ref_info)

    coords_a = _ca_coords(alpha) if alpha else np.empty((0, 3))
    coords_b = _ca_coords(beta) if beta else np.empty((0, 3))

    com_a, axes_a = _principal_axes(coords_a)
    com_b, axes_b = _principal_axes(coords_b)
    inter_axis_angle = _angle_between(axes_a[0], axes_b[0]) if axes_a.size and axes_b.size else np.nan
    twist_angle = _angle_between(axes_a[1], axes_b[1]) if axes_a.size and axes_b.size else np.nan

    # SASA precomputation
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    chainA_sasa = sum(res.sasa for res in alpha) if alpha else np.nan
    chainB_sasa = sum(res.sasa for res in beta) if beta else np.nan
    complex_sasa = sum(res.sasa for res in structure.get_residues())
    contact_area = (
        (chainA_sasa + chainB_sasa - complex_sasa) / 2.0
        if not np.isnan(chainA_sasa) and not np.isnan(chainB_sasa)
        else np.nan
    )

    feat = {
        # V-domain geometry
        "chainA_len": coords_a.shape[0],
        "chainB_len": coords_b.shape[0],
        "chainA_rg": _radius_of_gyration(coords_a),
        "chainB_rg": _radius_of_gyration(coords_b),
        "chainA_span": _span(coords_a),
        "chainB_span": _span(coords_b),
        "chainA_mean_b": _mean_bfactor(alpha) if alpha else np.nan,
        "chainB_mean_b": _mean_bfactor(beta) if beta else np.nan,
        "com_distance": float(np.linalg.norm(com_a - com_b)) if coords_a.size and coords_b.size else np.nan,
        "interdomain_axis_angle_deg": inter_axis_angle,
        "interdomain_twist_deg": twist_angle,
        "min_inter_ca_distance": _min_inter_chain_distance(coords_a, coords_b),
        "contact_fraction_lt8A": _contact_fraction(coords_a, coords_b, cutoff=8.0),
        "contact_fraction_lt5A": _contact_fraction(coords_a, coords_b, cutoff=5.0),
        "has_chainA": bool(alpha),
        "has_chainB": bool(beta),
        # SASA / global
        "chainA_sasa": chainA_sasa,
        "chainB_sasa": chainB_sasa,
        "complex_sasa": complex_sasa,
        "inter_chain_contact_area": contact_area,
        "framework_alignment_rmsd": framework_rmsd,
    }

    # Global shape descriptors (CA-based)
    all_ca = np.vstack([c for c in [coords_a, coords_b] if c.size]) if coords_a.size or coords_b.size else np.empty((0, 3))
    gyr_tensor = _gyration_tensor(all_ca)
    if not np.isnan(gyr_tensor).all():
        lams = np.sort(np.linalg.eigvalsh(gyr_tensor))[::-1]
        feat.update({
            "rg_all": float(np.sqrt(lams.sum())),
            "asphericity": _asphericity(lams),
            "acylindricity": _acylindricity(lams),
            "eccentricity": _eccentricity(lams),
        })
    if all_ca.size:
        mins = all_ca.min(axis=0)
        maxs = all_ca.max(axis=0)
        feat["bbox_volume"] = float(np.prod(maxs - mins))

    # CDR loop features and composition
    loops_a = _loop_residue_lists(alpha, "A", loop_keys_a, res_map_a)
    loops_b = _loop_residue_lists(beta, "B", loop_keys_b, res_map_b)
    feat.update(_loop_features(alpha, "A", com_a, sr, loops_a))
    feat.update(_loop_features(beta, "B", com_b, sr, loops_b))

    return feat


def _loop_features(chain, chain_label: str, domain_centroid: np.ndarray, sr: ShrakeRupley, loop_res_lists: Dict[str, List]) -> Dict[str, float]:
    if chain is None:
        return {}
    plane_normal = _plane_normal(_ca_coords(chain))
    out: Dict[str, float] = {}

    for loop_name, res_list in loop_res_lists.items():
        coords = _loop_coords(res_list)
        seq = _loop_sequence(res_list)
        centroid = coords.mean(axis=0) if coords.size else np.full(3, np.nan)
        orientation = _loop_orientation(coords)
        heights = _distance_to_plane(coords, domain_centroid, plane_normal)

        out.update({
            f"{chain_label}_{loop_name}_len": len(seq),
            f"{chain_label}_{loop_name}_rg": _radius_of_gyration(coords),
            f"{chain_label}_{loop_name}_span": _span(coords),
            f"{chain_label}_{loop_name}_centroid_x": centroid[0],
            f"{chain_label}_{loop_name}_centroid_y": centroid[1],
            f"{chain_label}_{loop_name}_centroid_z": centroid[2],
            f"{chain_label}_{loop_name}_height_mean": float(np.nanmean(heights)) if heights.size else np.nan,
            f"{chain_label}_{loop_name}_height_max": float(np.nanmax(np.abs(heights))) if heights.size else np.nan,
            f"{chain_label}_{loop_name}_orient_x": orientation[0],
            f"{chain_label}_{loop_name}_orient_y": orientation[1],
            f"{chain_label}_{loop_name}_orient_z": orientation[2],
            f"{chain_label}_{loop_name}_sasa_mean": _sasa_for_selection(res_list),
        })

        # side-chain reach in the middle of the loop
        sc_max, sc_mean, sc_p95 = _sidechain_stretch(res_list)
        out[f"{chain_label}_{loop_name}_sidechain_mid_max"] = sc_max
        out[f"{chain_label}_{loop_name}_sidechain_mid_mean"] = sc_mean
        out[f"{chain_label}_{loop_name}_sidechain_mid_p95"] = sc_p95

        out.update(_composition_features(seq, chain_label, loop_name))

    return out


def _composition_features(seq: List[str], chain_label: str, loop_name: str) -> Dict[str, float]:
    seq_upper = [aa.upper() for aa in seq]
    total = len(seq_upper) if seq_upper else 1  # avoid div0
    counts = {
        f"{chain_label}_{loop_name}_hydrophobic": sum(aa in HYDROPHOBIC for aa in seq_upper),
        f"{chain_label}_{loop_name}_positive": sum(aa in POSITIVE for aa in seq_upper),
        f"{chain_label}_{loop_name}_negative": sum(aa in NEGATIVE for aa in seq_upper),
        f"{chain_label}_{loop_name}_aromatic": sum(aa in AROMATIC for aa in seq_upper),
        f"{chain_label}_{loop_name}_polar": sum(aa in POLAR for aa in seq_upper),
    }
    freqs = {f"{chain_label}_{loop_name}_aa_{aa}": seq_upper.count(aa) / total for aa in list("ACDEFGHIKLMNPQRSTVWY")}
    counts.update(freqs)
    return counts


def _loop_residue_lists(chain, chain_label: str, loop_keys: Optional[Dict[str, List[str]]], res_map: Dict[str, object]) -> Dict[str, List]:
    """
    Build loop residue lists using ANARCI numbering when available; fall back to defaults.
    """
    loops: Dict[str, List] = {"cdr1": [], "cdr2": [], "cdr3": []}
    if chain is None:
        return loops
    if loop_keys:
        for name, keys in loop_keys.items():
            loops[name] = [res_map[k] for k in keys if k in res_map]
    else:
        ranges = DEFAULT_CDR_RANGES
        for name, res_range in ranges.items():
            loops[name] = _loop_residues(chain, res_range)
    return loops


# --- Metadata helpers -------------------------------------------------------

def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def load_metadata(csv_path: Path, encoding: str, id_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding=encoding)
    df[id_column] = df[id_column].astype(str)
    df = df.drop_duplicates(subset=id_column)
    return df


def prepare_reference(config: Dict) -> Optional[Dict[str, object]]:
    """
    Load reference structure and precompute framework maps for alignment.
    If config lacks 'reference_pdb', the first PDB in raw_pdb_dir is used.
    """
    pdb_dir = Path(config["raw_pdb_dir"])
    ref_path_cfg = config.get("reference_pdb")
    if ref_path_cfg:
        ref_path = Path(ref_path_cfg)
    else:
        pdbs = sorted(pdb_dir.glob("*.pdb"))
        if not pdbs:
            return None
        ref_path = pdbs[0]

    if not ref_path.exists():
        return None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ref", ref_path)[0]
    chains = {chain.id: chain for chain in structure}
    chain_ids = list(chains)
    alpha = chains.get("A") or chains.get("a") or (chains.get(chain_ids[0]) if chain_ids else None)
    beta = chains.get("B") or chains.get("b") or (chains.get(chain_ids[1]) if len(chain_ids) > 1 else None)

    def _prep_chain(chain):
        numbering, residues = _anarci_numbering_and_residues(chain)
        res_map = _residue_map_by_numbering(residues, numbering)
        fw_keys = _framework_keys_from_numbering(numbering)
        return {"chain": chain, "framework_keys": fw_keys, "res_map": res_map}

    info = {
        "structure": structure,
        "path": str(ref_path),
        "chains": {
            "A": _prep_chain(alpha),
            "B": _prep_chain(beta),
        },
    }
    return info


# --- Pipeline ---------------------------------------------------------------

def build_feature_table(config: Dict, limit: Optional[int] = None, ref_info: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    pdb_dir = Path(config["raw_pdb_dir"])
    metadata_csv = Path(config["metadata_csv"])
    id_col = config["id_column"]
    meta_cols = config.get("feature_columns", [])

    meta = load_metadata(metadata_csv, config.get("metadata_encoding", "utf-8"), id_col)
    meta[id_col] = meta[id_col].str.replace(".pdb", "", regex=False)

    records = []
    for i, pdb_path in enumerate(sorted(pdb_dir.glob("*.pdb"))):
        if limit is not None and i >= limit:
            break
        tcr_id = pdb_path.stem
        row: Dict[str, object] = {"tcr_id": tcr_id, "pdb_path": str(pdb_path)}
        try:
            row.update(compute_struct_features(pdb_path, ref_info=ref_info))
        except Exception as exc:  # keep going even if a structure fails
            row.update({"error": str(exc)})
        meta_row = meta.loc[meta[id_col] == tcr_id]
        if not meta_row.empty:
            for col in meta_cols:
                if col in meta_row.columns:
                    row[col] = meta_row.iloc[0][col]
        records.append(row)

    df = pd.DataFrame.from_records(records)
    return df


def save_outputs(df: pd.DataFrame, config: Dict) -> None:
    feature_path = Path(config["output_feature_table"])
    csv_path = Path(config["output_dataset_csv"])
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feature_path, index=False)
    df.to_csv(csv_path, index=False)


def main(args: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Extract structural features for maura_hnncc dataset")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"), help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print a preview instead of writing files")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of PDBs to process (for quick tests)")
    parsed = parser.parse_args(args)

    cfg = load_config(parsed.config)
    ref_info = prepare_reference(cfg)
    df = build_feature_table(cfg, limit=parsed.limit, ref_info=ref_info)
    if parsed.dry_run:
        print(df.head())
        print(f"Rows: {len(df)}")
        return
    save_outputs(df, cfg)
    print(f"Saved features to {cfg['output_feature_table']} and {cfg['output_dataset_csv']}")


if __name__ == "__main__":
    main()
