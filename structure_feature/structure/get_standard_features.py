import os, sys
from glob import glob
import json
import subprocess
import re
from typing import Tuple, Dict, List
from Bio.PDB import PDBParser, PDBIO
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tcr_com import tcr_mhci_geometrical_parameters
from cuboid_alignment import (
    get_framework_coords,
    compute_cuboid,
    superimpose_structures,
    interdomain_angle,
)

def get_cdr_conf(pdb_path):
    def avg(ag):
        return float(np.nanmean(ag.atoms.tempfactors))
    import MDAnalysis as mda
    pdb_u = mda.Universe(pdb_path)
    cdr1a_bfactors_avg = avg(pdb_u.select_atoms('chainID D and resid 25:42')) #.atoms.bfactors.mean()
    cdr1b_bfactors_avg = avg(pdb_u.select_atoms('chainID E and resid 25:42')) #.atoms.bfactors.mean()
    cdr2a_bfactors_avg = avg(pdb_u.select_atoms('chainID D and resid 58:77')) #.atoms.bfactors.mean()
    cdr2b_bfactors_avg = avg(pdb_u.select_atoms('chainID E and resid 58:77')) #.atoms.bfactors.mean()
    return cdr1a_bfactors_avg, cdr1b_bfactors_avg, cdr2a_bfactors_avg, cdr2b_bfactors_avg


def is_chain_in_pdb(pdb_path: str, input_chain_id: str) -> Tuple[bool, int, List[str]]:
    """
    Return (chain_exists, n_chains, present_chain_ids)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sample", pdb_path)
    model = structure[0]
    pdb_chain_ids = [str(chain.get_id()) for chain in model]
    chain_exists = input_chain_id in pdb_chain_ids
    return chain_exists, len(pdb_chain_ids), pdb_chain_ids

def run_docking(binary: str, pdb_path: str, mode: int = 0) -> str:
    """
    Run the docking executable and return its stdout as text.
    mode is passed as a string arg (required by subprocess).
    """
    cmd = [binary, pdb_path, str(mode)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Docking command failed (exit {e.returncode}).\n"
            f"CMD: {' '.join(cmd)}\nSTDERR:\n{e.stderr}"
        )
    
def parse_docking_stdout(stdout: str) -> Dict[str, object]:
    """
    Extract:
      - TCR center shift vector (Å) as a tuple of 3 floats
      - docking angle (float)
      - incident angle (float)

    Expected fragments:
      "TCR center shift (Å): 10.6192 -2.34785 2.77503"
      "ANGLES: 60.1212 15.1668 22.225"
    """
    # center shift vector
    m_center = re.search(
        r"TCR\s+center\s+shift\s*\(Å\)\s*:\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)",
        stdout, flags=re.IGNORECASE
    )
    if not m_center:
        raise ValueError("Could not find 'TCR center shift (Å): ...' line.")
    center_shift: Tuple[float, float, float] = tuple(map(float, m_center.groups()))

    # angles (order in your output is: docking, incident, <third>)
    m_angles = re.search(
        r"ANGLES\s*:\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)",
        stdout, flags=re.IGNORECASE
    )
    if not m_angles:
        raise ValueError("Could not find 'ANGLES: ...' line.")
    docking_angle = float(m_angles.group(1))
    incident_angle = float(m_angles.group(2))
    # third = float(m_angles.group(3))  # available if you need it

    return {
        "center_shift": center_shift,  # (x, y, z)
        "docking_angle": docking_angle,
        "incident_angle": incident_angle,
    }

if __name__=="__main__":
    base_dir = "/rsrch3/scratch/genomic_med/kkim14/data/khaled_killing/tcrmodel2_out"

    mhc_a = "A"
    mhc_b = None
    pep = "C"
    tcr_a = "D"
    tcr_b = "E"

    BIN = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                   "tcr_docking_angle", "tcr_docking_angle")

    # sanity check: path exists and is executable
    if not os.path.isfile(BIN):
        raise FileNotFoundError(f"Docking binary not found: {BIN}")
    if not os.access(BIN, os.X_OK):
        raise PermissionError(f"Docking binary is not executable: {BIN}")
    
    for root, dirs, files in os.walk(base_dir):
        if "statistics.json" not in files:
            continue
        
        json_path = os.path.join(root, "statistics.json")
        try:
            with open(json_path, "r") as f:
                statistics = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read {json_path}: {e}")
            continue
        
        models_list = [
            f for f in glob(os.path.join(root, "ranked_*.pdb"))
            if re.match(r".*/ranked_[0-9]\.pdb$", f) and "_proj" not in f
            ]
        for model in models_list:
            cdr1a_bfactors_avg, cdr1b_bfactors_avg, cdr2a_bfactors_avg, cdr2b_bfactors_avg = get_cdr_conf(model)
            statistics[os.path.basename(model).split('.pdb')[0]]['cdr1a_plddt'] = cdr1a_bfactors_avg
            statistics[os.path.basename(model).split('.pdb')[0]]['cdr1b_plddt'] = cdr1b_bfactors_avg
            statistics[os.path.basename(model).split('.pdb')[0]]['cdr2a_plddt'] = cdr2a_bfactors_avg
            statistics[os.path.basename(model).split('.pdb')[0]]['cdr2b_plddt'] = cdr2b_bfactors_avg
        
        with open(json_path, "w") as f:
            json.dump(statistics, f, indent=4)

        for pdbid, stats in statistics.items():
            if not pdbid.startswith("ranked"):
                continue

            pdb_path = os.path.join(root, f"{pdbid}.pdb")
            input_chain_IDs = list(filter(None, [mhc_a, mhc_b, tcr_a, tcr_b]))
            input_chain_IDs_upper = [x.upper() for x in input_chain_IDs]

            if os.path.exists(f"{pdb_path.replace('.pdb', '_proj.pdb')}"):
                print(f"⏭️  Skipping {pdbid}: projected pdb already present.")
                continue

            # 2) Validate chains; if any chain missing, print present chains and continue to next pdbid
            missing_chain = False
            present_chains_for_msg = None
            for input_chain_id in input_chain_IDs_upper:
                chain_bool, n_chains, present_chains = is_chain_in_pdb(pdb_path, input_chain_id)
                if not chain_bool:
                    print(
                        f"⚠️ Chain '{input_chain_id}' not found in '{pdb_path}'. "
                        f"Present chains: {present_chains}"
                    )
                    missing_chain = True
                    present_chains_for_msg = present_chains
                    break
                # keep the original strict check on chain count if you want
                if n_chains < 3 or n_chains > 5:
                    raise ValueError(
                        "The submitted PDB file contains an unexpected number of chains! (expected 5 chains)"
                    )
            if missing_chain:
                # skip this pdbid entirely
                continue

            # 3) Geometry (Class I case)
            if mhc_b is None:
                r, theta, phi = tcr_mhci_geometrical_parameters(pdb_path, mhc_a, tcr_a, tcr_b)
                statistics[pdbid]['binding_geometry_r'] = r
                statistics[pdbid]['binding_geometry_theta'] = theta
                statistics[pdbid]['binding_geometry_phi'] = phi

            # 4) Docking metrics
            if isinstance(stats, dict) and ("docking_angle" in stats) and (stats["docking_angle"] is not None):
                print(f"⏭️  Skipping {pdbid}: 'docking_angle' already present.")
                continue

            docking_result = run_docking(BIN, pdb_path, mode=0)
            parsed = parse_docking_stdout(docking_result)
            statistics[pdbid]['docking_angle'] = parsed['docking_angle']
            statistics[pdbid]['incident_angle'] = parsed['incident_angle']
            statistics[pdbid]['center_shift'] = parsed['center_shift']

            # 5) Interdomain angle via cuboid alignment
            if isinstance(stats, dict) and ("interdomain_angle" in stats) and (stats["interdomain_angle"] is not None):
                print(f"⏭️  Skipping {pdbid}: 'interdomain_angle' already present.")
                continue

            ref_path = "/rsrch3/home/genomic_med/kkim14/programs/TCR-CoM/Python-code/dependancies/ref_files/ref1.pdb"
            a_variable_res = list(range(0, 110))
            b_variable_res = list(range(0, 120))
            cuboid_scale = (40, 30, 25)

            parser = PDBParser(QUIET=True)

            print(f"[+] Loading reference structure: {ref_path}")
            ref_structure = parser.get_structure("ref", ref_path)[0]
            ref_coords = get_framework_coords(ref_structure, tcr_a, a_variable_res)
            ref_center, ref_axes, ref_corners = compute_cuboid(ref_coords, scale=cuboid_scale)

            mob_structure = parser.get_structure("mob", pdb_path)[0]
            mob_structure, rmsd = superimpose_structures(
                ref_structure=ref_structure,
                mob_structure=mob_structure,
                ref_chain=tcr_a,
                mob_chain=tcr_a,
                res_range=(min(a_variable_res), max(a_variable_res)),
                out_pdb=None
            )

            mob_a_coords = get_framework_coords(mob_structure, tcr_a, a_variable_res)
            mob_b_coords = get_framework_coords(mob_structure, tcr_b, b_variable_res)

            mob_center, mob_axes, mob_corners = compute_cuboid(mob_a_coords, scale=cuboid_scale)
            _, vb_axes, _ = compute_cuboid(mob_b_coords, scale=cuboid_scale)

            inter_angle = interdomain_angle(mob_axes, vb_axes)
            statistics[pdbid]['interdomain_angle'] = inter_angle

        # write back (same filename)
        with open(json_path, "w") as f:
            json.dump(statistics, f, indent=4)