#!/usr/bin/env python
## - Reference-dependent approach
"""Import modules"""
from __future__ import print_function
import sys
import os
import argparse
import numpy as np
import Bio.PDB
from Bio.PDB import Entity, Chain, Residue, Atom, PDBParser, PDBIO, Select
from Bio.PDB.vectors import Vector, rotaxis2m

reffile1 = os.path.join("/rsrch3/home/genomic_med/kkim14/programs/TCR-CoM/Python-code", "dependancies","ref_files","ref1.pdb")


def center_of_mass(entity, geometric=False):
    # Copyright (C) 2010, Joao Rodrigues (anaryin@gmail.com)
    # This code is part of the Biopython distribution and governed by its
    # license.  Please see the LICENSE file that should have been included
    # as part of this package.
    """
    Returns gravitic [default] or geometric center of mass of an Entity.
    Geometric assumes all masses are equal (geometric=True)
    """

    # Structure, Model, Chain, Residue
    if isinstance(entity, Entity.Entity):
        atom_list = entity.get_atoms()
    # List of Atoms
    elif hasattr(entity, "__iter__") and [x for x in entity if x.level == "A"]:
        atom_list = entity
    else:  # Some other weirdo object
        raise ValueError(
            "Center of Mass can only be calculated from the following objects:\n"
            "Structure, Model, Chain, Residue, list of Atoms."
        )

    masses = []
    positions = [[], [], []]  # [ [X1, X2, ..] , [Y1, Y2, ...] , [Z1, Z2, ...] ]

    for atom in atom_list:
        masses.append(atom.mass)

        for i, coord in enumerate(atom.coord.tolist()):
            positions[i].append(coord)

    # If there is a single atom with undefined mass complain loudly.
    if "ukn" in set(masses) and not geometric:
        raise ValueError(
            "Some Atoms don't have an element assigned.\n"
            "Try adding them manually or calculate the geometrical center of mass instead"
        )

    if geometric:
        return [sum(coord_list) / len(masses) for coord_list in positions]
    else:
        w_pos = [[], [], []]
        for atom_index, atom_mass in enumerate(masses):
            w_pos[0].append(positions[0][atom_index] * atom_mass)
            w_pos[1].append(positions[1][atom_index] * atom_mass)
            w_pos[2].append(positions[2][atom_index] * atom_mass)

        return [sum(coord_list) / sum(masses) for coord_list in w_pos]


def fetch_atoms(model, selection="A", atom_bounds=[1, 180]):
    """
    Function to fetch atoms from the defined "atom_bound" in "selection" of the "model"
    """
    selection = selection.upper()
    if not isinstance(atom_bounds, (list, tuple)) and len(atom_bounds) > 0:
        raise ValueError("expected non-empty list or tuple, got {}".format(atom_bounds))
    # making sure its a list of lists
    if not isinstance(atom_bounds[0], (list, tuple)):
        atom_bounds = [atom_bounds]
    if not all([len(b) == 2 for b in atom_bounds]):
        raise ValueError(
            "All bounds must be providing one upper "
            "and one lower bound, got {}".format(atom_bounds)
        )
    if not isinstance(selection, (tuple, list)):
        selection = [selection]
    result = []
    for sel in selection:
        for ref_res in model["%s" % sel]:
            resid = ref_res.get_id()[1]
            in_bounds = False
            for bounds in atom_bounds:
                in_bounds |= bounds[0] <= resid and resid <= bounds[1]
            if in_bounds:
                result.append(ref_res["CA"])
    return result


def fetch_entity(model, fetch_atoms=True, selection="A", res_ids=range(1, 180)):
    """
    Function to fetch atoms/resids from the defined "resid_bounds" in "selection" of the "model"
    """
    selection = selection.upper()
    # fetch atoms
    if fetch_atoms is True:
        result = []
        for sel in selection:
            for sample_res in model["%s" % sel]:
                resid = sample_res.get_id()[1]
                if resid in res_ids:
                    result.append(sample_res["CA"])
    # fetch_residues_indeces
    elif fetch_atoms is False:
        result = []
        for sel in selection:
            for sample_res in model["%s" % sel]:
                resid = sample_res.get_id()[1]
                if resid in res_ids:
                    result.append(resid)
    return result


def apply_transformation_to_atoms(model, rotmat, transvec):
    """
    Function to translate/rotate the model by the defined translation vector and rotation matrix
    """
    for chain in model:
        for res in chain:
            for atom in res:
                atom.transform(rotmat, transvec)


def is_chain_in_pdb(pdb_path, input_chain_id):
    """
    Function to check if a given chain exists in the PDB file.
    Returns (chain_exists, n_chains)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("sample", pdb_path)
    model = structure[0]

    pdb_chain_ids = [str(chain.get_id()) for chain in model]
    chain_exists = input_chain_id in pdb_chain_ids

    return chain_exists, len(pdb_chain_ids)


"""
Geometrical parameters of TCR-MHC class I
"""
def tcr_mhci_geometrical_parameters(
    pdb_path,
    mhc_a="A",
    tcr_a="D",
    tcr_b="E",
    mhc_a_init=1,
    mhc_a_final=179,
    tcr_a_init=1,
    tcr_a_final=109,
    tcr_b_init=1,
    tcr_b_final=116,
    ):
    """
    ARGUMENTS:
    pdbid = PDB-ID or name of the input PB file (str)
    mhc_a = ID of MHC alpha chain (str)
    tcr_a = ID of TCR alpha chain (str)
    tcr_b = ID of TCR beta chain (str)
    persist_structure = If true, save the processed structure as PDB-file (Boolean)
    """
    #################################################################
    # Define residues range to align and center of mass calculations#
    #################################################################
    mhc_a_resids = range(mhc_a_init, mhc_a_final + 1)
    tcr_a_resids = range(tcr_a_init, tcr_a_final + 1)
    tcr_b_resids = range(tcr_b_init, tcr_b_final + 1)

    ########################################################################################################
    # Import structure, align to reference, and calculate center of mass of CA atoms in MHCI binding groove#
    ########################################################################################################
    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    ref_structure = pdb_parser.get_structure("reference", reffile1)
    sample_structure = pdb_parser.get_structure("sample", pdb_path)
    # Use the first model in the pdb-files for alignment
    ref_model = ref_structure[0]
    sample_model = sample_structure[0]
    # Iterate of all residues in each model in order to define proper atoms
    # Sample structure
    sample_resids = fetch_entity(
        sample_model, fetch_atoms=False, selection=mhc_a, res_ids=mhc_a_resids
    )
    sample_atoms = fetch_entity(
        sample_model, fetch_atoms=True, selection=mhc_a, res_ids=sample_resids
    )
    # Reference structure
    ref_atoms = fetch_entity(
        ref_model, fetch_atoms=True, selection=mhc_a, res_ids=sample_resids
    )

    # Initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    super_imposer.apply(sample_model.get_atoms())
    
    # Calculate CoM of MHCI binding groove
    mhci_com = center_of_mass(sample_atoms, geometric=True)

    # Calculate CoM of vTCR
    tcr_atoms_for_com = fetch_entity(
        sample_model, fetch_atoms=True, selection=tcr_a, res_ids=tcr_a_resids
    )
    tcr_atoms_for_com += fetch_entity(
        sample_model, fetch_atoms=True, selection=tcr_b, res_ids=tcr_b_resids
    )
    vtcr_com = center_of_mass(tcr_atoms_for_com, geometric=True)
    print("MHC-CoM: ", [round(x, 2) for x in mhci_com])
    print("vTCR-CoM: ", [round(x, 2) for x in vtcr_com])
    
    def _calculate_com(tcr_com, mhc_com):
        dx, dy, dz = np.subtract(tcr_com, mhc_com)
        r = np.sqrt(np.sum(np.square(np.subtract(tcr_com, mhc_com))))
        theta = np.degrees(np.arctan2(dy, dx))
        phi = np.degrees(np.arccos(dz / r))
        print(
            "The Geomitrical parameters: r = {:.2f}, "
            "theta = {:.2f}, phi = {:.2f}".format(r, theta, phi)
        )
        return r, theta, phi

    r, theta, phi = _calculate_com(vtcr_com, mhci_com)
    # atom coordinate is not (u, v, d)
    return (r, theta, phi), sample_structure


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    parser = argparse.ArgumentParser(
        description="TCR-CoM calculates the geometrical parameters (r, theta, phi) of T cell receptors (TCR) on the top of MHC proteins"
    )
    parser.add_argument(
        "--pdb_path", type=str, required=True,
        help="Full or relative path to the input PDB file"
    )
    parser.add_argument("--mhc_a", type=str, default='A', help="ID of MHC alpha chain")
    parser.add_argument("--mhc_b", type=str, default=None, help="ID of MHC beta chain (optional)")
    parser.add_argument("--tcr_a", type=str, default='D', help="ID of TCR alpha chain")
    parser.add_argument("--tcr_b", type=str, default='E', help="ID of TCR beta chain")

    args = parser.parse_args()
    pdb_path = os.path.abspath(args.pdb_path)
    pdbid = os.path.splitext(os.path.basename(pdb_path))[0]
    if pdbid.endswith(".pdb"):
        pdbid = pdbid.split(".")[0]
    else:
        pdbid = pdbid
    mhc_a = args.mhc_a
    mhc_b = args.mhc_b
    tcr_a = args.tcr_a
    tcr_b = args.tcr_b

    input_chain_IDs = list(filter(None, [mhc_a, mhc_b, tcr_a, tcr_b]))
    input_chain_IDs_upper = [x.upper() for x in input_chain_IDs]

    for input_chain_id in input_chain_IDs_upper:
        chain_bool, n_chains = is_chain_in_pdb(pdb_path, input_chain_id)
        if not chain_bool:
            raise ValueError(
                f'Chain "{input_chain_id}" is not found in "{pdb_path}"!'
            )
        if n_chains < 3 or n_chains > 5:
            raise ValueError(
                "The submitted PDB file contains an unexpected number of chains! (expected 5 chains)"
            )

    # Determine which geometry function to use
    if mhc_b is None:
        (r, theta, phi), sample_structure = tcr_mhci_geometrical_parameters(pdbid, mhc_a, tcr_a, tcr_b)