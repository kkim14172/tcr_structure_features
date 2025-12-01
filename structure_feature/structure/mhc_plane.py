import numpy as np
from .plane_projection import Plane

# From PDB file
# Define MHC binding plane with central carbon atoms in the range of 1~179 MHC alpha resid.
mhc_chain_id = "A"
mhc_beta_sheet_resid = 95
atom_name = 'CA'

def mhc_plane(model):
    # Get the first model
    plane_points = model[
        (model.chain_id == mhc_chain_id) &
        (model.atom_name == "CA") &
        (np.isin(model.res_id, np.arange(1, 179+1)))
    ].coord  # -> (N, 3)
    
    op_coords = model[
        (model.chain_id == mhc_chain_id) &
        (model.atom_name == "CA") &
        (model.res_id == mhc_beta_sheet_resid)
    ].coord  # could be (1,3) or (k,3) if duplicates

    if op_coords.size == 0:
        raise ValueError(f"No CA found at resid {mhc_beta_sheet_resid} on chain {mhc_chain_id}.")
    opposite_point = op_coords[0] if op_coords.ndim == 2 else op_coords  # (3,)

    plane = Plane(plane_points, toward_point=None, opposite_point=opposite_point)
    return plane