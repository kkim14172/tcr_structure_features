import glob
import numpy as np
from Bio.PDB import PDBParser, Superimposer, PDBIO
from scipy.spatial.transform import Rotation

def superimpose_structures(
    ref_structure,
    mob_structure,
    ref_chain="A",
    mob_chain="A",
    res_range=(1, 180),
    out_pdb=None
):
    """
    Superimpose two protein structures based on Cα atoms and return the aligned mobile structure.

    Parameters
    ----------
    ref_path : str
        Path to the reference PDB file.
    mob_path : str
        Path to the mobile (to be aligned) PDB file.
    ref_chain : str
        Chain ID in the reference structure to use for alignment.
    mob_chain : str
        Chain ID in the mobile structure to be aligned.
    res_range : tuple(int, int)
        Residue index range (inclusive) used for alignment.
    out_pdb : str, optional
        Path to save the aligned structure. If None, no file is written.

    Returns
    -------
    mob_structure : Bio.PDB.Structure.Structure
        Mobile structure object after alignment (in memory).
    rmsd : float
        Root-mean-square deviation after alignment (Å).
    """
    # Select Cα atoms within residue range
    ref_atoms, mob_atoms = [], []
    for r1, r2 in zip(ref_structure[ref_chain], mob_structure[mob_chain]):
        if "CA" not in r1 or "CA" not in r2:
            continue
        res_id = r1.id[1]
        if res_range[0] <= res_id <= res_range[1]:
            ref_atoms.append(r1["CA"])
            mob_atoms.append(r2["CA"])

    if not ref_atoms or not mob_atoms:
        raise ValueError("No matching Cα atoms found within specified range.")

    # Perform superposition
    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    sup.apply(mob_structure.get_atoms())

    rmsd = sup.rms
    print(f"[✓] RMSD after alignment: {rmsd:.3f} Å")

    # Optionally save to file
    if out_pdb:
        io = PDBIO()
        io.set_structure(mob_structure)
        io.save(out_pdb)
        print(f"[✓] Saved aligned structure → {out_pdb}")

    return mob_structure, rmsd


def compute_cuboid(coords, scale=(40, 30, 25)):
    """
    Construct a cuboid (rectangular box) aligned with the principal axes
    of the input coordinates (e.g., V-domain framework Cα atoms).

    Parameters
    ----------
    coords : (N, 3) ndarray
        Cartesian coordinates of framework atoms (e.g., Cα positions)
    scale : tuple of float
        Cuboid dimensions along principal axes (Å)

    Returns
    -------
    center : ndarray (3,)
        Center of the cuboid (mean position)
    axes : ndarray (3, 3)
        Principal axes as column vectors
    corners : ndarray (8, 3)
        Cartesian coordinates of cuboid vertices
    """
    # Center the coordinates
    center = np.mean(coords, axis=0)
    coords_centered = coords - center

    # PCA to find principal axes
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # largest first
    axes = eigvecs[:, order]

    # Half-lengths along each principal direction
    half_lengths = np.array(scale) / 2.0

    # Construct all 8 corners (± along each axis)
    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                offset = sx * half_lengths[0] * axes[:, 0] \
                       + sy * half_lengths[1] * axes[:, 1] \
                       + sz * half_lengths[2] * axes[:, 2]
                corners.append(center + offset)

    corners = np.array(corners)
    return center, axes, corners

def interdomain_angle(axes_1, axes_2):
    """
    Compute the interdomain angle (twist) between Vα and Vβ domains.

    Parameters
    ----------
    axes_1 : ndarray, shape (3, 3)
        Principal axes of the first cuboid (e.g., Vα domain)
    axes_2 : ndarray, shape (3, 3)
        Principal axes of the second cuboid (e.g., Vβ domain)

    Returns
    -------
    float
        Angle (degrees) between the first principal axes of the two domains.
    """
    v_a = axes_1[:, 0]
    v_b = axes_2[:, 0]

    # compute cosine of the angle
    cosang = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b))
    # ensure numerical safety (avoid arccos domain error)
    cosang = np.clip(cosang, -1.0, 1.0)

    # convert to degrees
    angle_main = np.degrees(np.arccos(cosang))
    return angle_main

def get_framework_coords(structure, chain_id, resid_subset):
    """
    Extracts framework Cα coordinates from a PDB structure.

    Parameters
    ----------
    structure : pdb.Structure
        
    chain_id : str
        Chain ID (e.g., 'D' or 'E')
    resid_subset : list of int
        Residue IDs used for framework superposition

    Returns
    -------
    np.ndarray of shape (N, 3)
    """
    chain = structure[chain_id]
    coords = []
    for res in chain:
        if res.id[1] in resid_subset and "CA" in res:
            coords.append(res["CA"].coord)
    return np.array(coords)

def rotation_to_euler(R, order='xyz', degrees=True):
    """
    Convert a rotation matrix to Euler angles (φ, ψ, θ).
    """
    r = Rotation.from_matrix(R)
    return r.as_euler(order, degrees=degrees)

def compute_relative_euler(R_i, R_ref, order='xyz'):
    """
    Compute Euler angles of R_i relative to reference rotation R_ref.
    """
    R_rel = R_i @ R_ref.T
    return rotation_to_euler(R_rel, order=order)

def euler_distance_matrix(euler_angles):
    """
    Compute Euclidean distance matrix between all Euler-angle triplets.
    euler_angles: array (N, 3)
    """
    n = len(euler_angles)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(euler_angles[i] - euler_angles[j])
            D[i, j] = D[j, i] = d
    return D

def generate_grid(center, axes, scale=(40, 30, 25), spacing=2.0):
    """
    Generate a 3D grid of points inside the cuboid aligned with given principal axes.

    Parameters
    ----------
    center : (3,) array
        Cuboid center.
    axes : (3,3) array
        Columns are unit vectors defining cuboid orientation.
    scale : tuple
        Cuboid side lengths (Å).
    spacing : float
        Grid spacing (Å).

    Returns
    -------
    grid_points : (N,3) array
        Cartesian coordinates of all grid points.
    """
    nx, ny, nz = [int(s / spacing) for s in scale]
    xs = np.linspace(-scale[0]/2, scale[0]/2, nx)
    ys = np.linspace(-scale[1]/2, scale[1]/2, ny)
    zs = np.linspace(-scale[2]/2, scale[2]/2, nz)

    grid_local = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    grid_global = grid_local @ axes.T + center
    return grid_global

def pairwise_distances(coords):
    """
    Compute pairwise Euclidean distances for N 3D coordinates.
    Returns an (N,N) symmetric matrix.
    """
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diffs, axis=2)

def variance_gridpoint(grid_coords):
    """
    Compute variance var(g_i) for one grid index across all structures
    using the equation from Hoffmann et al. (Front. Immunol. 2020).

    Parameters
    ----------
    grid_coords : (n, 3) ndarray
        Coordinates of the same grid point across n structures.

    Returns
    -------
    float
        Variance value for this grid point.
    """
    n = grid_coords.shape[0]
    delta = pairwise_distances(grid_coords)  # δ_s(g_i,k, g_i,l)
    delta_mean = np.mean(delta)
    term = delta - delta_mean
    var_gi = np.sum(term) / (n**2 - 1)
    return var_gi

def compute_CoR_variance(grid_sets):
    """
    Compute var(g_i) for all grid points (Eq. in image).

    Parameters
    ----------
    grid_sets : list of (N,3) arrays
        List of identically indexed grids from n structures.

    Returns
    -------
    var_profile : (N,) ndarray
        Variance for each grid index.
    CoR_index : int
        Index of minimal variance point.
    CoR_coord : (3,) ndarray
        Coordinate of CoR (mean position at that index).
    """
    grids = np.stack(grid_sets, axis=0)   # shape (n, N, 3)
    n, N, _ = grids.shape

    var_profile = np.zeros(N)
    for i in range(N):
        var_profile[i] = variance_gridpoint(grids[:, i, :])

    CoR_index = np.argmin(var_profile)
    CoR_coord = np.mean(grids[:, CoR_index, :], axis=0)
    return var_profile, CoR_index, CoR_coord


if __name__=="__main__":
    # Path to reference and sample PDBs
    ref_path = "/rsrch3/home/genomic_med/kkim14/programs/TCR-CoM/Python-code/dependancies/ref_files/ref1.pdb"
    sample_dir = "/rsrch3/home/genomic_med/kkim14/programs/tcrmodel2/experiments/test_clsI/*.pdb"  # folder with all TCRs to compare
    sample_paths = sorted(glob.glob(sample_dir))

    # Framework residues used for alignment (example)
    a_variable_res = list(range(0, 110))  # typical Vα framework region
    b_variable_res = list(range(0, 120))  # typical Vα framework region

    # Chain to align (Vα)
    a_chain = "D"
    b_chain = "E"

    # Cuboid and grid parameters
    cuboid_scale = (40, 30, 25)
    grid_spacing = 2.0

    parser = PDBParser(QUIET=True)

    print(f"[+] Loading reference structure: {ref_path}")
    ref_structure = parser.get_structure("ref", ref_path)[0]
    ref_coords = get_framework_coords(ref_structure, a_chain, a_variable_res)
    ref_center, ref_axes, ref_corners = compute_cuboid(ref_coords, scale=cuboid_scale)

    # Generate a standard grid for reference
    ref_grid = generate_grid(ref_center, ref_axes, scale=cuboid_scale, spacing=grid_spacing)

    #############################
    # 3. Load and Superimpose All Samples
    #############################

    grid_sets = []
    for pdb_path in sample_paths:
        print(f"\n[+] Processing: {pdb_path}")
        mob_structure = parser.get_structure("mob", pdb_path)[0]
        
        # Superimpose onto reference using Cα atoms in Vα framework
        mob_structure, rmsd = superimpose_structures(
            ref_structure=ref_structure,
            mob_structure=mob_structure,
            ref_chain=a_chain,
            mob_chain=a_chain,
            res_range=(min(a_variable_res), max(a_variable_res)),
            out_pdb=None  # no file saving, in-memory only
        )

        # Extract aligned framework Cα coordinates
        mob_a_coords = get_framework_coords(mob_structure, a_chain, a_variable_res)
        mob_b_coords = get_framework_coords(mob_structure, b_chain, b_variable_res)

        # Construct cuboid and grid for this structure
        mob_center, mob_axes, mob_corners = compute_cuboid(mob_a_coords, scale=cuboid_scale)
        _, vb_axes, _ = compute_cuboid(mob_b_coords, scale=cuboid_scale)
        
        inter_angle = interdomain_angle(mob_axes, vb_axes)
        print(f"[✓] Interdomain twist angle: {inter_angle:.6f} ")

        mob_grid = generate_grid(mob_center, mob_axes, scale=cuboid_scale, spacing=grid_spacing)
        grid_sets.append(mob_grid)

    print(f"\n[✓] Loaded and aligned {len(grid_sets)} structures.")

    #############################
    # 4. Compute CoR Variance
    #############################

    var_profile, CoR_index, CoR_coord = compute_CoR_variance(grid_sets)
    print(f"\n[✓] Center of Rotation (CoR) found at grid index {CoR_index}")
    print(f"    CoR coordinate (Å): {np.round(CoR_coord, 3)}")

    # Identify low-variance region
    vmin, vmax = np.min(var_profile), np.max(var_profile)
    print(f"    Variance range: {vmin:.6f} – {vmax:.6f}")





    # from typing import List, Union, Optional, Literal, Tuple, Dict
    # import numpy as np
    # import MDAnalysis as mda
    # from dataclasses import dataclass

    # # ---------------------------- math utils ----------------------------

    # def _project_to_SO3(M: np.ndarray) -> np.ndarray:
    #     """Project a 3x3 matrix to the closest proper rotation (det=+1)."""
    #     U, _, Vt = np.linalg.svd(M)
    #     R = U @ Vt
    #     if np.linalg.det(R) < 0:
    #         U[:, -1] *= -1
    #         R = U @ Vt
    #     return R

    # def apply_rigid(xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    #     """Apply rigid transform to Nx3 coordinates."""
    #     return (R @ xyz.T).T + t

    # # --------------------------- data structure -------------------------

    # @dataclass
    # class CuboidFrame:
    #     # World-frame definition (Å)
    #     origin_world: np.ndarray      # min-corner (local [xmin,ymin,zmin]) mapped to world
    #     axes_world: np.ndarray        # 3x3, columns are unit x,y,z axes in world frame
    #     lengths: np.ndarray           # (lx, ly, lz), full side lengths (Å)
    #     center_world: np.ndarray      # cuboid center in world frame

    #     # Convenience transforms
    #     def to_local(self, xyz_world: np.ndarray) -> np.ndarray:
    #         # local = A^T (x - origin)
    #         A = np.asarray(self.axes_world, float)
    #         return (A.T @ (np.asarray(xyz_world, float) - self.origin_world).T).T

    #     def to_world(self, xyz_local: np.ndarray) -> np.ndarray:
    #         # world = origin + A (local)
    #         A = np.asarray(self.axes_world, float)
    #         return self.origin_world + (A @ np.asarray(xyz_local, float).T).T

    # # ----------------------- cuboid construction ------------------------

    # def build_cuboid_local_frame(
    #     domain: Union[mda.core.groups.AtomGroup, mda.Universe],
    #     *,
    #     selector: Optional[str] = None,     # e.g. 'chainID D and protein and name CA'
    #     method: Literal["pca", "inertia"] = "pca",
    #     use_masses: bool = True,
    #     padding: float = 0.0                # Å added to each min/max extent
    # ) -> CuboidFrame:
    #     """
    #     Define a rectangular cuboid around a domain, aligned to its principal axes.
    #     Returns a right-handed local coordinate system and world<->local transforms.
    #     """
    #     # --- collect atoms ---
    #     if isinstance(domain, mda.Universe):
    #         if not selector:
    #             raise ValueError("When passing a Universe, you must provide `selector`.")
    #         ag = domain.atoms.select_atoms(selector)
    #     else:
    #         ag = domain
    #     if ag.n_atoms < 3:
    #         raise ValueError("Need at least 3 atoms in the domain.")

    #     X = ag.positions.astype(float)  # (N,3)

    #     # masses / weights
    #     masses_ok = hasattr(ag, "masses") and np.all(np.isfinite(ag.masses))
    #     if use_masses and masses_ok:
    #         m = ag.masses.reshape(-1, 1)
    #         w = m / m.sum()
    #         center = (w * X).sum(axis=0)
    #         Xc = X - center
    #     else:
    #         center = X.mean(axis=0)
    #         Xc = X - center
    #         m = None  # only used for inertia

    #     # --- principal axes ---
    #     if method == "pca":
    #         Xcw = np.sqrt(w) * Xc if use_masses and masses_ok else Xc
    #         _, _, Vt = np.linalg.svd(Xcw, full_matrices=False)
    #         A = Vt.T
    #     elif method == "inertia":
    #         if m is None:
    #             m = np.ones((Xc.shape[0], 1))
    #         r2 = np.sum(Xc**2, axis=1, keepdims=True)
    #         I = (m * r2).sum() * np.eye(3) - (Xc * m)`.T @ Xc
#         evals, evecs = np.linalg.eigh(I)
#         A = evecs[:, np.argsort(evals)]  # columns = principal axes
#     else:
#         raise ValueError("method must be 'pca' or 'inertia'")

#     # Orthonormalize & enforce right-handed frame
#     A = _project_to_SO3(A)

#     # --- compute tight extents in local frame ---
#     u_centered = Xc @ A
#     umin = u_centered.min(axis=0) - padding
#     umax = u_centered.max(axis=0) + padding

#     lengths = umax - umin
#     center_local = 0.5 * (umax + umin)
#     origin_local = umin

#     center_world = center + A @ center_local
#     origin_world = center + A @ origin_local

#     return CuboidFrame(
#         origin_world=np.asarray(origin_world, float),
#         axes_world=np.asarray(A, float),
#         lengths=np.asarray(lengths, float),
#         center_world=np.asarray(center_world, float),
#     )

# # --------------------- alignment & pose measures --------------------

# def cuboid_superimpose(
#     moving_frame: CuboidFrame,
#     fixed_frame: CuboidFrame,
#     *,
#     anchor: Literal["center", "origin"] = "center",
#     apply_to: Optional[Union[mda.Universe, mda.core.groups.AtomGroup]] = None
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Compute rigid transform (R, t) that maps moving cuboid into fixed cuboid frame.
#     Optionally apply it to an MDAnalysis Universe/AtomGroup.

#     Returns
#     -------
#     R : (3,3)
#     t : (3,)
#     """
#     A_fix = np.asarray(fixed_frame.axes_world, float)
#     A_mov = np.asarray(moving_frame.axes_world, float)
#     A_fix = _project_to_SO3(A_fix)
#     A_mov = _project_to_SO3(A_mov)

#     p_fix = fixed_frame.center_world if anchor == "center" else fixed_frame.origin_world
#     p_mov = moving_frame.center_world if anchor == "center" else moving_frame.origin_world

#     R = _project_to_SO3(A_fix @ A_mov.T)
#     t = p_fix - R @ p_mov

#     if apply_to is not None:
#         if isinstance(apply_to, mda.Universe):
#             coords = apply_to.atoms.positions
#             apply_to.atoms.positions = apply_rigid(coords, R, t)
#         else:
#             coords = apply_to.positions
#             apply_to.positions = apply_rigid(coords, R, t)

#     return R, t

# def _rot_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
#     """Convert a proper rotation matrix (3x3) to axis-angle (unit axis, degrees)."""
#     R = np.asarray(R, float)
#     tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
#     theta = np.arccos(tr)

#     if np.isclose(theta, 0.0):
#         return np.array([1.0, 0.0, 0.0]), 0.0

#     if np.isclose(theta, np.pi):
#         x = np.sqrt(max(0.0, (R[0, 0] + 1) / 2.0))
#         y = np.sqrt(max(0.0, (R[1, 1] + 1) / 2.0))
#         z = np.sqrt(max(0.0, (R[2, 2] + 1) / 2.0))
#         x = np.copysign(x, R[2, 1] - R[1, 2])
#         y = np.copysign(y, R[0, 2] - R[2, 0])
#         z = np.copysign(z, R[1, 0] - R[0, 1])
#         axis = np.array([x, y, z])
#     else:
#         axis = np.array([
#             R[2, 1] - R[1, 2],
#             R[0, 2] - R[2, 0],
#             R[1, 0] - R[0, 1],
#         ]) / (2.0 * np.sin(theta))

#     axis = axis / np.linalg.norm(axis)
#     return axis, float(np.degrees(theta))

# def _euler_zyx_from_R(R: np.ndarray) -> Tuple[float, float, float]:
#     """ZYX Euler angles (roll X, pitch Y, yaw Z) in degrees for R_rel."""
#     R = np.asarray(R, float)
#     if np.isclose(R[2,0], -1.0):
#         yaw = np.pi/2; pitch = 0.0; roll = np.arctan2(R[0,1], R[0,2])
#     elif np.isclose(R[2,0],  1.0):
#         yaw = -np.pi/2; pitch = 0.0; roll = np.arctan2(-R[0,1], -R[0,2])
#     else:
#         yaw   = np.arcsin(-R[2,0])
#         pitch = np.arctan2(R[2,1], R[2,2])
#         roll  = np.arctan2(R[1,0], R[0,0])
#     return tuple(np.degrees([roll, pitch, yaw]))

# def measure_cuboid_pose(
#     moving_frame: CuboidFrame,
#     fixed_frame: CuboidFrame,
#     *,
#     report_euler: bool = True
# ) -> Dict[str, np.ndarray | float]:
#     """
#     Measure orientation and position of `moving_frame` relative to `fixed_frame`.
#     """
#     A_fix = _project_to_SO3(np.asarray(fixed_frame.axes_world, float))
#     A_mov = _project_to_SO3(np.asarray(moving_frame.axes_world, float))

#     # Relative rotation (moving -> fixed) in fixed basis
#     R_rel = A_fix.T @ A_mov

#     axis, angle_deg = _rot_to_axis_angle(R_rel)

#     def _angle_between(u, v):
#         c = float(np.clip(np.dot(u, v), -1.0, 1.0))
#         return float(np.degrees(np.arccos(c)))

#     axis_deviation_deg = np.array([
#         _angle_between(A_fix[:,0], A_mov[:,0]),
#         _angle_between(A_fix[:,1], A_mov[:,1]),
#         _angle_between(A_fix[:,2], A_mov[:,2]),
#     ])

#     c_fix = np.asarray(fixed_frame.center_world, float)
#     c_mov = np.asarray(moving_frame.center_world, float)
#     o_fix = np.asarray(fixed_frame.origin_world, float)
#     o_mov = np.asarray(moving_frame.origin_world, float)

#     center_offset_fixed = A_fix.T @ (c_mov - c_fix)
#     origin_offset_fixed = A_fix.T @ (o_mov - o_fix)

#     out: Dict[str, np.ndarray | float] = {
#         "R_rel": R_rel,
#         "axis": axis,
#         "angle_deg": angle_deg,
#         "axis_deviation_deg": axis_deviation_deg,   # [Δx, Δy, Δz]
#         "center_offset_fixed": center_offset_fixed, # (Å)
#         "origin_offset_fixed": origin_offset_fixed, # (Å)
#     }
#     if report_euler:
#         out["euler_zyx_deg"] = np.array(_euler_zyx_from_R(R_rel))
#     return out


# if __name__=="__main__":
#     u = mda.Universe("complex1.pdb")
#     v = mda.Universe("complex2.pdb")

#     fixed = build_cuboid_local_frame(u, selector="chainID D and protein and name CA", method="pca", use_masses=False)
#     moving = build_cuboid_local_frame(v, selector="chainID D and protein and name CA", method="pca", use_masses=False)

#     # Measure without modifying coordinates
#     metrics = measure_cuboid_pose(moving, fixed)
#     print(metrics["angle_deg"], metrics["axis_deviation_deg"], metrics["center_offset_fixed"])

#     # Get and (optionally) apply the rigid transform that aligns 'moving' to 'fixed'
#     R, t = cuboid_superimpose(moving, fixed, anchor="center")
#     # apply to the whole Universe
#     _ = cuboid_superimpose(moving, fixed, anchor="center", apply_to=v)
