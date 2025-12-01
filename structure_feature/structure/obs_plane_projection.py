import numpy as np

from Bio.PDB.vectors import Vector, rotaxis2m
from copy import deepcopy
from .obs_define_plane_from_points import _safe_unit, define_plane_from_points

def _choose_perp(n):
    """Choose a deterministic vector roughly ⟂ to n for basis construction."""
    n = np.asarray(n, float)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.99 else np.array([0.0, 1.0, 0.0])
    x = seed - np.dot(seed, n) * n
    return _safe_unit(x)


def _z_rotation(phi):
    """Rotation matrix about +Z by angle phi."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)


# --------- GLOBAL (rotation-based) ---------
def global_projection(points, normal, origin, ref_vector=None, return_R=False):
    """
    Project 3D points onto a plane via rotation.

    Steps:
      1) Translate by 'origin'
      2) Build R that maps plane normal -> +Z (deterministic even near 180°)
      3) (Optional) If ref_vector is given, apply a roll about +Z so its projection aligns to +X
      4) In rotated frame, set z=0 (orthogonal projection)
      5) Rotate back (R.T) and translate to original frame

    Args:
        points     : (N,3) array
        normal     : (3,) plane normal (need not be unit; will be normalized)
        origin     : (3,) a point on plane
        ref_vector : (3,) optional, defines +u direction after projection
        return_R   : bool, also return R and in-plane world axes

    Returns:
        proj_2d : (N,2) coordinates in the plane-aligned XY frame (u,v)
        proj_3d : (N,3) projected 3D points in original frame
        distances : (N,) signed perpendicular distances to the plane (+ along normal)
        [optional extras if return_R=True]
          R            : (3,3) rotation that maps plane -> XY
          x_hat_world  : (3,) world-space unit vector of +u
          y_hat_world  : (3,) world-space unit vector of +v

    Round-trip reconstruction:
        uvz = np.c_[proj_2d, np.zeros(len(proj_2d))]
        p3d = (R.T @ uvz.T).T + origin
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")

    n = _safe_unit(normal)              # unit normal
    o = np.asarray(origin, dtype=float).reshape(3)

    # 1) translate
    P0 = P - o

    # 2) rotation n -> ez
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(n, ez), -1.0, 1.0))
    angle = np.arccos(dot)

    if np.isclose(angle, 0.0):
        R = np.eye(3)
    elif np.isclose(angle, np.pi):
        axis = _choose_perp(n)
        R = rotaxis2m(np.pi, Vector(axis))
    else:
        axis = _safe_unit(np.cross(n, ez))
        R = rotaxis2m(angle, Vector(axis))

    # Optional: pin in-plane +u to ref_vector
    if ref_vector is not None:
        rv = np.asarray(ref_vector, float)
        rv_in_plane = rv - np.dot(rv, n) * n
        rv_in_plane = _safe_unit(rv_in_plane)
        rv_rot = R @ rv_in_plane  # now in rotated frame
        phi = np.arctan2(rv_rot[1], rv_rot[0])  # angle to +X
        R = _z_rotation(-phi) @ R                # roll so ref aligns with +X

    # rotate into plane frame (plane -> XY)
    P_rot = (R @ P0.T).T

    # 2D coords before rotating back
    proj_2d = P_rot[:, :2].copy()

    # zero Z for orthogonal projection in rotated frame
    P_rot[:, 2] = 0.0

    # rotate back and translate
    proj_3d = (R.T @ P_rot.T).T + o

    # signed perpendicular distance to the plane (+ along n)
    distances = (P - o) @ n

    if return_R:
        x_hat_world = R.T[:, 0]
        y_hat_world = R.T[:, 1]
        return proj_2d, proj_3d, distances, R, x_hat_world, y_hat_world

    return proj_2d, proj_3d, distances

def _get_plane_projected_coordinates(structure, points):
        from copy import deepcopy
        normal, origin, _ = define_plane_from_points(points)

        projected_structure = deepcopy(structure)
        proj_model = projected_structure[0]

        # Collect all atoms as (N,3) in the *aligned* frame
        all_atoms = [a for a in structure.get_atoms()]
        all_coords = np.array([a.get_coord() for a in all_atoms], float)

        # Project
        proj_2d, proj_3d, d = global_projection(all_coords, normal, origin)

        # Write back into the copied structure in the same atom order
        i = 0
        for chain in proj_model:
            for residue in chain:
                for atom in residue:
                    atom.set_coord(np.append(proj_2d[i], d[i]))
                    i += 1

        return projected_structure


# --------- LOCAL (analytic projection + basis) ---------
def local_projection(points, normal, origin, ref_vector=None, return_basis=False):
    """
    Orthogonally project points and express them in a locally built in-plane basis.

    Basis selection:
      - If ref_vector is provided, +u is the normalized projection of ref_vector onto the plane.
      - Otherwise, choose a deterministic vector ⟂ normal as +u.
      - +v = normal × +u  (right-handed)

    Args:
        points       : (N,3)
        normal       : (3,)
        origin       : (3,)
        ref_vector   : (3,) optional to pin +u
        return_basis : bool, also return (x_axis, y_axis) in world frame

    Returns:
        proj_2d : (N,2) (u,v) in local plane coordinates
        proj_3d : (N,3) projected points in world frame
        [optional] x_axis, y_axis : (3,), (3,) world-space unit vectors spanning the plane

    Round-trip reconstruction:
        p3d = origin + proj_2d[:, [0]] * x_axis + proj_2d[:, [1]] * y_axis
    """
    P = np.asarray(points, float)
    n = _safe_unit(normal)
    o = np.asarray(origin, float).reshape(3)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")

    # Analytic orthogonal projection to plane
    v = P - o                  # (N,3)
    dist = v @ n               # (N,)
    proj_3d = P - np.outer(dist, n)

    # Build in-plane basis
    if ref_vector is not None:
        rv = np.asarray(ref_vector, float)
        x_axis = rv - np.dot(rv, n) * n   # project ref into plane
        x_axis = _safe_unit(x_axis)
    else:
        x_axis = _choose_perp(n)

    y_axis = _safe_unit(np.cross(n, x_axis))  # ensure right-handed
    # (Optionally re-orthogonalize x to be extra safe)
    x_axis = _safe_unit(np.cross(y_axis, n))

    # 2D coordinates
    rel = proj_3d - o
    u = rel @ x_axis
    v2 = rel @ y_axis
    proj_2d = np.stack([u, v2], axis=1)

    if return_basis:
        return proj_2d, proj_3d, x_axis, y_axis
    
    return proj_2d, proj_3d

# --------- Convenience helpers ---------
def atoms_to_uvd(points, plane_points, ref_vector=None, method="global", return_meta=False):
    """
    Compute (u,v,d) for an arbitrary (N,3) array of points, given 'plane_points' to define the plane.

    Args:
        points        : (N,3) array
        plane_points  : iterable of Atoms or (M,3) array used to fit plane
        ref_vector    : optional (3,) vector to pin +u
        method        : "global" or "local"
        return_meta   : if True, return axes/rotation info

    Returns:
        proj_2d, d    : (N,2), (N,)
        [meta]        : dict with keys depending on method
    """
    n, o, _ = define_plane_from_points(plane_points)
    P = np.asarray(points, float)

    if method == "global":
        out = global_projection(P, n, o, ref_vector=ref_vector, return_R=return_meta)
        if return_meta:
            proj_2d, _, d, R, xw, yw = out
            meta = {"R": R, "x_hat_world": xw, "y_hat_world": yw, "origin": o, "normal": n}
            return proj_2d, d, meta
        proj_2d, _, d = out
        return proj_2d, d
    elif method == "local":
        out = local_projection(P, n, o, ref_vector=ref_vector, return_basis=return_meta)
        if return_meta:
            proj_2d, _, xw, yw = out
            # signed distance for local method:
            d = (P - o) @ _safe_unit(n)
            meta = {"x_hat_world": xw, "y_hat_world": yw, "origin": o, "normal": _safe_unit(n)}
            return proj_2d, d, meta
        proj_2d, _ = out
        d = (P - o) @ _safe_unit(n)
        return proj_2d, d
    else:
        raise ValueError("method must be 'global' or 'local'")


def project_structure_to_plane(structure, plane_points, *, method="global", ref_vector=None, inplace=False, return_meta=False):
    """
    Project every atom of a Biopython Structure to (u,v,d) coordinates defined by a plane.

    Args:
        structure     : Bio.PDB.Structure (assumed pre-aligned as desired)
        plane_points  : iterable of Atoms or (M,3) array used to fit the plane
        method        : "global" (default) or "local"
        ref_vector    : optional (3,) to pin +u
        inplace       : if True, modify structure; else work on a deepcopy
        return_meta   : if True, return (projected_structure, meta_dict)

    Returns:
        projected_structure
        [meta] : dict with plane origin/normal and axes (and R for global)
    """
    n, o, _ = define_plane_from_points(plane_points)

    target = structure if inplace else deepcopy(structure)
    model = target[0]

    # Pull all atom coords in original order
    atoms = [a for a in model.get_atoms()]
    coords = np.array([a.get_coord() for a in atoms], float)

    if method == "global":
        proj_2d, _, d, R, xw, yw = global_projection(coords, n, o, ref_vector=ref_vector, return_R=True)
        meta = {"method": "global", "origin": o, "normal": _safe_unit(n),
                "R": R, "x_hat_world": xw, "y_hat_world": yw}
    else:
        proj_2d, _ = local_projection(coords, n, o, ref_vector=ref_vector, return_basis=False)
        d = (coords - o) @ _safe_unit(n)
        # Recompute axes if meta requested
        if return_meta:
            _, _, xw, yw = local_projection(coords[:2] if len(coords) >= 2 else np.vstack([o, o + _choose_perp(n)]),
                                            n, o, ref_vector=ref_vector, return_basis=True)
        else:
            xw = yw = None
        meta = {"method": "local", "origin": o, "normal": _safe_unit(n),
                "x_hat_world": xw, "y_hat_world": yw}

    # Write (u,v,d) back in the same order
    i = 0
    for chain in model:
        for residue in chain:
            for atom in residue:
                atom.set_coord(np.array([proj_2d[i, 0], proj_2d[i, 1], d[i]], float))
                i += 1

    if return_meta:
        return target, meta
    return target