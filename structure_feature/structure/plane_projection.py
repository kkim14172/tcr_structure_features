import numpy as np
from Bio.PDB import Entity

# ---------- Plane fit ----------
def define_plane_from_points(entity):
    """
    Fit a plane to a set of 3D points.
    Accepts:
      - Bio.PDB Structure/Model/Chain/Residue (Entity) → uses all atoms
      - Iterable of Bio.PDB Atom objects
      - ndarray of shape (N,3)

    Returns:
        normal : (3,) unit vector normal to the plane
        centroid : (3,) centroid of the points (a point on the plane)
        d      : scalar offset (plane equation: n·x + d = 0)
    """
    # Case 1: raw coordinates
    if isinstance(entity, np.ndarray):
        coords = np.asarray(entity, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("If passing ndarray, it must have shape (N,3).")
    # Case 2: Bio.PDB Entity (Structure/Model/Chain/Residue)
    elif isinstance(entity, Entity.Entity):
        atoms = list(entity.get_atoms())
        if not atoms:
            raise ValueError("Entity has no atoms.")
        coords = np.array([a.get_coord() for a in atoms], dtype=float)
    # Case 3: iterable of Atom objects
    elif hasattr(entity, "__iter__"):
        atoms = [a for a in entity if getattr(a, "level", None) == "A"]
        if not atoms:
            raise ValueError("Iterable provided, but contains no Atom ('A') level objects.")
        coords = np.array([a.get_coord() for a in atoms], dtype=float)
    else:
        raise ValueError(
            "define_plane_from_points expects an ndarray (N,3), a Bio.PDB Entity, "
            "or an iterable of Atom objects."
        )

    centroid = coords.mean(axis=0)
    centered = coords - centroid
    # SVD: the last right-singular vector is the plane normal
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1, :]
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))
    return normal, centroid, d

# ---------- Orientation + frame ----------

def orient_normal_toward(normal, plane_points, toward_point=None, opposite_point=None, 
                         tol: float = 0.0, return_flipped: bool = False,):
    """
    Orient a plane normal relative to a reference point.

    Parameters
    ----------
    normal : (3,) array-like
        Plane normal (need not be unit).
    plane_points : (3,) array-like
        A point on the plane (e.g., plane centroid).
    toward_point : (3,) array-like or None
        If provided and `opposite_point` is None, ensure the returned normal points
        from `plane_point` TOWARD this point (i.e., dot(n, toward - plane_point) >= tol).
    opposite_point : (3,) array-like or None
        If provided, ensure the returned normal points AWAY FROM this point
        (i.e., dot(n, opposite - plane_point) <= -tol).
        If both `toward_point` and `opposite_point` are provided, `opposite_point` takes precedence.
    tol : float
        Tolerance around zero for the dot-product test. Use a small value like 1e-12 if needed.
    return_flipped : bool
        If True, also return a boolean indicating whether the normal was flipped.

    Returns
    -------
    n : (3,) ndarray
        Unit-length oriented normal.
    flipped : bool (optional)
        Whether the normal was flipped. Returned only if `return_flipped=True`.

    Raises
    ------
    ValueError
        If no reference point is provided, or if `normal` is zero-length.
    """
    n = np.asarray(normal, dtype=float)
    if not np.isfinite(n).all() or np.linalg.norm(n) == 0:
        raise ValueError("`normal` must be a finite, non-zero 3-vector.")
    n /= np.linalg.norm(n)

    p0 = np.asarray(plane_points, dtype=float)
    if not np.isfinite(p0).all() or p0.shape != (3,):
        raise ValueError("`plane_point` must be a finite 3-vector.")

    use_opposite = opposite_point is not None
    use_toward = toward_point is not None

    if not (use_opposite or use_toward):
        raise ValueError("Provide at least one of `toward_point` or `opposite_point`.")

    if use_opposite:
        q = np.asarray(opposite_point, dtype=float)
        mode = "away"
    else:
        q = np.asarray(toward_point, dtype=float)
        mode = "toward"

    if not np.isfinite(q).all() or q.shape != (3,):
        raise ValueError("Reference point must be a finite 3-vector.")

    v = q - p0
    v_norm = np.linalg.norm(v)
    flipped = False

    # If reference equals plane point, we cannot determine direction; leave n as-is.
    if v_norm == 0:
        return (n, flipped) if return_flipped else n

    s = float(np.dot(n, v))

    if mode == "away":
        # want dot(n, v) <= -tol; if too positive, flip
        if s > tol:
            n = -n
            flipped = True
    else:  # mode == "toward"
        # want dot(n, v) >= tol; if too negative, flip
        if s < -tol:
            n = -n
            flipped = True

    return (n, flipped) if return_flipped else n


def plane_coordinate_frame(normal):
    """Return a right-handed (u, v, n) orthonormal basis from a (unit or non-unit) normal."""
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    # choose a reference not almost parallel to n
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)  # ensures u, v ⟂ n
    v /= np.linalg.norm(v)
    # Ensure right-handedness: u x v should be +n
    if np.dot(np.cross(u, v), n) < 0:
        v = -v
    return u, v, n

# ---------- Transforms (vectorized) ----------
def world_to_plane(x, R, origin):
    """
    Transform points from world coords to local plane coords.
    Supports shape (3,) or (N,3).
    """
    X = np.asarray(x, float)
    return (X - origin) @ R  # (N,3)@(3,3)->(N,3) or (3,)@(3,3)->(3,)

def plane_to_world(xp, R, origin):
    """
    Transform points from local plane coords to world coords.
    Supports shape (3,) or (N,3).
    """
    XP = np.asarray(xp, float)
    return XP @ R.T + origin

# ---------- Global projection ----------
def geometric_projection(points, normal, centroid, toward_point=None, return_signed=True):
    """
    Orthogonal (global) projection of point(s) onto the plane defined by (normal, centroid).
    If `toward_point` is provided, orient normal toward it (fixes signed distance convention).
    """
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    if toward_point is not None:
        n = orient_normal_toward(n, centroid, toward_point)

    X = np.asarray(points, float)
    if X.shape[-1] != 3:
        raise ValueError("`points` must have shape (..., 3).")

    delta = np.tensordot(X - centroid, n, axes=([-1], [0]))  # signed distances (...,)
    X_proj = X - np.expand_dims(delta, -1) * n
    return (X_proj, delta) if return_signed else X_proj

# ---------- Plane object ----------
class Plane:
    def __init__(self, plane_points, toward_point=None, opposite_point=None, normal=None, centroid=None, R=None):
        
        if any(x is None for x in (normal, centroid)):
            normal, centroid, d = define_plane_from_points(plane_points)
        
        self.centroid = centroid
        # self.d = d
        if (toward_point is not None) or (opposite_point is not None):
            self.normal = orient_normal_toward(normal, centroid, toward_point, opposite_point)
        else:
            self.normal = normal

        if R is None:
            u, v, n = plane_coordinate_frame(self.normal)
            self.R = np.column_stack([u, v, n])  # columns are basis vectors
        else:
            self.R = R

    def coordinate_transform(self, points, R=None, centroid=None):
        """Return local (u, v, z') coords for the input points."""
        R = self.R if R is None else R
        centroid = self.centroid if centroid is None else centroid
        return world_to_plane(points, R, centroid)

    def geometric_projection(self, points):
        """Return (projected_points_world, signed_distances)."""
        n = self.normal / np.linalg.norm(self.normal)
        X = np.asarray(points, float)
        delta = np.tensordot(X - self.centroid, n, axes=([-1], [0]))
        X_proj = X - np.expand_dims(delta, -1) * n
        return X_proj, delta
