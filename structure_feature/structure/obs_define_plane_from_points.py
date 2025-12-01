import numpy as np
from typing import Iterable, Optional, Tuple, Union
from Bio.PDB import Entity
from Bio.PDB.Atom import Atom

ArrayLike = Union[np.ndarray, Iterable[Atom], Entity.Entity]

def _as_coords(entity: ArrayLike) -> np.ndarray:
    """
    Normalize input into an (N,3) float array of Cartesian coordinates.

    Accepts:
      - ndarray of shape (N,3)
      - Bio.PDB Entity (Structure/Model/Chain/Residue) → uses all atoms
      - Iterable of Bio.PDB Atom objects (level 'A')
    """
    # Case 1: raw array
    if isinstance(entity, np.ndarray):
        coords = np.asarray(entity, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.size == 0:
            raise ValueError("ndarray must have shape (N,3) with N>0.")
        return coords

    # Case 2: Bio.PDB Entity
    if isinstance(entity, Entity.Entity):
        atoms = list(entity.get_atoms())
        if not atoms:
            raise ValueError("Entity has no atoms.")
        return np.array([a.get_coord() for a in atoms], dtype=float)

    # Case 3: iterable of Atom
    if hasattr(entity, "__iter__"):
        atoms = [a for a in entity if getattr(a, "level", None) == "A"]
        if not atoms:
            raise ValueError("Iterable contains no Atom-level ('A') objects.")
        return np.array([a.get_coord() for a in atoms], dtype=float)

    raise ValueError(
        "Unsupported input. Provide an ndarray (N,3), a Bio.PDB Entity, "
        "or an iterable of Bio.PDB Atom objects."
    )

def _safe_unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Degenerate direction: zero-length vector encountered.")
    return v / n

def _orient_normal(
    normal: np.ndarray,
    origin: np.ndarray,
    toward_point: Optional[Union[np.ndarray, Iterable[float]]] = None,
    opposite_point: Optional[Union[np.ndarray, Iterable[float]]] = None,
) -> np.ndarray:
    """
    Optionally orient the normal:
      - If toward_point is given, flip n so that (toward_point - origin)·n >= 0
      - Else if opposite_point is given, flip n so that (opposite_point - origin)·n <= 0
      - Else return as-is.
    """
    n = np.asarray(normal, float)
    o = np.asarray(origin, float)

    if toward_point is not None:
        tp = np.asarray(toward_point, float)
        if np.dot(tp - o, n) < 0.0:
            n = -n
    elif opposite_point is not None:
        op = np.asarray(opposite_point, float)
        if np.dot(op - o, n) > 0.0:
            n = -n
    return n

def define_plane_from_points(
    entity: ArrayLike,
    *,
    toward_point: Optional[Union[np.ndarray, Iterable[float]]] = None,
    opposite_point: Optional[Union[np.ndarray, Iterable[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a best-fit plane (least squares) to 3D points and return (normal, origin, d)
    such that the plane equation is: normal · x + d = 0.

    Parameters
    ----------
    entity : ndarray | Entity | Iterable[Atom]
        Coordinates or atom collection used to define the plane.
    toward_point : array-like, optional
        If provided, orient the normal so it points toward this point.
    opposite_point : array-like, optional
        If provided (and toward_point is None), orient the normal so it points
        away from this point.

    Returns
    -------
    normal : (3,) ndarray
        Unit normal vector of the fitted plane.
    origin : (3,) ndarray
        Centroid of the input points, a point on the plane.
    d : float
        Scalar offset in the plane equation normal·x + d = 0.

    Notes
    -----
    - Uses SVD on centered coordinates; the right-singular vector associated
      with the smallest singular value is the plane normal.
    - If both `toward_point` and `opposite_point` are None, the normal’s sign is arbitrary.
    """
    coords = _as_coords(entity)
    if coords.shape[0] < 3:
        raise ValueError("At least 3 non-collinear points are required to define a plane.")

    origin = coords.mean(axis=0)
    centered = coords - origin

    # SVD: last row of V^T (vh[-1]) is direction of least variance → plane normal
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _safe_unit(vh[-1, :])

    # Optional orientation
    normal = _orient_normal(normal, origin, toward_point=toward_point, opposite_point=opposite_point)

    d = -float(np.dot(normal, origin))
    return normal, origin, d
