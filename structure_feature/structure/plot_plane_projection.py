import numpy as np
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable, get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import ConvexHull
import matplotlib.patches as Polygon

cmaps = {
    "TRA": "Greens_r",
    "TRB": "Reds_r",
    "Peptide": "Blues_r",
    "MHC helix1": "Greys_r",
    "MHC helix2": "Greys_r",
}

sns.set_theme(style="white", context="paper", font_scale=1.1)

def stats_box_from_agg(agg_dict, title):
    lines = [f"group: {title}"]
    total_pts = sum(v.shape[0] for v in agg_dict.values() if v is not None)
    lines.append(f"ranked_0 structures: {len(set(agg_dict.keys()))}")  # note: this line is optional/inexact
    lines.append(f"total points: {total_pts}")
    return "\n".join(lines)

def plot_atoms_near_plane(
    chain_positions,         # dict[str -> np.ndarray (N,3) of (u,v,d)]
    *,
    pdbid="sample",
    statistics=None,
    z_thresh=1.0,
    color_by="abs",          # "abs" or "signed"
    out_png=None,
    cmaps=cmaps,              # dict[str -> cmap name or matplotlib colormap]
    share_scale=True,        # share color scale across chains (only for color_by="abs")
    norms_override=None,
    kde_keys=("CDR1α","CDR1β","CDR2α","CDR2β","CDR3α","CDR3β"),
    kde=True,
    kde_levels=(0.3, 0.6, 0.9),     # contour levels on seaborn's normalized density
    kde_gridsize=200,               # KDE evaluation grid resolution
    kde_bw_adjust=1.0,              # seaborn bw_adjust (>=1.0 smoother KDE)
    kde_legend=False,
    colorbar_legend=False,
    dpi=300,
    figsize=(8, 6),
    fixed_limits=None,
    show_fig=True,
    ax=None,
    ):
    """
    Seaborn version of your plot:
      • Points: seaborn.scatterplot with continuous colormaps per chain.
      • KDE: seaborn.kdeplot with filled contour bands for CDR loops.
      • Peptide/MHC helices: convex hull patches + labels.
      • Per-chain colorbars using inset axes.

    Notes
    -----
    - `kde_levels` act on seaborn's normalized density (0–1).
    - `kde_gridsize` controls KDE smoothness/resolution (larger = smoother/slower).
    - `kde_bw_adjust` > 1.0 = smoother, < 1.0 = sharper.
    """

    # ----------------- Info box text -----------------
    info = None
    if statistics:
        stats = {
            "killing": statistics.get("property", None),
            "allele": statistics.get("allele", None),
            "peptide": statistics.get("peptide", None),
            "conf": statistics.get("ranking_confidence", None),
            "IpLDDT": statistics.get("IpLDDT", None),
            "r": statistics.get("binding_geometry_r", None),
            "theta": statistics.get("binding_geometry_theta", None),
            "phi": statistics.get("binding_geometry_phi", None),
            "docking": statistics.get("docking_angle", None),
            "incident": statistics.get("incident_angle", None),
            "interdomain": statistics.get("interdomain_angle", None),
            "cdr1a_iplddt": statistics.get("cdr1a_plddt", None),
            "cdr1b_iplddt": statistics.get("cdr1b_plddt", None),
            "cdr2a_iplddt": statistics.get("cdr2a_plddt", None),
            "cdr2b_iplddt": statistics.get("cdr2b_plddt", None),
            "cdr3a_iplddt": statistics.get("cdr3a_plddt", None),
            "cdr3b_iplddt": statistics.get("cdr3b_plddt", None),
        }
        if "center_shift" in statistics and statistics["center_shift"] is not None:
            shift = statistics["center_shift"]
            if isinstance(shift, (list, tuple)) and len(shift) >= 2:
                stats["shift_x"], stats["shift_y"] = shift[0], shift[1]

        lines = []
        for k, v in stats.items():
            if v is None:
                continue
            if isinstance(v, (float, int)):
                lines.append(f"{k}: {v:.0f}")
            else:
                lines.append(f"{k}: {v}")
        info = "\n".join(lines)

    # ----------------- Default colormaps -----------------
    # Normalize user-provided cmaps: allow strings like "Greens_r"
    if cmaps is None:
        cmaps = {}
    else:
        cmaps = {
            k: (sns.color_palette(v, as_cmap=True) if isinstance(v, str) else v)
            for k, v in cmaps.items()
        }

    for cid in chain_positions:
        if cid not in cmaps:
            if "a" in cid.lower():
                cmaps[cid] = sns.color_palette("crest", as_cmap=True)
            elif "b" in cid.lower():
                cmaps[cid] = sns.color_palette("mako", as_cmap=True)
            else:
                cmaps[cid] = sns.color_palette("viridis", as_cmap=True)


    # Fixed styles for special groups (drawn as hulls later)
    fixed_groups = {"Peptide", "MHC helix1", "MHC helix2"}

    # ----------------- Filter & organize data -----------------
    data = {}
    for name, arr in chain_positions.items():
        if arr is None:
            continue
        pts = np.asarray(arr)
        if name not in fixed_groups:
            mask = np.abs(pts[:, 2]) < z_thresh
            pts = pts[mask]
        if pts.size == 0:
            continue
        data[name] = {"u": pts[:, 0], "v": pts[:, 1], "d": pts[:, 2]}

    if not data:
        print(f"[warn] No atoms within |d| < {z_thresh} Å to plot.")
        return

    # CDR-only data for KDE overlays (use all points; let seaborn handle grids)
    cdr_data = {}
    if kde:
        for key in kde_keys:
            arr = chain_positions.get(key)
            if arr is None:
                continue
            pts = np.asarray(getattr(arr, "positions", arr), float).reshape(-1, 3)
            if pts.size >= 10:  # require a minimum for a stable KDE
                cdr_data[key] = {"u": pts[:, 0], "v": pts[:, 1]}

    # ----------------- Color normalization -----------------
    if color_by == "abs":
        vals = {cid: np.abs(d["d"]) for cid, d in data.items()}
        if share_scale:
            all_vals = np.concatenate([v for v in vals.values() if v.size])
            shared = Normalize(vmin=all_vals.min(), vmax=all_vals.max())
            norms = {cid: shared for cid in data}
        else:
            norms = {cid: Normalize(vmin=v.min(), vmax=v.max()) for cid, v in vals.items()}
        def val_getter(dvals):
            return np.abs(dvals)
    elif color_by == "signed":
        norms = {cid: Normalize(vmin=d["d"].min(), vmax=d["d"].max()) for cid, d in data.items()}
        def val_getter(dvals):
            return dvals
    else:
        raise ValueError('color_by must be "abs" or "signed"')

    # allow caller to override per-chain Normalize objects
    if isinstance(locals().get("norms_override", None), dict):
        for k, n in norms_override.items():
            if k in norms:
                norms[k] = n

    # ----------------- Figure / axis -----------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # ----------------- Scatter by chain (seaborn) -----------------
    # Keep a handle for colorbars (per chain).
    mappables = {}

    for name, d in data.items():
        if name in fixed_groups or name in kde_keys:
            continue  # drawn as hulls below

        u, v, dvals = d["u"], d["v"], d["d"]
        if u.size == 0:
            continue

        cmap = cmaps.get(name, sns.color_palette("viridis", as_cmap=True))
        norm = norms[name]
        colors = cmap(norm(val_getter(dvals)))

        # seaborn scatterplot for styling (legend=False, we add explicit labels elsewhere)
        sns.scatterplot(
            x=u, y=v,
            s=10, linewidth=0, alpha=0.65,
            hue=None, palette=None,
            color=None, edgecolor=None,
            ax=ax, legend=False,
            zorder=2
        )

        # Matplotlib re-coloring (so we can attach a proper ScalarMappable & colorbar)
        coll = ax.collections[-1]  # the scatter we just drew
        coll.set_facecolor(colors)

        mappables[name] = ScalarMappable(norm=norm, cmap=cmap)

    # ----------------- Peptide / MHC convex hull patches -----------------
    if "Peptide" in data:
        u, v = data["Peptide"]["u"], data["Peptide"]["v"]
        points_2d = np.column_stack((u, v))
        if points_2d.shape[0] < 3:
            return
        hull = ConvexHull(points_2d)
        verts = points_2d[hull.vertices]
        poly = plt.Polygon(
            verts, closed=True, facecolor="blue", edgecolor="none", alpha=0.4, zorder=3
        )
        ax.add_patch(poly)
        centroid = verts.mean(axis=0)
        ax.text(
            # centroid[0] - 3, centroid[1] - 12.5,
            centroid[0] - 3, -10,
            "Peptide\nN-terminus", color="black", fontsize=8, 
            # fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="blue", alpha=0.4),
            zorder=3+1
        )

    if "MHC helix1" in data:
        u, v = data["MHC helix1"]["u"], data["MHC helix1"]["v"]
        # _hull_patch((u, v), face="lightgrey", label="MHC Helix 1", alpha=0.4, z=1)
        points_2d = np.column_stack((u, v))
        if points_2d.shape[0] < 3:
            return
        hull = ConvexHull(points_2d)
        verts = points_2d[hull.vertices]
        poly = plt.Polygon(
            verts, closed=True, facecolor="grey", edgecolor="none", alpha=0.4, zorder=1
        )
        ax.add_patch(poly)
        centroid = verts.mean(axis=0)
        ax.text(
            # centroid[0] - 6, centroid[1] - 20,
            centroid[0] - 5, -20,
            "HLA α1\nC-terminus", color="black", fontsize=8, 
            # fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="none", edgecolor="none", alpha=0.4),
            zorder=1+1
        )

    if "MHC helix2" in data:
        u, v = data["MHC helix2"]["u"], data["MHC helix2"]["v"]
        # _hull_patch((u, v), face="lightgrey", label="MHC Helix 2", alpha=0.4, z=1)
        points_2d = np.column_stack((u, v))
        if points_2d.shape[0] < 3:
            return
        hull = ConvexHull(points_2d)
        verts = points_2d[hull.vertices]
        poly = plt.Polygon(
            verts, closed=True, facecolor="grey", edgecolor="none", alpha=0.4, zorder=1
        )
        ax.add_patch(poly)
        centroid = verts.mean(axis=0)
        ax.text(
            # centroid[0] + 4, centroid[1] - 20,
            centroid[0] + 5, -20,
            "HLA α2\nN-terminus", color="black", fontsize=8, 
            # fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="none", edgecolor="none", alpha=0.4),
            zorder=1+1
        )

    # ----------------- KDE overlays for CDR loops (seaborn.kdeplot) -----------------
    if kde and cdr_data:
        # color set for KDE bands (distinct from scatter colormaps)
        kde_palette = {
            "CDR1α": "C0", "CDR2α": "C4", "CDR3α": "C9",
            "CDR1β": "C1", "CDR2β": "C8", "CDR3β": "C6",
        }
        for key, dv in cdr_data.items():
            # Use seaborn's normalized density; draw filled bands.
            sns.kdeplot(
                x=dv["u"], y=dv["v"],
                ax=ax,
                fill=True,
                levels=sorted(kde_levels),
                thresh=kde_levels[0] if len(kde_levels) else 0.0,
                # bw_adjust=kde_bw_adjust,
                # gridsize=kde_gridsize,
                color=kde_palette.get(key, "0.4"),
                alpha=0.6,
                zorder=3.5
            )
        # --- manual legend for CDR overlays ---
        if kde_legend is True:
            legend_elements = [
                Patch(facecolor=col, edgecolor='none', alpha=0.8, label=key)
                for key, col in kde_palette.items()
            ]
            ax.legend(
                handles=legend_elements,
                title="CDR regions",
                loc="upper right",
                bbox_to_anchor=(1.1, 1.1),
                frameon=False,
                fontsize=9,
                title_fontsize=10
            )

    # ----------------- Axes cosmetics -----------------
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u (plane X′)")
    ax.set_ylabel("v (plane Y′)")
    ax.set_title(f"{pdbid}", pad=6)
    if fixed_limits is not None:
        ax.set_xlim(fixed_limits[0])
        ax.set_ylim(fixed_limits[1])

    sns.despine(ax=ax, trim=True)
    ax.grid(False)

    # ----------------- Compact per-chain colorbars -----------------
    if (mappables) and (colorbar_legend is True):
        divider = make_axes_locatable(ax)

        _cbar_height = 0.80   # 80% of original height
        _cbar_shift  = 0.10   # shift up by 10% of original height

        pads = np.linspace(0.06, 0.26, num=len(mappables))
        for (cid, sm), pad in zip(mappables.items(), pads):
            cax = divider.append_axes("right", size="2.5%", pad=float(pad))
            cb = plt.colorbar(sm, cax=cax)
            cb.ax.set_title(f"{cid}", pad=4, fontsize=9)
            # cb.ax.set_xticks([0,5,10,15])
            # cb.ax.set_xticklabels([0,5,10,15])
            cb.ax.tick_params(labelsize=7)
            # cb.set_label('Distance to the\nbinding groove plane', size=10)
            pos = cb.ax.get_position()
            cb.ax.set_position([
                pos.x0,
                pos.y0 + _cbar_shift * pos.height,
                pos.width,
                _cbar_height * pos.height
            ])
            # cb.set_label('Distance to the binding plane', size=10)
        

    # ----------------- Side info box -----------------
    if info:
        fig.text(
            0.02, 0.5, info,
            ha="left", va="center",
            fontsize=8.5, family="monospace", color="black",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF7CC", edgecolor="#333333", alpha=0.9)
        )

    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        print(f"[ok] Saved: {out_png}")

    if show_fig:
        fig.show()
    else:
        plt.close(fig)

    return fig, ax, mappables

def make_shared_norms(chain_sets, *, z_thresh=20, color_by="signed",
                      ignore=set(("Peptide","MHC helix1","MHC helix2")),
                      skip=set(("CDR1α","CDR1β","CDR2α","CDR2β","CDR3α","CDR3β"))):
    vals = defaultdict(list)
    for chain_positions in chain_sets:
        for name, arr in chain_positions.items():
            if name in ignore:
                continue
            pts = np.asarray(arr)
            if pts.size == 0:
                continue
            # apply same z-threshold logic as the plotter
            if name not in ignore:
                pts = pts[np.abs(pts[:,2]) < z_thresh]
            if pts.size == 0:
                continue
            d = pts[:,2]
            if color_by == "abs":
                vals[name].append(np.abs(d))
            else:
                vals[name].append(d)

    norms = {}
    for name, chunks in vals.items():
        allv = np.concatenate(chunks) if chunks else np.array([])
        if allv.size:
            vmin, vmax = float(allv.min()), float(allv.max())
            # guard against degenerate ranges
            if vmin == vmax:
                vmin -= 1e-6; vmax += 1e-6  # noqa: E702
            norms[name] = Normalize(vmin=vmin, vmax=vmax)
    return norms