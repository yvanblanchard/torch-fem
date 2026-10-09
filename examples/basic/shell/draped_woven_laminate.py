"""Shear-angle effect on cured woven composites - torch-fem Laminate.

Reproduces the approach of Aridhi et al., "Textile composite structural analysis
taking into account the forming process", Composites Part B 166 (2019) 773-784
(hal-02399005), with the `torchfem.fabric` module:

1. Cured ply properties versus in-plane shear angle (bisector frame), with the
   paper's model: tension-only warp and weft yarns + isotropic matrix, i.e.
   two equivalent UD plies (dual UD-ply model of the biaxial layer).
2. Paper Sec. 5.1: bias-extension test (70 x 210 mm, d = 50 mm, gamma_A = 57 deg).
   a) Forming simulation (large strain membrane, torch-fem Planar) with the
      paper's forming law: yarn tension + non-linear shear modulus G12(gamma)
      (Eq. 22); shear angle vs displacement compared with Eq. 21 (cf. Fig. 9).
   b) Tensile test on the cured, deformed specimen with and without fibre
      reorientation, using either the analytical 3-zone field or the yarn
      field predicted by the forming simulation.
3. Paper Sec. 5.2-like: draped hemisphere (KinDrape kinematic drape instead of
   the paper's Abaqus stamping), in-plane displacement on one edge, opposite
   edge clamped; reaction force and stresses with and without reorientation.

Run::

    python examples/basic/shell/draped_woven_laminate.py --kindrape ../KinDrapeApp
    # or with saved KinDrape results:
    python examples/basic/shell/draped_woven_laminate.py --drapes ply.npz

Units: N, mm, MPa.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from torchfem import Shell  # noqa: E402
from torchfem.fabric import (  # noqa: E402
    G12_GLASS_PP,
    BiaxialPly,
    DrapedPly,
    bias_extension_forming,
    bias_extension_kinematics,
    bias_extension_shear,
    build_draped_laminate,
    direction_angles,
    element_frames,
    grid_to_shell_mesh,
    map_directions,
    nominal_directions,
    shear_modulus,
    shear_stress,
    yarn_stresses,
)
from torchfem.mesh import rect_tri  # noqa: E402

torch.set_default_dtype(torch.float64)

# Categorical colours (validated reference palette, first 3 slots) and ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Paper Sec. 5.2: commingled glass/PP plain weave, E_11 = E_22 = 35400 MPa,
# thickness 1.2 mm. Polypropylene matrix values are assumed (not in the paper).
GLASS_PP = BiaxialPly.aridhi(E_1=35400.0, E_2=35400.0, E_m=1500.0, nu_m=0.40, t=1.2)

# Paper Sec. 5.1: 8-harness satin glass/PA66 (Solvay). Constants are NOT given
# in the paper -> representative values (assumed).
GLASS_PA66 = BiaxialPly.aridhi(E_1=21000.0, E_2=21000.0, E_m=3000.0, nu_m=0.35, t=2.0)


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


# ---------------------------------------------------------------------------
# 1. Cured ply properties versus shear angle
# ---------------------------------------------------------------------------
def study_properties(out: Path, ply: BiaxialPly) -> None:
    g_deg = torch.linspace(0.0, 60.0, 121)
    c = ply.bisector_constants(torch.deg2rad(g_deg))
    print("  gamma     E_x      E_y     G_xy   nu_xy   (bisector frame, MPa)")
    for gd in (0, 10, 20, 30, 40, 50, 60):
        i = int(gd * 2)
        print(
            f"  {gd:5d} {c['E_x'][i]:8.0f} {c['E_y'][i]:8.0f} "
            f"{c['G_xy'][i]:8.0f}  {c['nu_xy'][i]:6.3f}"
        )

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    g = g_deg.numpy()
    ax[0].plot(g, c["E_x"] / 1e3, color=C1, lw=2)
    ax[0].plot(g, c["E_y"] / 1e3, color=C2, lw=2)
    ax[0].text(g[-1], c["E_x"][-1] / 1e3, " E_x (bisector)", color=INK, va="center",
               fontsize=9)
    ax[0].text(g[-1], c["E_y"][-1] / 1e3 + 1.5, " E_y", color=INK, va="bottom", fontsize=9)
    ax[0].set_ylabel("Young's modulus [GPa]")
    ax[0].set_title("Moduli in the bisector frame", color=INK, loc="left")
    ax[1].plot(g, c["G_xy"] / 1e3, color=C1, lw=2)
    ax[1].set_ylabel("G_xy [GPa]")
    ax[1].set_title("In-plane shear modulus", color=INK, loc="left")
    ax[2].plot(g, c["nu_xy"], color=C1, lw=2)
    ax[2].set_ylabel("ν_xy [-]")
    ax[2].set_title("Poisson's ratio", color=INK, loc="left")
    for a in ax:
        a.set_xlabel("Shear angle γ [deg]")
        a.set_xlim(0, 72)
        _style(a)
    fig.suptitle("Cured glass/PP plain weave (Aridhi model): effect of the forming shear",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "1a_properties_vs_shear.png", dpi=150)
    plt.close(fig)

    # Polar plot of the tension modulus, angle measured from the bisector
    phi = torch.linspace(0, 2 * math.pi, 361)
    gammas = [0.0, 30.0, 50.0]
    E = ply.modulus(torch.deg2rad(torch.tensor(gammas)), phi) / 1e3
    fig = plt.figure(figsize=(5.6, 5.2))
    ax = fig.add_subplot(projection="polar")
    for k, (gd, col) in enumerate(zip(gammas, (C1, C2, C3))):
        ax.plot(phi, E[k], color=col, lw=2, label=f"γ = {gd:.0f}°")
    ax.set_title("Tension modulus E(φ) [GPa]\nφ from the yarn bisector", color=INK,
                 fontsize=10)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    ax.tick_params(colors=MUTED)
    ax.set_rlabel_position(100)
    fig.tight_layout()
    fig.savefig(out / "1b_polar_modulus.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Bias-extension test then tension of the cured specimen (paper Sec. 5.1)
# ---------------------------------------------------------------------------
def study_bias_forming(out: Path) -> dict:
    """Forming step: bias-extension test with the non-linear G12(gamma) law."""
    W, L, d = 70.0, 210.0, 50.0
    r = bias_extension_forming(W, L, d, E_yarn=35400.0, G12_coeffs=G12_GLASS_PP,
                               thickness=1.2, n_w=14, n_inc=50)
    cen = r["nodes"][r["elements"]].mean(1)
    core = ((cen[:, 1] - L / 2).abs() < 15) & ((cen[:, 0] - W / 2).abs() < 8)
    g_fe = torch.rad2deg(r["gamma"][:, core].median(dim=1).values)
    dd = r["d"]
    g_an = torch.tensor([math.degrees(bias_extension_shear(W, L, x)) for x in dd.tolist()])
    print(f"  {len(r['elements'])} yarn-aligned Quad1, max yarn strain "
          f"{(r['stretch'] - 1).abs().max():.1e}")
    for k in range(0, len(dd), 10):
        print(f"  d = {dd[k]:4.1f} mm   gamma_A FE = {g_fe[k]:5.1f} deg   Eq.21 = "
              f"{g_an[k]:5.1f} deg   force = {r['force'][k]:7.2f} N")

    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4),
                           gridspec_kw={"width_ratios": [1, 1, 1, 0.55]})
    gg = torch.linspace(0, 1.05, 100)
    ax[0].plot(torch.rad2deg(gg), shear_modulus(gg), color=C1, lw=2, label="G₁₂(γ), tangent")
    ax[0].plot(torch.rad2deg(gg), shear_stress(gg), color=C2, lw=2, label="τ(γ) = ∫G₁₂ dγ")
    ax[0].set_xlabel("Shear angle γ [deg]")
    ax[0].set_ylabel("MPa")
    ax[0].set_title("Paper Eq. 22 (glass/PP)", color=INK, loc="left")
    ax[0].legend(frameon=False)
    ax[1].plot(dd, g_an, color=MUTED, lw=1.5, ls="--", label="Analytical (Eq. 21)")
    ax[1].plot(dd, g_fe, "o", color=C1, ms=4, label="torch-fem, zone A centre")
    ax[1].set_xlabel("Machine displacement d [mm]")
    ax[1].set_ylabel("Shear angle [deg]")
    ax[1].set_title("Shear angle vs displacement (cf. Fig. 9)", color=INK, loc="left")
    ax[1].legend(frameon=False)
    ax[2].plot(dd, r["force"], color=C1, lw=2)
    ax[2].set_xlabel("Machine displacement d [mm]")
    ax[2].set_ylabel("Clamp force [N]")
    ax[2].set_title("Forming load (t = 1.2 mm)", color=INK, loc="left")
    for a in ax[:3]:
        _style(a)
    x = (r["nodes"] + r["u"][-1]).numpy()
    q = r["elements"]
    tri = torch.cat([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).numpy()
    gl = torch.rad2deg(r["gamma"][-1].abs())
    tp = ax[3].tripcolor(x[:, 0], x[:, 1], tri, facecolors=torch.cat([gl, gl]).numpy(),
                         cmap="Blues", edgecolors="none")
    ax[3].set_aspect("equal")
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    ax[3].set_title(f"γ [deg], d = {d:g} mm", color=INK, loc="left", fontsize=10)
    fig.colorbar(tp, ax=ax[3], shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "2a_bias_forming_G12.png", dpi=150)
    plt.close(fig)
    return r


def _cured_tension(nodes, elements, plies, bottom, top, du=1.0, tmode="none"):
    lam, _ = build_draped_laminate(plies, thickness_mode=tmode)
    m = Shell(nodes, elements, lam)
    m.constraints[bottom] = True
    m.constraints[top] = True
    disp = torch.zeros_like(m.displacements)
    disp[top, 1] = du
    m.displacements = disp
    _, f, _, _, _ = m.solve()
    return f[top, 1].sum().item() / du


def study_bias_extension(out: Path, ply: BiaxialPly, forming: dict | None = None) -> None:
    """Tension of the cured specimen after the bias-extension test."""
    W, L, d = 70.0, 210.0, 50.0
    r2 = 1 / math.sqrt(2)
    K = {}

    # (a) Analytical three-zone kinematics (Eq. 21), deformed geometry
    X, elements = rect_tri(36, 106, W, L, variant="zigzag")
    kin_n = bias_extension_kinematics(X, W, L, d, n_path=800)
    kin_e = bias_extension_kinematics(X[elements].mean(1), W, L, d)
    nodes = torch.hstack([kin_n["x"], torch.zeros(len(X), 1)])
    n_elem = len(elements)
    z = torch.zeros(n_elem, 1)
    th1 = direction_angles(nodes, elements, torch.hstack([kin_e["f1"], z]))
    th2 = direction_angles(nodes, elements, torch.hstack([kin_e["f2"], z]))
    n1 = direction_angles(nodes, elements, torch.tensor([r2, r2, 0.0]).expand(n_elem, 3))
    n2 = direction_angles(nodes, elements, torch.tensor([-r2, r2, 0.0]).expand(n_elem, 3))
    bottom = X[:, 1] < 1e-9
    top = X[:, 1] > L - 1e-9
    K["with reorientation (Eq. 21 zones)"] = _cured_tension(
        nodes, elements, [DrapedPly(ply, th1, th2)], bottom, top)
    K["without (orthogonal ±45°)"] = _cured_tension(
        nodes, elements, [DrapedPly(ply, n1, n2)], bottom, top)
    k_areal = _cured_tension(nodes, elements, [DrapedPly(ply, th1, th2)], bottom, top,
                             tmode="areal")

    # (b) Yarn field and geometry predicted by the forming simulation
    if forming is not None:
        q = forming["elements"]
        tri = torch.cat([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        xf = forming["nodes"] + forming["u"][-1]
        fn = torch.hstack([xf, torch.zeros(len(xf), 1)])
        zf = torch.zeros(len(tri), 1)
        f1 = torch.cat([forming["f1"][-1]] * 2)
        f2 = torch.cat([forming["f2"][-1]] * 2)
        a1 = direction_angles(fn, tri, torch.hstack([f1, zf]))
        a2 = direction_angles(fn, tri, torch.hstack([f2, zf]))
        K["with reorientation (G₁₂(γ) forming FE)"] = _cured_tension(
            fn, tri, [DrapedPly(ply, a1, a2)], forming["bottom"], forming["top"])

    for name, k in K.items():
        print(f"  {name:40s} K = {k / 1e3:6.3f} kN/mm")
    print(f"  {'with reorientation + t0/cos γ':40s} K = {k_areal / 1e3:6.3f} kN/mm")
    keys = list(K)
    print(f"  stiffness ratio with/without = {K[keys[0]] / K[keys[1]]:.1f} "
          "(paper Fig. 10: ~2.4 / ~0.5 kN/mm, experiment ~2.6 kN/mm)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    x = nodes[:, :2].numpy()
    tp = ax[0].tripcolor(x[:, 0], x[:, 1], elements.numpy(),
                         facecolors=torch.rad2deg(kin_e["gamma"]).numpy(),
                         cmap="Blues", edgecolors="none")
    sel = torch.arange(0, n_elem, 151)
    c = nodes[elements[sel]].mean(1)
    for f, col in ((kin_e["f1"], C2), (kin_e["f2"], C3)):
        ax[0].quiver(c[:, 0], c[:, 1], f[sel, 0], f[sel, 1], color=col, angles="xy",
                     scale_units="xy", scale=0.12, width=0.008, headwidth=0,
                     headlength=0, headaxislength=0, pivot="middle")
    ax[0].set_aspect("equal")
    ax[0].set_title("Cured bias specimen, γ [deg]\nwarp / weft directions",
                    color=INK, loc="left")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    fig.colorbar(tp, ax=ax[0], shrink=0.7)

    u = np.linspace(0, 3.2, 2)
    order = [k for k in keys if "without" not in k] + [keys[1]]
    for name, col, ls in zip(order, (C1, C3, C2), ("-", (0, (4, 3)), "-")):
        ax[1].plot(u, K[name] * u / 1e3, color=col, lw=2, ls=ls, label=name)
    ax[1].plot([0, 1.5], [0, 4.0], ls="--", color=MUTED, lw=1.2)
    ax[1].text(1.55, 4.0, "paper exp., linear part (~2.6 kN/mm)", color=MUTED,
               fontsize=8, va="center")
    ax[1].set_xlabel("Displacement [mm]")
    ax[1].set_ylabel("Load [kN]")
    ax[1].set_title(f"Tension of the cured specimen (glass/PA66 assumed, t = {ply.t} mm)",
                    color=INK, loc="left", fontsize=10)
    ax[1].legend(frameon=False)
    _style(ax[1])
    fig.tight_layout()
    fig.savefig(out / "2_bias_extension_tension.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Draped hemisphere, in-plane loading of the cured part (paper Sec. 5.2)
# ---------------------------------------------------------------------------
def run_kindrape(kindrape_dir: Path, ang: float, d: float, grid: int) -> np.ndarray:
    spec = importlib.util.spec_from_file_location(
        "KinDrape_eff_NR", kindrape_dir / "KinDrape_eff_NR.py"
    )
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)
    org = [grid // 2 - 1, grid // 2 - 1]
    node, _, _, _ = kd.KinDrape_eff_NR(d, [grid, grid], [0.0, 0.0], ang, org, 0.0,
                                       False, "hemisphere")
    return node


def plot_field(ax, fig, nodes, elements, values, title, cmap="Blues", label="",
               vlim=None):
    x = nodes.numpy()
    v = values.numpy()
    pc = Poly3DCollection(x[elements.numpy()], cmap=cmap, edgecolor="none")
    pc.set_array(v)
    pc.set_clim(*(vlim if vlim is not None else (v.min(), v.max())))
    ax.add_collection3d(pc)
    for i, lim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        lim(x[:, i].min(), x[:, i].max())
    ax.set_box_aspect(np.ptp(x, axis=0))
    ax.view_init(35, -60)
    ax.set_axis_off()
    ax.set_title(title, color=INK, fontsize=10)
    fig.colorbar(pc, ax=ax, shrink=0.6, label=label)


def study_dome(out: Path, grids, angles, ply: BiaxialPly, scale: float, u2: float):
    ref = grid_to_shell_mesh(torch.as_tensor(grids[0]) * scale)
    nodes, elements, ij = ref["nodes"], ref["elements"], ref["ij"]
    cen = nodes[elements].mean(1)
    print(f"  shell mesh: {len(nodes)} nodes, {len(elements)} Tria1, R = {scale:g} mm")

    draped, nominal = [], []
    for k, g in enumerate(grids):
        src = ref if k == 0 else grid_to_shell_mesh(torch.as_tensor(g) * scale)
        src_cen = src["nodes"][src["elements"]].mean(1)
        th1 = direction_angles(nodes, elements, map_directions(src_cen, src["warp"], cen))
        th2 = direction_angles(nodes, elements, map_directions(src_cen, src["weft"], cen))
        draped.append(DrapedPly(ply, th1, th2))
        a = math.radians(angles[k])
        n1, n2 = nominal_directions(nodes, elements,
                                    torch.tensor([math.cos(a), math.sin(a), 0.0]))
        nominal.append(DrapedPly(ply, n1, n2))
        gd = torch.rad2deg(draped[-1].gamma.abs())
        print(f"  ply {k} ({angles[k]:g} deg): |gamma| mean {gd.mean():.1f} deg, "
              f"max {gd.max():.1f} deg")

    # Paper Fig. 15: lower boundary clamped, displacement in direction 2 on the
    # opposite boundary (grid rows j = 0 and j = max of the draped patch).
    clamped = ij[:, 1] == 0
    loaded = ij[:, 1] == int(ij[:, 1].max())
    frames = element_frames(nodes, elements)[:, :2, :]  # rows e1, e2

    res = {}
    for name, plies in (("without reorientation", nominal), ("with reorientation", draped)):
        lam, info = build_draped_laminate(plies)
        m = Shell(nodes, elements, lam)
        m.constraints[clamped] = True
        m.constraints[loaded] = True
        disp = torch.zeros_like(m.displacements)
        disp[loaded, 1] = u2
        m.displacements = disp
        _, f, sigma, _, _ = m.solve(aggregate_integration_points=False)
        F2 = f[loaded, 1].sum().item()
        # top-station stress in global axes
        s_glob = torch.einsum("eai,eab,ebj->eij", frames, sigma[-1], frames)
        ys = yarn_stresses(m, sigma, info)
        s_L = torch.stack([s["s11"] for s in ys]).amax((0, 1))
        res[name] = {"F2": F2, "s22": s_glob[:, 1, 1]}
        print(f"  {name:24s} reaction F2 = {F2 / 1e3:7.3f} kN   "
              f"max σ22 (top) = {s_glob[:, 1, 1].max():7.1f} MPa   "
              f"max yarn-layer σ_L = {s_L.max():7.1f} MPa")
    k0, k1 = list(res)
    print(f"  force ratio with/without = {res[k1]['F2'] / res[k0]['F2']:.2f} "
          "(paper Fig. 17: force lower with reorientation)")

    fig = plt.figure(figsize=(14, 4.6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    plot_field(ax, fig, nodes, elements, torch.rad2deg(draped[0].gamma.abs()),
               "Shear angle |γ| (KinDrape)", label="deg")
    vmax = max(res[k]["s22"].abs().max().item() for k in res)
    for i, k in enumerate((k0, k1)):
        ax = fig.add_subplot(1, 3, 2 + i, projection="3d")
        plot_field(ax, fig, nodes, elements, res[k]["s22"], f"σ₂₂ top, {k}",
                   cmap="PuOr", label="MPa", vlim=(-vmax, vmax))
    fig.tight_layout()
    fig.savefig(out / "3_dome_inplane_loading.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--drapes", nargs="*", default=[], help="KinDrape .npz files (bottom->top)")
    p.add_argument("--kindrape", type=Path, help="KinDrapeApp folder (runs KinDrape)")
    p.add_argument("--angles", nargs="*", type=float, default=[0.0])
    p.add_argument("--d", type=float, default=0.075, help="KinDrape cell size (R = 1)")
    p.add_argument("--grid", type=int, default=24)
    p.add_argument("--scale", type=float, default=78.0, help="hemisphere radius [mm]")
    p.add_argument("--u2", type=float, default=0.5, help="edge displacement [mm]")
    p.add_argument("--out", type=Path, default=Path("draped_woven_out"))
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    print("[1] Cured ply properties vs shear angle (glass/PP, paper Sec. 5.2 data)")
    study_properties(a.out, GLASS_PP)
    print("[2a] Forming: bias-extension test with non-linear G12(gamma) (Eqs. 21-22)")
    forming = study_bias_forming(a.out)
    print("[2b] Tension of the cured bias specimen (paper Sec. 5.1)")
    study_bias_extension(a.out, GLASS_PA66, forming)

    grids, angles = [], []
    for f in a.drapes:
        data = np.load(f, allow_pickle=True)
        grids.append(data["nodes"])
        angles.append(float(data["parameters"].item().get("ang", 0.0)))
    if not grids and a.kindrape is not None:
        cache = {}
        for ang in a.angles:
            if ang not in cache:
                print(f"  running KinDrape, initial angle {ang:g} deg")
                cache[ang] = run_kindrape(a.kindrape, ang, a.d, a.grid)
            grids.append(cache[ang])
            angles.append(ang)
    if grids:
        print("[3] Draped hemisphere, in-plane loading of the cured part (paper Sec. 5.2)")
        study_dome(a.out, grids, angles, GLASS_PP, a.scale, a.u2)
    else:
        print("[3] skipped (pass --drapes or --kindrape)")
    print(f"Figures written to {a.out.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
