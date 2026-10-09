import math

import pytest
import torch

from torchfem import Shell
from torchfem.fabric import (
    BiaxialPly,
    DrapedPly,
    WovenPly,
    bias_extension_kinematics,
    bias_extension_shear,
    build_draped_laminate,
    direction_angles,
    grid_to_shell_mesh,
    shear_angle,
)
from torchfem.mesh import rect_tri
from torchfem.utils import stiffness2voigt

torch.set_default_dtype(torch.float64)

E, EM, NUM = 35400.0, 1500.0, 0.4
ARIDHI = BiaxialPly.aridhi(E, E, EM, NUM, t=1.2)
WOVEN = WovenPly(60000.0, 60000.0, 0.05, 4000.0, 3500.0, 3500.0, 0.3, E_T_yarn=8000.0)


def test_aridhi_closed_form_in_bisector_frame():
    """Eqs. 14-20: tension-only yarns at -+theta + isotropic matrix."""
    gamma = torch.deg2rad(torch.tensor([0.0, 20.0, 45.0, 57.0]))
    Q = ARIDHI.bisector_stiffness(gamma)
    th = math.pi / 4 - gamma / 2
    c, s = torch.cos(th), torch.sin(th)
    q = EM / (1 - NUM**2)
    Gm = EM / (2 * (1 + NUM))
    assert torch.allclose(Q[:, 0, 0], 2 * E * c**4 + q)
    assert torch.allclose(Q[:, 1, 1], 2 * E * s**4 + q)
    assert torch.allclose(Q[:, 0, 1], 2 * E * c**2 * s**2 + NUM * q)
    assert torch.allclose(Q[:, 2, 2], 2 * E * c**2 * s**2 + Gm)
    assert torch.allclose(Q[:, 0, 2], torch.zeros(4), atol=1e-9)
    assert torch.allclose(Q[:, 1, 2], torch.zeros(4), atol=1e-9)


@pytest.mark.parametrize(
    "woven",
    [
        WOVEN,
        WovenPly(
            70000.0, 50000.0, 0.06, 4500.0, 3000.0, 3000.0, 0.25, warp_fraction=0.6
        ),
    ],
)
def test_from_woven_recovers_unsheared_properties(woven):
    ply = BiaxialPly.from_woven(woven)
    th = torch.tensor([0.3])
    Q = stiffness2voigt(ply.stiffness(th, th + math.pi / 2))[0]
    # rotate back to the warp frame by evaluating at theta = 0
    Q0 = stiffness2voigt(ply.stiffness(torch.zeros(1), torch.full((1,), math.pi / 2)))[
        0
    ]
    S = torch.linalg.inv(Q0)
    assert 1 / S[0, 0] == pytest.approx(woven.E_1)
    assert 1 / S[1, 1] == pytest.approx(woven.E_2)
    assert -S[0, 1] / S[0, 0] == pytest.approx(woven.nu_12)
    assert 1 / S[2, 2] == pytest.approx(woven.G_12)
    assert torch.linalg.eigvalsh(Q).min() > 0


def test_bias_modulus_increases_with_shear():
    E_x = ARIDHI.bisector_constants(torch.deg2rad(torch.arange(0.0, 61.0, 5.0)))["E_x"]
    assert (E_x[1:] > E_x[:-1]).all()


def test_shear_angle_sign():
    t1 = torch.tensor([0.0, 0.0, 0.3])
    t2 = torch.tensor([math.pi / 2, math.pi / 3, 0.3 + math.pi / 2 + 0.2])
    assert torch.allclose(shear_angle(t1, t2), torch.tensor([0.0, math.pi / 6, -0.2]))


def _flat(nx=4, ny=4, Lx=1.0, Ly=1.0):
    nodes, elements = rect_tri(nx, ny, Lx, Ly)
    return torch.hstack([nodes, torch.zeros(len(nodes), 1)]), elements


@pytest.mark.parametrize("ply", [ARIDHI, BiaxialPly.from_woven(WOVEN)])
def test_superposed_and_dual_ud_subplies_same_membrane(ply):
    nodes, elements = _flat()
    n = len(elements)
    t1 = torch.rand(n) * math.pi
    t2 = t1 + math.pi / 2 - torch.rand(n) * 0.9
    A = []
    for rep in ("superposed", "subplies"):
        lam, _ = build_draped_laminate([DrapedPly(ply, t1, t2)], representation=rep)
        sec = Shell(nodes, elements, lam).section
        assert sec is not None
        A.append(
            sum(
                stiffness2voigt(m.C) * t[:, None, None]  # type: ignore[attr-defined]
                for m, t in zip(sec.materials, sec.thicknesses)
            )
        )
    assert torch.allclose(A[0], A[1], rtol=1e-10)


def test_fe_tension_along_bisector_matches_clt():
    L, W = 40.0, 10.0
    nodes, elements = _flat(17, 5, L, W)
    n = len(elements)
    gamma = math.radians(40.0)
    a = math.pi / 4 - gamma / 2
    t1 = direction_angles(
        nodes, elements, torch.tensor([math.cos(a), -math.sin(a), 0.0]).expand(n, 3)
    )
    t2 = direction_angles(
        nodes, elements, torch.tensor([math.cos(a), math.sin(a), 0.0]).expand(n, 3)
    )
    lam, _ = build_draped_laminate([DrapedPly(ARIDHI, t1, t2)])
    m = Shell(nodes, elements, lam)
    left = nodes[:, 0] < 1e-9
    right = nodes[:, 0] > L - 1e-9
    m.constraints[:, 2:] = True
    m.constraints[left, 0] = True
    m.constraints[torch.argmin(nodes[:, 0] + nodes[:, 1]), 1] = True
    # consistent nodal forces of a uniform traction on the right edge
    yr = nodes[right, 1]
    order = torch.argsort(yr)
    h = torch.diff(yr[order])
    w = torch.zeros(len(yr))
    w[order[:-1]] += 0.5 * h
    w[order[1:]] += 0.5 * h
    F = 100.0
    m.forces[torch.nonzero(right).ravel(), 0] = F * w / W
    u, _, _, _, _ = m.solve()
    E_fe = (F / (W * ARIDHI.t)) / (u[right, 0].mean().item() / L)
    E_clt = ARIDHI.bisector_constants(torch.tensor([gamma]))["E_x"].item()
    assert E_fe == pytest.approx(E_clt, rel=1e-6)


def test_areal_thickness_mode():
    nodes, elements = _flat(3, 3)
    n = len(elements)
    g = math.radians(40.0)
    lam, _ = build_draped_laminate(
        [DrapedPly(ARIDHI, torch.zeros(n), torch.full((n,), math.pi / 2 - g))],
        thickness_mode="areal",
    )
    m = Shell(nodes, elements, lam)
    assert torch.allclose(m.thickness, torch.full((n,), ARIDHI.t / math.cos(g)))


def test_bias_extension_kinematics():
    W, L, d = 70.0, 210.0, 50.0
    assert math.degrees(bias_extension_shear(W, L, d)) == pytest.approx(57.33, abs=0.01)
    X = torch.tensor(
        [[0.0, L], [W, L], [0.0, 0.0], [35.0, 105.0], [10.0, 30.0], [35.0, 5.0]]
    )
    k = bias_extension_kinematics(X, W, L, d, n_path=2000)
    assert k["zone"].tolist() == [0, 0, 0, 2, 1, 0]
    # top clamp translated by d, bottom fixed, yarns inextensible
    assert torch.allclose(k["x"][:2, 1], torch.full((2,), L + d), atol=0.2)
    assert torch.allclose(k["x"][2], X[2])
    gam = math.pi / 2 - torch.acos((k["f1"] * k["f2"]).sum(-1))
    assert torch.allclose(gam, k["gamma"])
    assert torch.allclose(k["f1"].norm(dim=-1), torch.ones(6))


def test_grid_to_shell_mesh_recovers_imposed_shear():
    g = math.radians(25.0)
    i, j = torch.meshgrid(torch.arange(6.0), torch.arange(5.0), indexing="ij")
    wx, wy = math.cos(math.pi / 2 - g), math.sin(math.pi / 2 - g)
    grid = torch.stack([i + j * wx, j * wy, 0.0 * i], -1)
    grid[0, 0] = float("nan")  # an undraped node removes one cell
    mesh = grid_to_shell_mesh(grid)
    assert len(mesh["elements"]) == 2 * (5 * 4 - 1)
    t1 = direction_angles(mesh["nodes"], mesh["elements"], mesh["warp"])
    t2 = direction_angles(mesh["nodes"], mesh["elements"], mesh["weft"])
    assert torch.allclose(shear_angle(t1, t2), torch.full_like(t1, g))


def test_g12_law_derivatives():
    from torchfem.fabric import shear_energy, shear_modulus, shear_stress

    g = torch.tensor([0.0, 0.3, 0.8, -0.5], requires_grad=True)
    tau = torch.autograd.grad(shear_energy(g).sum(), g, create_graph=True)[0]
    G = torch.autograd.grad(tau.sum(), g)[0]
    assert torch.allclose(tau, shear_stress(g.detach()))
    assert torch.allclose(G[1:], shear_modulus(g.detach())[1:])
    # paper Eq. 22 values
    assert shear_modulus(torch.tensor(0.0)) == pytest.approx(0.051)
    assert shear_modulus(torch.tensor(1.0)) == pytest.approx(
        8.48 - 12.0972 + 6.1275 - 0.83 + 0.051
    )


@pytest.mark.parametrize("gamma", [0.0, 0.4, 0.9])
def test_forming_membrane_trellis_tangent(gamma):
    """Pure trellising of yarns at 0/90: dW/dgamma = tau, d2W/dgamma2 = G12."""
    from torchfem.fabric import WovenFormingMembrane, shear_modulus, shear_stress

    m = WovenFormingMembrane(35400.0, 35400.0).vectorize(1)
    c, s = math.cos(gamma), math.sin(gamma)
    F = torch.tensor([[1.0, s], [0.0, c]])[None]
    z = torch.zeros(1, 2, 2)
    P, _, C = m.step(z, F, z, torch.zeros(1, 0), z, torch.ones(1, 1), 0)
    dF = torch.tensor([[0.0, c], [0.0, -s]])
    ddF = torch.tensor([[0.0, -s], [0.0, -c]])
    tau = (P[0] * dF).sum()
    G = torch.einsum("ij,ijkl,kl->", dF, C[0], dF) + (P[0] * ddF).sum()
    assert tau.item() == pytest.approx(
        shear_stress(torch.tensor(gamma)).item(), abs=1e-12
    )
    assert G.item() == pytest.approx(
        shear_modulus(torch.tensor(gamma)).item(), rel=1e-9
    )
    st = WovenFormingMembrane.yarn_state(F, m.params)
    assert st["gamma"].item() == pytest.approx(gamma)


def test_bias_extension_forming_coarse():
    from torchfem.fabric import bias_extension_forming

    W, L, d = 70.0, 210.0, 30.0
    r = bias_extension_forming(W, L, d, E_yarn=35400.0, n_w=7, n_inc=15)
    cen = r["nodes"][r["elements"]].mean(1)
    core = ((cen[:, 1] - L / 2).abs() < 15) & ((cen[:, 0] - W / 2).abs() < 10)
    g_core = torch.rad2deg(r["gamma"][-1, core]).median().item()
    g_eq21 = math.degrees(bias_extension_shear(W, L, d))
    assert (r["stretch"] - 1).abs().max() < 2e-3  # quasi-inextensible yarns
    assert g_core == pytest.approx(g_eq21, rel=0.15)
    # zone C at the clamps stays unsheared
    clamp_zone = (cen[:, 1] < 10) & ((cen[:, 0] - W / 2).abs() < 15)
    assert torch.rad2deg(r["gamma"][-1, clamp_zone].abs()).max() < 2.0
    # forming load increases monotonically (shear stiffening)
    assert (r["force"][1:] > r["force"][:-1]).all()
