"""Structural modelling of sheared (draped) biaxial woven plies.

Implements the approach of

    A. Aridhi, M. Arfaoui, T. Mabrouki, N. Naouar, Y. Denis, M. Zarroug,
    P. Boisse, "Textile composite structural analysis taking into account the
    forming process", Composites Part B 166 (2019) 773-784.
    https://hal.science/hal-02399005

Forming a woven reinforcement on a double-curved part rotates the warp and weft
yarns about their cross-overs: after curing the angle between the yarn
directions ``f_1`` and ``f_2`` is ``pi/2 - gamma`` (in-plane shear angle
``gamma``). The cured ply is then orthotropic in the *bisector* frame of the two
yarn directions, not in the original warp/weft frame.

Aridhi et al. (Sec. 4, Eqs. 14-20) superpose three stiffnesses: tension-only
stiffness along the warp (``E_1``), along the weft (``E_2``) and an isotropic
matrix (``E_m, nu_m``):

$$
    \\mathbb{C} = E_1\\,\\mathbf{f}_1^{\\otimes 4} + E_2\\,\\mathbf{f}_2^{\\otimes 4}
        + \\mathbb{C}_m .
$$

With the yarns at ``-+theta`` from the bisector, ``theta = pi/4 - gamma/2``,
``c = cos(theta)``, ``s = sin(theta)``, a balanced ply has in the bisector
frame (Voigt, engineering shear)

$$
    Q_{11} = 2Ec^4 + Q^m_{11},\\; Q_{22} = 2Es^4 + Q^m_{11},\\;
    Q_{12} = 2Ec^2s^2 + Q^m_{12},\\; Q_{66} = 2Ec^2s^2 + G_m,\\;
    Q_{16} = Q_{26} = 0 .
$$

**Dual UD plies.** The model is exactly the sum of two equivalent UD "yarn
layers" (warp and weft) with volume fractions ``v_1 + v_2 = 1``:

$$
    \\mathbb{C} = v_1\\,\\mathbf{R}(\\theta_1)\\star\\mathbb{C}^{y_1}
        + v_2\\,\\mathbf{R}(\\theta_2)\\star\\mathbb{C}^{y_2}
$$

where, for the Aridhi model, ``C^{y_a} = (E_a / v_a) e_1^{(x)4} + C_m`` (a UD
ply with ``Q_L = E_a/v_a + Q^m_11``, ``Q_T = Q^m_11``, ``Q_12 = Q^m_12``,
``Q_66 = G_m``). The same machinery accepts general UD yarn layers, e.g.
identified from measured woven ply constants (`BiaxialPly.from_woven`).

Laminate representations:

- ``"superposed"``: one layer per woven ply with the summed stiffness (as in
  the paper; no artificial membrane/bending coupling).
- ``"subplies"``: the two UD yarn layers as two `Laminate` layers of
  thicknesses ``v_1 t`` and ``v_2 t`` (same membrane stiffness; a small
  B-coupling appears because the sub-plies sit at different heights).

Optional extension (not in the paper, listed there as future work):
``thickness_mode="areal"`` accounts for the thickening of the sheared ply,
``t = t_0 / cos(gamma)`` (fibre volume and volume fraction conserved).

Angles in this module are *geometric*: counter-clockwise from the first local
axis of each shell element (the `Shell` orientation projected on the element)
about the element normal. They are converted internally to the `Laminate`
angle convention.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .laminate import Laminate
from .materials import Hyperelastic3D, OrthotropicElasticityPlaneStress
from .rotations import planar_rotation
from .utils import stiffness2voigt


# ---------------------------------------------------------------------------
# Yarn layer (equivalent UD ply)
# ---------------------------------------------------------------------------
@dataclass
class YarnLayer:
    """Equivalent UD yarn layer (L = along the yarn, T = transverse)."""

    E_L: float
    E_T: float
    nu_LT: float
    G_LT: float

    @classmethod
    def from_Q(cls, QL: float, QT: float, Q12: float, Q66: float) -> YarnLayer:
        """Build from reduced plane-stress stiffness components."""
        if QT <= 0.0 or QL <= 0.0 or Q12 * Q12 >= QL * QT or Q66 <= 0.0:
            raise ValueError(
                f"Non-physical yarn layer (Q_L={QL:.4g}, Q_T={QT:.4g}, "
                f"Q_12={Q12:.4g}, Q_66={Q66:.4g})."
            )
        return cls(QL - Q12 * Q12 / QT, QT - Q12 * Q12 / QL, Q12 / QT, Q66)

    @property
    def Q(self) -> tuple[float, float, float, float]:
        """Reduced plane-stress stiffness ``(Q_L, Q_T, Q_12, Q_66)``."""
        nu_TL = self.nu_LT * self.E_T / self.E_L
        d = 1.0 - self.nu_LT * nu_TL
        return self.E_L / d, self.E_T / d, self.nu_LT * self.E_T / d, self.G_LT

    def tensor(self) -> Tensor:
        """Fourth-order plane-stress stiffness (2, 2, 2, 2) in yarn axes."""
        QL, QT, Q12, Q66 = self.Q
        C = torch.zeros(2, 2, 2, 2)
        C[0, 0, 0, 0] = QL
        C[1, 1, 1, 1] = QT
        C[0, 0, 1, 1] = C[1, 1, 0, 0] = Q12
        C[0, 1, 0, 1] = C[0, 1, 1, 0] = C[1, 0, 0, 1] = C[1, 0, 1, 0] = Q66
        return C


# ---------------------------------------------------------------------------
# Biaxial (woven) ply made of two yarn layers
# ---------------------------------------------------------------------------
@dataclass
class WovenPly:
    """Measured constants of the *unsheared* cured woven ply (warp=1, weft=2).

    For a balanced fabric the woven constants alone cannot split the
    longitudinal and transverse yarn-layer stiffnesses; ``E_T_yarn`` (the
    transverse modulus of a UD ply of the same constituents) closes the
    problem. For an unbalanced fabric give the actual ``warp_fraction``.
    """

    E_1: float
    E_2: float
    nu_12: float
    G_12: float
    G_13: float
    G_23: float
    t: float
    E_T_yarn: float | None = None
    warp_fraction: float = 0.5
    rho: float = 1.0


@dataclass
class BiaxialPly:
    """Woven ply described by two equivalent UD yarn layers.

    Args:
        warp, weft: Yarn layers (stiffness of a sub-ply of thickness ``v t``).
        t: Cured thickness of the unsheared ply.
        warp_fraction: Volume fraction ``v_1`` of the warp yarn layer.
        G_13, G_23: Transverse shear moduli of the ply (taken isotropic in the
            plane, i.e. their mean is used whatever the yarn orientation).
        rho: Mass density.
    """

    warp: YarnLayer
    weft: YarnLayer
    t: float
    warp_fraction: float = 0.5
    G_13: float = 1.0
    G_23: float = 1.0
    rho: float = 1.0

    # ----- constructors ----------------------------------------------------
    @classmethod
    def aridhi(
        cls,
        E_1: float,
        E_2: float,
        E_m: float,
        nu_m: float,
        t: float,
        G_13: float | None = None,
        G_23: float | None = None,
        rho: float = 1.0,
    ) -> BiaxialPly:
        """Aridhi et al. (2019) model: tension-only yarns + isotropic matrix.

        Args:
            E_1, E_2: Ply-level tensile stiffness of the warp / weft yarns
                (Eqs. 17-18; in the paper the yarn moduli of the forming model).
            E_m, nu_m: Isotropic matrix (Eq. 19).
            t: Ply thickness.
            G_13, G_23: Transverse shear moduli (default: matrix ``G_m``).
        """
        q = E_m / (1.0 - nu_m * nu_m)
        G_m = E_m / (2.0 * (1.0 + nu_m))
        v = 0.5
        warp = YarnLayer.from_Q(E_1 / v + q, q, nu_m * q, G_m)
        weft = YarnLayer.from_Q(E_2 / v + q, q, nu_m * q, G_m)
        G_13 = G_m if G_13 is None else G_13
        G_23 = G_m if G_23 is None else G_23
        return cls(warp, weft, t, v, G_13, G_23, rho)

    @classmethod
    def from_woven(cls, ply: WovenPly) -> BiaxialPly:
        """Identify identical warp/weft yarn layers from woven ply constants.

        The superposition at ``gamma = 0`` must give the woven reduced
        stiffness ``Q^w``:

        $$
            Q^w_{11} = v_1 Q_L + v_2 Q_T,\\quad Q^w_{22} = v_2 Q_L + v_1 Q_T,
            \\quad Q^w_{12} = Q_{12},\\quad Q^w_{66} = Q_{66}.
        $$

        Unbalanced: solved for ``Q_L, Q_T``. Balanced: closed by
        ``E_T = Q_T - Q_12^2/Q_L`` which gives
        ``Q_L^2 - (2Q^w_11 - E_T) Q_L + Q_12^2 = 0``.
        """
        v1 = ply.warp_fraction
        v2 = 1.0 - v1
        nu21 = ply.nu_12 * ply.E_2 / ply.E_1
        d = 1.0 - ply.nu_12 * nu21
        Qw11, Qw22 = ply.E_1 / d, ply.E_2 / d
        Q12 = ply.nu_12 * ply.E_2 / d
        if abs(v1 - 0.5) > 1e-6:
            det = v1 * v1 - v2 * v2
            QL = (v1 * Qw11 - v2 * Qw22) / det
            QT = (v1 * Qw22 - v2 * Qw11) / det
        else:
            if abs(ply.E_1 - ply.E_2) > 1e-6 * ply.E_1:
                raise ValueError(
                    "E_1 != E_2 with warp_fraction = 0.5: set warp_fraction to "
                    "the actual warp fibre fraction of the unbalanced fabric."
                )
            if ply.E_T_yarn is None:
                raise ValueError("A balanced fabric requires E_T_yarn.")
            b = 2.0 * Qw11 - ply.E_T_yarn
            disc = b * b - 4.0 * Q12 * Q12
            if disc <= 0.0:
                raise ValueError("E_T_yarn is too large for these woven constants.")
            QL = 0.5 * (b + math.sqrt(disc))
            QT = 2.0 * Qw11 - QL
        yarn = YarnLayer.from_Q(QL, QT, Q12, ply.G_12)
        return cls(yarn, yarn, ply.t, v1, ply.G_13, ply.G_23, ply.rho)

    # ----- stiffness ---------------------------------------------------------
    def stiffness(self, theta_1: Tensor, theta_2: Tensor) -> Tensor:
        """Plane-stress stiffness ``(n, 2, 2, 2, 2)`` in element axes for yarn
        angles ``theta_1`` (warp) and ``theta_2`` (weft)."""
        v1 = self.warp_fraction
        return v1 * rotate_stiffness(self.warp.tensor(), theta_1) + (
            1.0 - v1
        ) * rotate_stiffness(self.weft.tensor(), theta_2)

    def bisector_stiffness(self, gamma: Tensor) -> Tensor:
        """Reduced stiffness ``(n, 3, 3)`` in the bisector frame for shear
        angles ``gamma`` (yarns at ``-+(pi/4 - gamma/2)`` from the bisector)."""
        gamma = torch.as_tensor(gamma, dtype=torch.get_default_dtype())
        a = math.pi / 4 - gamma / 2
        return stiffness2voigt(self.stiffness(-a, a))

    def bisector_constants(self, gamma: Tensor) -> dict[str, Tensor]:
        """Engineering constants of the sheared ply in the bisector frame
        (x = bisector, y = normal to it) versus the shear angle."""
        S = torch.linalg.inv(self.bisector_stiffness(gamma))
        return {
            "E_x": 1.0 / S[:, 0, 0],
            "E_y": 1.0 / S[:, 1, 1],
            "G_xy": 1.0 / S[:, 2, 2],
            "nu_xy": -S[:, 0, 1] / S[:, 0, 0],
        }

    def modulus(self, gamma: Tensor, phi: Tensor) -> Tensor:
        """Young's modulus along direction ``phi`` (from the bisector) for each
        shear angle: tensor of shape ``(len(gamma), len(phi))``."""
        S = torch.linalg.inv(self.bisector_stiffness(gamma))
        phi = torch.as_tensor(phi, dtype=S.dtype)
        c, s = torch.cos(phi), torch.sin(phi)
        n = torch.stack([c * c, s * s, c * s], -1)  # stress direction (Voigt)
        return 1.0 / torch.einsum("pi,gij,pj->gp", n, S, n)


def rotation_ccw(theta: Tensor) -> Tensor:
    """Rotation matrices ``(..., 2, 2)`` whose columns are the base vectors
    rotated counter-clockwise by ``theta``."""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def rotate_stiffness(C: Tensor, theta: Tensor) -> Tensor:
    """Express a material-axes stiffness in element axes, material axis 1 at
    the geometric (CCW) angle ``theta``. Returns ``(..., 2, 2, 2, 2)``."""
    R = rotation_ccw(theta)
    return torch.einsum("...ia,...jb,...kc,...ld,abcd->...ijkl", R, R, R, R, C)


def shear_angle(theta_1: Tensor, theta_2: Tensor) -> Tensor:
    """Signed shear angle ``gamma = pi/2 - (theta_2 - theta_1)``."""
    a = torch.remainder(theta_2 - theta_1 + math.pi, 2 * math.pi) - math.pi
    return 0.5 * math.pi - a


# ---------------------------------------------------------------------------
# Material holding an arbitrary (element-wise) plane-stress stiffness
# ---------------------------------------------------------------------------
class AnisotropicElasticityPlaneStress(OrthotropicElasticityPlaneStress):
    """Linear elastic plane-stress material with a given stiffness tensor.

    Args:
        C: Stiffness tensor of shape `(n_elem, 2, 2, 2, 2)` (element axes).
        G_13, G_23: Transverse shear moduli, shape `(n_elem,)`.
        rho: Mass density, shape `(n_elem,)`.
    """

    def __init__(self, C: Tensor, G_13: Tensor, G_23: Tensor, rho: Tensor):
        S = torch.linalg.inv(stiffness2voigt(C))
        super().__init__(
            1.0 / S[:, 0, 0],
            1.0 / S[:, 1, 1],
            -S[:, 0, 1] / S[:, 0, 0],
            1.0 / S[:, 2, 2],
            G_13,
            G_23,
            rho,
        )
        self.C = C.clone()

    def rotate(self, R: Tensor) -> AnisotropicElasticityPlaneStress:
        """Returns a copy with the stiffness tensor rotated by ``R``."""
        new = copy.copy(self)
        new.C = torch.einsum(
            "...ijkl,...mi,...nj,...ok,...pl->...mnop", self.C, R, R, R, R
        )
        return new


# ---------------------------------------------------------------------------
# Laminate assembly
# ---------------------------------------------------------------------------
def _laminate_angle_sign() -> float:
    """Sign mapping a geometric CCW angle to the `Laminate` angle argument.

    `Laminate` rotates layers with `planar_rotation`; this checks at runtime
    which way that turns the fibre so results do not depend on the convention.
    """
    m = OrthotropicElasticityPlaneStress(10.0, 1.0, 0.0, 1.0).vectorize(1)
    m = m.rotate(planar_rotation(torch.tensor([0.3])))
    # C_1112 > 0 when the fibre lies at a positive CCW angle
    return 1.0 if m.C[0, 0, 0, 0, 1] > 0 else -1.0


@dataclass
class DrapedPly:
    """A biaxial ply with its yarn directions in every shell element.

    Args:
        ply: Ply material.
        theta_1: Warp angle per element (geometric, CCW from the shell local axis 1).
        theta_2: Weft angle per element.
    """

    ply: BiaxialPly
    theta_1: Tensor
    theta_2: Tensor

    @property
    def gamma(self) -> Tensor:
        return shear_angle(self.theta_1, self.theta_2)


@dataclass
class LayerInfo:
    """Post-processing data of one laminate layer."""

    ply_index: int
    kind: Literal["woven", "warp", "weft"]
    ply: BiaxialPly
    theta_1: Tensor
    theta_2: Tensor


def build_draped_laminate(
    plies: list[DrapedPly],
    representation: Literal["superposed", "subplies"] = "superposed",
    thickness_mode: Literal["none", "areal"] = "none",
    max_gamma: float = math.radians(70.0),
    n_simpson: int = 3,
) -> tuple[Laminate, list[LayerInfo]]:
    """Create a torch-fem `Laminate` of draped biaxial plies (bottom to top).

    Args:
        plies: Draped plies from bottom to top.
        representation: ``"superposed"`` (one layer per ply) or ``"subplies"``
            (warp and weft UD sub-plies).
        thickness_mode: ``"none"`` keeps the nominal thickness (as in Aridhi
            et al.); ``"areal"`` uses ``t0 / cos(gamma)``.
        max_gamma: Shear angle used to cap ``1/cos(gamma)``.
        n_simpson: Simpson points per layer.

    Returns:
        The laminate and per-layer information for stress recovery.
    """
    sign = _laminate_angle_sign()
    materials, thicknesses, angles, info = [], [], [], []
    for k, dp in enumerate(plies):
        p = dp.ply
        n = dp.theta_1.shape[0]
        v1 = p.warp_fraction
        t = torch.full((n,), p.t)
        if thickness_mode == "areal":
            t = t / torch.cos(dp.gamma.abs().clamp(max=max_gamma))
        elif thickness_mode != "none":
            raise ValueError(f"Unknown thickness_mode '{thickness_mode}'.")
        g_t = 0.5 * (p.G_13 + p.G_23)

        if representation == "superposed":
            C = p.stiffness(dp.theta_1, dp.theta_2)
            g = torch.full((n,), g_t)
            materials.append(
                AnisotropicElasticityPlaneStress(C, g, g, torch.full((n,), p.rho))
            )
            thicknesses.append(t)
            angles.append(torch.zeros(n))
            info.append(LayerInfo(k, "woven", p, dp.theta_1, dp.theta_2))
        elif representation == "subplies":
            subplies: tuple[
                tuple[Literal["warp", "weft"], YarnLayer, Tensor, float], ...
            ]
            subplies = (
                ("warp", p.warp, dp.theta_1, v1),
                ("weft", p.weft, dp.theta_2, 1.0 - v1),
            )
            for kind, yarn, theta, frac in subplies:
                materials.append(
                    OrthotropicElasticityPlaneStress(
                        yarn.E_L, yarn.E_T, yarn.nu_LT, yarn.G_LT, g_t, g_t, p.rho
                    )
                )
                thicknesses.append(frac * t)
                angles.append(sign * theta)
                info.append(LayerInfo(k, kind, p, theta, theta))
        else:
            raise ValueError(f"Unknown representation '{representation}'.")

    return Laminate(materials, thicknesses, angles, n_simpson=n_simpson), info


# ---------------------------------------------------------------------------
# Stress recovery per yarn family
# ---------------------------------------------------------------------------
def yarn_stresses(shell, sigma: Tensor, info: list[LayerInfo]) -> list[dict]:
    """Stresses carried by each yarn layer, in its own axes, at every station.

    The station strain is recovered from the station stress and the layer
    stiffness; the yarn-layer stress is ``C^y : (R^T eps R)`` (L along the
    yarn, T transverse).

    Args:
        shell: The solved `Shell` model.
        sigma: Station stresses ``(n_z, n_elem, 2, 2)`` from
            ``shell.solve(aggregate_integration_points=False)`` (Tria1).
        info: Layer information from `build_draped_laminate`.

    Returns:
        One dict per station: ``layer``, ``ply``, ``family`` (list) and
        ``s11, s22, t12`` of shape ``(n_families, n_elem)``.
    """
    lam = shell.section
    out = []
    for j in range(lam.n_z):
        k = int(lam.layer[j])
        li = info[k]
        Cv = stiffness2voigt(lam.materials[k].C)
        s = sigma[j]
        sv = torch.stack([s[..., 0, 0], s[..., 1, 1], s[..., 0, 1]], -1)
        e = torch.linalg.solve(Cv, sv)  # engineering shear strain
        eps = torch.stack(
            [
                torch.stack([e[..., 0], 0.5 * e[..., 2]], -1),
                torch.stack([0.5 * e[..., 2], e[..., 1]], -1),
            ],
            -2,
        )
        if li.kind == "woven":
            fams = [
                ("warp", li.ply.warp, li.theta_1),
                ("weft", li.ply.weft, li.theta_2),
            ]
        else:
            yarn = li.ply.warp if li.kind == "warp" else li.ply.weft
            fams = [(li.kind, yarn, li.theta_1)]
        s11, s22, t12 = [], [], []
        for _, yarn, th in fams:
            R = rotation_ccw(th)
            el = torch.einsum("...ai,...ab,...bj->...ij", R, eps, R)
            ev = torch.stack([el[..., 0, 0], el[..., 1, 1], 2.0 * el[..., 0, 1]], -1)
            sl = torch.einsum("ij,...j->...i", stiffness2voigt(yarn.tensor()), ev)
            s11.append(sl[..., 0])
            s22.append(sl[..., 1])
            t12.append(sl[..., 2])
        out.append(
            {
                "layer": k,
                "ply": li.ply_index,
                "family": [f[0] for f in fams],
                "s11": torch.stack(s11),
                "s22": torch.stack(s22),
                "t12": torch.stack(t12),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Geometry: yarn directions from a kinematic draping grid (e.g. KinDrape)
# ---------------------------------------------------------------------------
def grid_to_shell_mesh(grid_nodes: Tensor) -> dict[str, Tensor]:
    """Triangulate a draped fabric grid and extract yarn directions.

    Args:
        grid_nodes: Draped node positions ``(n_i, n_j, 3)``; warp yarns run
            along index ``i``, weft yarns along index ``j``. Cells with NaN
            nodes (undraped) are skipped.

    Returns:
        Dict with ``nodes (n_nod, 3)``, ``ij (n_nod, 2)`` grid indices,
        ``elements (n_elem, 3)``, element ``warp``/``weft`` unit vectors
        ``(n_elem, 3)`` and ``cell (n_elem,)`` = ``i + j (n_i - 1)`` (KinDrape).
    """
    G = torch.as_tensor(grid_nodes, dtype=torch.get_default_dtype())
    ni, nj, _ = G.shape
    valid = ~torch.isnan(G).any(-1)
    ids = -torch.ones(ni, nj, dtype=torch.long)
    ids[valid] = torch.arange(int(valid.sum()))
    nodes = G[valid]
    ij = torch.nonzero(valid)

    elems, warp, weft, cell = [], [], [], []
    for j in range(nj - 1):
        for i in range(ni - 1):
            q = ids[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]]
            if (q < 0).any():
                continue
            w = 0.5 * ((G[i + 1, j] - G[i, j]) + (G[i + 1, j + 1] - G[i, j + 1]))
            f = 0.5 * ((G[i, j + 1] - G[i, j]) + (G[i + 1, j + 1] - G[i + 1, j]))
            for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                elems.append(torch.stack(tri))
                warp.append(w)
                weft.append(f)
                cell.append(i + j * (ni - 1))
    return {
        "nodes": nodes,
        "ij": ij,
        "elements": torch.stack(elems),
        "warp": torch.nn.functional.normalize(torch.stack(warp), dim=-1),
        "weft": torch.nn.functional.normalize(torch.stack(weft), dim=-1),
        "cell": torch.tensor(cell),
    }


def element_frames(
    nodes: Tensor, elements: Tensor, orientation=(1.0, 0.0, 0.0)
) -> Tensor:
    """Local shell frames ``(n_elem, 3, 3)``, rows ``[e1, e2, n]``, built as in
    `Shell`: ``e1`` is ``orientation`` projected on the element (edge 0->1 where
    the projection vanishes) and ``n`` the Newell mean-plane normal."""
    x = nodes[elements]
    edge1 = x[:, 1] - x[:, 0]
    rel = x - x.mean(dim=1, keepdim=True)
    area = torch.linalg.cross(rel, rel.roll(-1, dims=1), dim=-1).sum(dim=1)
    n = torch.nn.functional.normalize(area, dim=-1)
    o = torch.as_tensor(orientation, dtype=nodes.dtype).expand_as(n)
    proj = o - (o * n).sum(dim=-1, keepdim=True) * n
    degen = (proj.norm(dim=-1) < 1e-8).unsqueeze(-1)
    e1 = torch.nn.functional.normalize(torch.where(degen, edge1, proj), dim=-1)
    e2 = torch.nn.functional.normalize(torch.linalg.cross(n, e1), dim=-1)
    return torch.stack([e1, e2, n], dim=1)


def direction_angles(nodes: Tensor, elements: Tensor, d: Tensor) -> Tensor:
    """Geometric angle of 3D directions ``d (n_elem, 3)`` projected onto each
    element plane, measured CCW from the element's local axis 1."""
    a = torch.einsum("eij,ej->ei", element_frames(nodes, elements), d)
    return torch.atan2(a[:, 1], a[:, 0])


def map_directions(src_points: Tensor, src_dirs: Tensor, dst_points: Tensor) -> Tensor:
    """Nearest-neighbour transfer of direction vectors between meshes (e.g.
    from a draping grid to a structural mesh)."""
    from scipy.spatial import KDTree

    _, idx = KDTree(src_points.numpy()).query(dst_points.numpy())
    return src_dirs[torch.as_tensor(idx)]


def nominal_directions(
    nodes: Tensor, elements: Tensor, ref: Tensor
) -> tuple[Tensor, Tensor]:
    """Design-intent yarn angles ignoring forming: warp = projection of the
    global direction ``ref`` on each element, weft perpendicular to it."""
    a = torch.einsum("eij,j->ei", element_frames(nodes, elements), ref.to(nodes.dtype))
    th1 = torch.atan2(a[:, 1], a[:, 0])
    return th1, th1 + 0.5 * math.pi


# ---------------------------------------------------------------------------
# Bias-extension test kinematics (pin-jointed net, Aridhi et al. Sec. 5.1)
# ---------------------------------------------------------------------------
def bias_extension_shear(W: float, L: float, d: float) -> float:
    """Shear angle in the central zone A (Eq. 21): ``pi/2 - 2 acos((D+d)/(sqrt2 D))``
    with ``D = L - W``."""
    D = L - W
    return math.pi / 2 - 2.0 * math.acos((D + d) / (math.sqrt(2.0) * D))


def bias_extension_kinematics(
    X: Tensor, W: float, L: float, d: float, n_path: int = 400
) -> dict[str, Tensor]:
    """Zones, yarn directions and deformed positions in a bias-extension test.

    Initial specimen ``[0, W] x [0, L]`` loaded along y, yarns at +-45 deg:
    ``g_1 = (1, 1)/sqrt2``, ``g_2 = (-1, 1)/sqrt2``. A yarn through a point is
    *gripped* if it reaches a clamp (y = 0 or y = L) before a free edge.
    Zone C (2 gripped yarns) is rigid, zone A (none) is in pure shear
    ``gamma_A`` with yarns at ``+-(pi/4 - gamma_A/2)`` from the load axis, and in
    zone B (one gripped yarn) the yarn family parallel to its border with zone
    C keeps its initial direction while the other one takes its zone-A
    direction, i.e. ``gamma_B = gamma_A / 2``. This piecewise-homogeneous
    deformation gradient is compatible, so deformed positions are obtained by
    integrating ``F dX`` along straight paths from the bottom clamp centre.

    Args:
        X: Points ``(n, 2)`` in the initial configuration.
        W, L, d: Specimen width, length and machine displacement.
        n_path: Integration steps for the deformed positions.

    Returns:
        Dict with ``zone`` (0=C, 1=B, 2=A), ``gamma``, yarn directions
        ``f1``, ``f2`` ``(n, 2)`` and deformed positions ``x`` ``(n, 2)``.
    """
    gA = bias_extension_shear(W, L, d)
    r2 = 1.0 / math.sqrt(2.0)
    g1 = torch.tensor([r2, r2])
    g2 = torch.tensor([-r2, r2])
    a = math.pi / 4 - gA / 2
    fA1 = torch.tensor([math.sin(a), math.cos(a)])
    fA2 = torch.tensor([-math.sin(a), math.cos(a)])
    G_inv = torch.linalg.inv(torch.stack([g1, g2], 1))

    def classify(P):
        x, y = P[..., 0], P[..., 1]
        # yarn along g1 (y - x = const): down-left reaches x=0 after a rise x
        grip1 = (y <= x) | (L - y <= W - x)
        # yarn along g2 (y + x = const): down-right reaches x=W after W - x
        grip2 = (y <= W - x) | (L - y <= x)
        zone = 2 - grip1.long() - grip2.long()
        f1 = torch.where(
            (zone == 2)[..., None] | ((zone == 1) & grip1)[..., None], fA1, g1
        )
        f2 = torch.where(
            (zone == 2)[..., None] | ((zone == 1) & grip2)[..., None], fA2, g2
        )
        # zone C: both initial directions
        f1 = torch.where((zone == 0)[..., None], g1, f1)
        f2 = torch.where((zone == 0)[..., None], g2, f2)
        return zone, f1, f2

    # In zone B the gripped family rotates to its zone-A direction (it crosses
    # the B/A border); the free family runs parallel to the B/C border and
    # keeps its initial direction.
    zone, f1, f2 = classify(X)
    gamma = torch.tensor([0.0, 0.5 * gA, gA])[zone]

    X0 = torch.tensor([0.5 * W, 0.0])
    s = (torch.arange(n_path) + 0.5) / n_path
    P = X0 + s[:, None, None] * (X - X0)[None]  # (n_path, n, 2)
    _, pf1, pf2 = classify(P)
    F = torch.stack([pf1, pf2], -1) @ G_inv  # (n_path, n, 2, 2)
    dX = (X - X0) / n_path
    x = X0 + torch.einsum("pnij,nj->ni", F, dX)
    return {
        "zone": zone,
        "gamma": gamma,
        "f1": f1,
        "f2": f2,
        "x": x,
        "gamma_A": torch.tensor(gA),
    }


# ---------------------------------------------------------------------------
# Forming: non-linear (shear-angle dependent) in-plane shear modulus
# ---------------------------------------------------------------------------
# Aridhi et al. (2019) Eq. 22, commingled glass/PP plain weave (gamma in rad,
# G in MPa), from the composite reinforcement benchmark (Cao et al. 2008):
#   G_12(gamma) = 8.48 g^4 - 12.0972 g^3 + 6.1275 g^2 - 0.83 g + 0.051
G12_GLASS_PP = (0.051, -0.83, 6.1275, -12.0972, 8.48)  # a_0 ... a_4


def shear_modulus(gamma: Tensor, coeffs=G12_GLASS_PP) -> Tensor:
    """Tangent in-plane shear modulus ``G_12(|gamma|) = sum_k a_k |gamma|^k``."""
    g = torch.as_tensor(gamma).abs()
    return sum((a * g**k for k, a in enumerate(coeffs)), torch.zeros_like(g))


def shear_stress(gamma: Tensor, coeffs=G12_GLASS_PP) -> Tensor:
    """Shear stress ``tau(gamma) = sign(gamma) int_0^|gamma| G_12``, so that
    ``d tau / d gamma = G_12(gamma)`` (the paper's rate law integrated)."""
    gamma = torch.as_tensor(gamma)
    g = gamma.abs()
    return torch.sign(gamma) * sum(
        a * g ** (k + 1) / (k + 1) for k, a in enumerate(coeffs)
    )


def shear_energy(gamma: Tensor, coeffs=G12_GLASS_PP) -> Tensor:
    """Shear energy density ``Phi(gamma) = int_0^|gamma| tau``."""
    g = torch.as_tensor(gamma).abs()
    return sum(
        (a * g ** (k + 2) / ((k + 1) * (k + 2)) for k, a in enumerate(coeffs)),
        torch.zeros_like(g),
    )


def _cross2(a: Tensor, b: Tensor) -> Tensor:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _shear_angle(a1: Tensor, a2: Tensor, f01: Tensor, f02: Tensor) -> Tensor:
    """Signed ``gamma = pi/2 - angle(a1, a2)``, keeping the handedness of the
    initial yarn pair ``(f01, f02)`` and differentiable up to yarn locking."""
    hand = torch.sign(_cross2(f01, f02))
    return torch.atan2((a1 * a2).sum(-1), hand * _cross2(a1, a2))


def _forming_psi(F: Tensor, p: Tensor) -> Tensor:
    """Energy of the woven membrane, ``p = [E1, E2, f01(2), f02(2), a0..a4]``."""
    E1, E2 = p[0], p[1]
    a1 = F @ p[2:4]
    a2 = F @ p[4:6]
    l1 = torch.linalg.norm(a1)
    l2 = torch.linalg.norm(a2)
    gamma = _shear_angle(a1, a2, p[2:4], p[4:6])
    # gamma^2 |gamma|^k (not |gamma|^(k+2)) keeps Phi''(0) = a_0 under autograd
    g2, g = gamma * gamma, gamma.abs()
    phi = 0.5 * p[6] * g2 + sum(
        p[6 + k] * g2 * g**k / ((k + 1) * (k + 2)) for k in range(1, 5)
    )
    return 0.5 * E1 * torch.log(l1) ** 2 + 0.5 * E2 * torch.log(l2) ** 2 + phi


class WovenFormingMembrane(Hyperelastic3D):
    """Membrane law of a woven reinforcement / molten prepreg during forming.

    Hyperelastic counterpart of the rate constitutive law of Aridhi et al.
    (2019), Sec. 2 (Eqs. 7-13 and 22): the stiffness is carried by the two
    yarn directions, which follow the material (``f_a = F f0_a / |F f0_a|``),
    with a tensile modulus ``E_a`` each and an in-plane shear stiffness
    ``G_12(gamma)`` that depends on the current shear angle (shear locking):

    $$
        \\psi(\\mathbf{F}) = \\sum_{a=1}^{2} \\tfrac{1}{2} E_a (\\ln\\lambda_a)^2
            + \\Phi(\\gamma), \\qquad
        \\Phi''(\\gamma) = G_{12}(\\gamma),
        \\quad \\sin\\gamma = \\mathbf{f}_1\\cdot\\mathbf{f}_2 .
    $$

    For the small yarn strains of a forming process, the fibre terms match
    the hypoelastic law (log strain in the fibre-rotated frames) and the shear
    stress conjugate to ``gamma`` follows ``d tau = G_12(gamma) d gamma``. The
    stress and tangent are obtained by automatic differentiation (2D ``F``,
    use with `Planar`).

    Args:
        E_1, E_2: Yarn tensile moduli (MPa).
        f01, f02: Initial warp / weft directions in the plane (unit vectors).
        G12_coeffs: Coefficients ``a_0..a_4`` of ``G_12(gamma)`` (MPa).
    """

    dim = 2

    def __init__(
        self,
        E_1: float,
        E_2: float,
        f01=(1.0, 0.0),
        f02=(0.0, 1.0),
        G12_coeffs=G12_GLASS_PP,
        rho: float = 1.0,
    ):
        if len(G12_coeffs) != 5:
            raise ValueError("G12_coeffs must hold a_0 ... a_4.")
        dtype = torch.get_default_dtype()
        f01 = torch.nn.functional.normalize(torch.as_tensor(f01, dtype=dtype), dim=0)
        f02 = torch.nn.functional.normalize(torch.as_tensor(f02, dtype=dtype), dim=0)
        params = torch.cat(
            [torch.tensor([E_1, E_2]), f01, f02, torch.as_tensor(G12_coeffs)]
        ).to(torch.get_default_dtype())
        super().__init__(_forming_psi, params, rho)

    @staticmethod
    def yarn_state(F: Tensor, params: Tensor) -> dict[str, Tensor]:
        """Current yarn directions, stretches and shear angle from ``F``.

        Args:
            F: Deformation gradients ``(..., 2, 2)``.
            params: Vectorized material parameters ``(..., 11)``.
        """
        a1 = torch.einsum("...ij,...j->...i", F, params[..., 2:4])
        a2 = torch.einsum("...ij,...j->...i", F, params[..., 4:6])
        l1, l2 = a1.norm(dim=-1), a2.norm(dim=-1)
        f1, f2 = a1 / l1[..., None], a2 / l2[..., None]
        gamma = _shear_angle(a1, a2, params[..., 2:4], params[..., 4:6])
        return {"f1": f1, "f2": f2, "stretch_1": l1, "stretch_2": l2, "gamma": gamma}


def bias_specimen_mesh(W: float, L: float, n_w: int) -> tuple[Tensor, Tensor]:
    """Quad mesh of a bias-extension specimen aligned with the +-45 deg yarns.

    Element edges follow the yarns (as in Aridhi et al., Fig. 8), which avoids
    the shear locking of inextensible yarns crossing element edges. Nodes lie
    on the lattice ``i h g_1 + j h g_2`` with ``h sqrt(2) = W / n_w`` so that the
    yarn lines through the clamp corners (the zone boundaries) are mesh lines.
    Cells with all corners in ``[0, W] x [0, L]`` are kept (saw-tooth edges of
    size ``h``).

    Args:
        W, L: Specimen width and length; ``L / W * n_w`` must be an integer.
        n_w: Number of lattice steps across the width.

    Returns:
        ``nodes (n, 2)`` and Quad1 ``elements (m, 4)``.
    """
    p = W / n_w  # lattice period along x and y
    n_l = L / p
    if abs(n_l - round(n_l)) > 1e-9:
        raise ValueError("L / W * n_w must be an integer.")
    r2 = 1.0 / math.sqrt(2.0)
    h = p * r2
    g1 = torch.tensor([r2, r2])
    g2 = torch.tensor([-r2, r2])
    n = int(round(n_l)) + n_w + 2
    ii, jj = torch.meshgrid(
        torch.arange(-n, n + 1.0), torch.arange(-n, n + 1.0), indexing="ij"
    )
    P = ii[..., None] * h * g1 + jj[..., None] * h * g2
    tol = 1e-9 * L
    inside = (P[..., 0] > -tol) & (P[..., 0] < W + tol)
    inside &= (P[..., 1] > -tol) & (P[..., 1] < L + tol)
    ids = -torch.ones(ii.shape, dtype=torch.long)
    ids[inside] = torch.arange(int(inside.sum()))
    q = torch.stack([ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]], -1)
    q = q.reshape(-1, 4)
    q = q[(q >= 0).all(1)]
    used = torch.unique(q)
    remap = -torch.ones(int(inside.sum()), dtype=torch.long)
    remap[used] = torch.arange(len(used))
    return P[inside][used], remap[q]


def bias_extension_forming(
    W: float,
    L: float,
    d: float,
    E_yarn: float,
    G12_coeffs=G12_GLASS_PP,
    thickness: float = 1.0,
    n_w: int = 20,
    n_inc: int = 50,
    max_step: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Bias-extension forming simulation with the `WovenFormingMembrane` law.

    Large-strain membrane (torch-fem `Planar`) of a specimen
    with yarns at +-45 deg, clamped on bands of depth ``h / sqrt2`` at both
    ends, the top band moved by ``d`` in at least ``n_inc`` increments of at
    most ``max_step`` (Newton needs ~1 mm steps from the undeformed state,
    where the shear stiffness ``G_12(0)`` is very low).

    Returns:
        Dict with ``nodes``, ``elements`` (Quad1), ``u`` (n_inc+1, n, 2), ``d``
        (n_inc+1,), ``force`` (n_inc+1,) on the moving clamp, ``gamma``,
        ``f1``, ``f2``, ``stretch`` per increment and element, and the model.
    """
    from .planar import Planar

    nodes, elements = bias_specimen_mesh(W, L, n_w)
    r2 = 1.0 / math.sqrt(2.0)
    mat = WovenFormingMembrane(E_yarn, E_yarn, (r2, r2), (-r2, r2), G12_coeffs)
    model = Planar(nodes, elements, mat, thickness=thickness)
    grip = 0.5 * W / n_w + 1e-9 * L
    bottom = nodes[:, 1] < grip
    top = nodes[:, 1] > L - grip
    model.constraints[bottom | top] = True
    disp = torch.zeros_like(model.displacements)
    disp[top, 1] = d
    model.displacements = disp
    n_inc = max(n_inc, math.ceil(d / max_step))
    inc = torch.linspace(0.0, 1.0, n_inc + 1)
    u, f, _, F, _ = model.solve(
        increments=inc,
        return_intermediate=True,
        max_iter=50,
        verbose=verbose,
    )
    u, f, F = u.detach(), f.detach(), F.detach()
    st = WovenFormingMembrane.yarn_state(F, mat.params.repeat(len(elements), 1))
    return {
        "nodes": nodes,
        "elements": elements,
        "u": u,
        "d": d * inc,
        "force": f[:, top, 1].sum(-1),
        "gamma": st["gamma"],
        "f1": st["f1"],
        "f2": st["f2"],
        "stretch": torch.stack([st["stretch_1"], st["stretch_2"]], -1),
        "model": model,
        "top": top,
        "bottom": bottom,
    }
