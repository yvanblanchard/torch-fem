# Laminate

A `Laminate` stacks plane-stress layers into a section for a [Shell](../models/shell.md). It takes the place of the material there, and the shell integrates the layers through the thickness during the analysis rather than reducing them to ABD matrices, so a layer may be nonlinear or carry internal state:

```py
import torch
from torchfem import Laminate, Shell
from torchfem.materials import OrthotropicElasticityPlaneStress

gfrp = OrthotropicElasticityPlaneStress(
    E_1=54000.0, E_2=9400.0, nu_12=0.33, G_12=5500.0, G_13=5500.0, G_23=3000.0
)

layup = Laminate(
    materials=[gfrp] * 4,
    thicknesses=[0.25] * 4,
    angles=[0.0, torch.pi / 2, torch.pi / 2, 0.0],
)

model = Shell(nodes, elements, layup)
```

Layers are given from the bottom surface upwards. `symmetric` mirrors the half-stack about the mid-plane, and `offset` moves the reference surface the shell nodes sit on:

![Stacking sequences of three laminates](../images/laminate/laminate_stacking_light.png#only-light)
![Stacking sequences of three laminates](../images/laminate/laminate_stacking_dark.png#only-dark)

A `Ply` carries a global identifier and an optional boolean element mask, so a ply may cover only part of the mesh, e.g. a local reinforcement or a ply drop. `Laminate.from_plies` stacks plies from the bottom surface upwards, gives each ply zero thickness outside its elements, and keeps the identifiers in `ply_ids`, one per layer:

```py
from torchfem import Ply

patch = elements_near_hole  # boolean mask with shape [n_elem]

layup = Laminate.from_plies(
    [
        Ply(id=1, material=gfrp, thickness=0.25, angle=0.0),
        Ply(id=2, material=gfrp, thickness=0.25, angle=torch.pi / 4, elements=patch),
        Ply(id=3, material=gfrp, thickness=0.25, angle=torch.pi / 2),
    ]
)
```

Layer materials may also be vectorized with one entry per element, to vary the properties of a layer over the mesh.

::: torchfem.Ply
    options:
        show_root_toc_entry: false
        docstring_section_style: list

::: torchfem.Laminate
    options:
        show_root_toc_entry: false
        docstring_section_style: list
        members:
            - __init__
            - from_plies
            - vectorize
            - plot
