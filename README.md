# lumix

`lumix` is a small JAX/Flax library for optical neural networks.

It is organized around concrete Linen layers:

- `ClementsLinear`
- `WilliamsonNonlinearity`
- `PowerReadout`

The pure mathematical kernels live under `lumix.functional`.

## Example

```python
from flax import linen as nn
from lumix.linen.clements import ClementsLinear
from lumix.linen.readout import PowerReadout
from lumix.linen.williamson import WilliamsonNonlinearity


class ElectroOpticNet(nn.Module):
    classes: int = 10

    @nn.compact
    def __call__(self, x):
        x = ClementsLinear(width=16)(x)
        x = WilliamsonNonlinearity(train_gain=True)(x)
        x = ClementsLinear(width=16)(x)
        x = WilliamsonNonlinearity(train_gain=True)(x)
        x = PowerReadout(classes=self.classes)(x)
        return x
```

## Tidy3D inverse-design template

```python
import numpy as np
import lumix.inverse_design as lid

target = np.zeros((16, 16), dtype=np.complex128)

device = lid.DeviceDesignSpec(
    simulation_mode="2d_effective_index",
    wavelength_um=1.55,
    background_eps=1.44**2,
    core_eps=3.48**2,
    core_thickness_um=0.22,
    design_region_size_um=(21.0, 21.0),
    input_pitch_um=1.25,
    output_pitch_um=1.25,
    waveguide_width_um=0.45,
    pixel_size_um=0.05,
    curved_container=lid.CurvedContainerSpec(
        enabled=True,
        corner_radius_um=0.8,
        box_thickness_um=0.5 * 0.55,
        inner_overlap_px=2,
        taper_overlap_px=2,
    ),
    port_taper=lid.PortTaperSpec(
        mouth_width_um=1.25,
        waveguide_width_um=0.45,
        length_um=3.1,
        samples=101,
        initial_profile="linear",
        # For a 16x16 device with 1.25 um pitch, adjacent_touch makes the
        # taper mouths touch with no inter-taper gap inside the 21 um box.
        mouth_mode="adjacent_touch",
        mouth_gap_um=0.0,
    ),
)

template = lid.from_matrix_array(target, device=device)

matrix_params = template.initial_optimization_params(scope="matrix")
taper_params = template.initial_optimization_params(scope="taper")
joint_params = template.initial_optimization_params(scope="matrix_and_taper")

sim = template.base_simulation_with_params(joint_params)
```

By default, the source offset, monitor offset, PML gap, and minimum straight port lead each resolve from the device wavelength with a `1.5 * wavelength_um` buffer. The FDTD runtime defaults to `10 ps`. Per-input simulations include output mode monitors only; set `include_input_monitors=True` on `DeviceDesignSpec` when input-side diagnostics or reflection objectives need them.

Supported static taper profiles are `linear`, `quadratic`, `raised_cosine`, `inverted_quarter_circle`, and `local_adiabatic`.

`PortTaperSpec.mouth_mode="fixed"` keeps every taper mouth at `mouth_width_um`. `mouth_mode="adjacent_touch"` ignores the fixed mouth width for multi-port sides and uses adjacent port midlines as mouth boundaries, so neighboring taper mouths touch with no gap. Set `mouth_gap_um` to keep an explicit fabrication gap between adjacent mouths.
