# Port Tapers Are Explicit in V1

The v1 Left-to-Right Port Layout will default to straight waveguide entry into the design region, with Port Tapers enabled only when explicitly requested. This keeps the default geometry simple while preserving an extension point for designs where port width and design-region interface width need a transition.

**Considered Options**

- Always include tapers.
- Never include tapers in v1.
- Default to straight entry and make tapers explicit.

**Consequences**

Preflight Inspection should make straight entry and any enabled tapers visible. The Device Design Specification should not silently infer tapers from matrix dimensions.

`PortTaperSpec` is the explicit v1 surface for polygon tapers. It supports the matrix-device default scaffold:

- mouth width: `1.25 um` for the fixed-mouth fallback
- waveguide width: `0.45 um` when explicitly validated against `DeviceDesignSpec.waveguide_width_um`
- length: `3.1 um`
- samples: `101`
- initial profile: `linear`
- mouth mode: `adjacent_touch`
- mouth gap: `0.0 um`

The current 16x16 matrix scaffold uses a `21 x 21 um` design region with `1.25 um` input/output pitch and an approximately `0.8 um` rounded design-region corner radius. With `mouth_mode="adjacent_touch"` and `mouth_gap_um=0.0`, the 16 taper mouths touch side by side across a `20.0 um` active port span inside the `21.0 um` design box.

The default longitudinal port buffer is wavelength-based: source offset, monitor offset, PML gap, and minimum straight port lead each default to `1.5 * wavelength_um` unless explicitly overridden in `DeviceDesignSpec`. The default FDTD runtime is `10 ps`. Per-input simulations include only output mode monitors by default; input-side mode monitors are opt-in with `DeviceDesignSpec(include_input_monitors=True)`.

Supported static profile presets are:

- `linear`
- `quadratic`
- `raised_cosine`
- `inverted_quarter_circle`
- `local_adiabatic`

The `local_adiabatic` preset uses a square-root width law, `W(t) = sqrt(W0^2 + (W1^2 - W0^2) t)`, as a simple local-adiabatic approximation that changes more slowly on the wide side of the taper. Freeform taper optimization is represented by changing the generated `input_taper_widths_um` and `output_taper_widths_um` arrays through `scope="taper"` or `scope="matrix_and_taper"`, not by a separate static profile name.

Supported mouth layout modes are:

- `fixed`: every taper uses `mouth_width_um` at the design-region interface.
- `adjacent_touch`: for multi-port sides, each taper mouth is bounded by the midpoint between adjacent port centers. This makes neighboring mouths touch when `mouth_gap_um=0.0`. A positive `mouth_gap_um` symmetrically shrinks each shared boundary so the gap is explicit.

For `adjacent_touch`, the first and last mouths extrapolate by the nearest adjacent port pitch. This keeps the full port row filled without requiring an independent array of edge widths. Explicit nonuniform port positions are supported as long as positions are unique and the requested `mouth_gap_um` leaves positive mouth width.

Matrix and taper parameterization are separate. Users switch optimization intent through `template.initial_optimization_params(scope=...)` with `scope="matrix"`, `scope="taper"`, or `scope="matrix_and_taper"`. The Tidy3D `InverseDesignMulti` path remains matrix-topology-only; polygon taper parameters are Lumix-side shape parameters used when constructing local simulation templates.
