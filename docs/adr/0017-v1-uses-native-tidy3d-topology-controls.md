# V1 Uses Native Tidy3D Topology Controls

The v1 Lumix-to-Tidy3D API will accept native Tidy3D inverse-design topology controls, such as `tdi.FilterProject` and `tdi.ErosionDilationPenalty`, rather than exposing a Lumix-owned fabrication transform DSL. Lumix owns matrix extraction, device layout, port construction, curved-container layout, and objective wiring; Tidy3D owns topology transformations and penalties inside the generated design region.

**Considered Options**

- Define Lumix-specific fabrication transform and penalty classes.
- Add a backend selector to choose native Tidy3D or Lumix-side fabrication execution.
- Accept native Tidy3D topology objects directly.

**Consequences**

Users configure fabrication-aware topology behavior using the same objects that Tidy3D will receive in `TopologyDesignRegion.transformations` and `TopologyDesignRegion.penalties`. The curved container is treated separately as physical device layout: it rounds the trainable region, adds the fixed ring, applies the design mask, and uses a builder-managed bounding-box mesh override. Non-native SSP or NanoComp-style behavior is deferred unless a later use case requires an explicit extension.
