# Public API Namespace for Inverse Design

The public API for Lumix-to-Tidy3D conversion will live under `lumix.inverse_design`, with Tidy3D-specific implementation details kept behind that namespace. This keeps user-facing imports concise while preserving Lumix's core JAX/Flax layer APIs as simulator-independent.

**Considered Options**

- Put conversion helpers directly in core Lumix namespaces.
- Use a public `lumix.tidy3d` namespace.
- Use `lumix.inverse_design` as the public namespace with Tidy3D as an optional integration detail.

**Consequences**

Core Lumix imports should remain independent of Tidy3D. The inverse-design namespace must produce clear errors when the optional simulator dependency is missing.
