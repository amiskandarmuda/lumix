# V1 Uses 2D Effective-Index Inverse Design

The first Lumix-to-Tidy3D Inverse Design Template will target 2D effective-index simulations only. Full 3D support is deferred because it changes the interpretation of thickness, source and monitor windows, z-boundaries, meshing, runtime cost, and validation enough that it should be added as a deliberate extension rather than hidden inside the first converter.

**Considered Options**

- Support 2D effective-index templates only in v1.
- Support both 2D and 3D templates from the start.

**Consequences**

The v1 Device Design Specification should be explicit enough to avoid geometry guessing, but it should not promise 3D behavior. Future 3D support should add a separate simulation-mode branch instead of changing the meaning of existing 2D fields.
