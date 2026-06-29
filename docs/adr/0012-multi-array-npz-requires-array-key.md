# Multi-Array NPZ Requires an Array Key

Matrix File Inputs loaded from `.npz` files must be unambiguous. A `.npz` containing exactly one array may be loaded directly, but a `.npz` containing multiple arrays must fail unless the user supplies an explicit array key.

**Considered Options**

- Load the first array in a multi-array `.npz`.
- Prefer conventional keys such as `matrix` or `target`.
- Require an explicit array key when multiple arrays exist.

**Consequences**

Diagnostics should list available array keys so the user can choose deliberately. Silent first-array selection is not acceptable for Target Matrix conversion.
