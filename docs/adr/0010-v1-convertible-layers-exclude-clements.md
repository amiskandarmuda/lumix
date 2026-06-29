# V1 Convertible Layers Exclude Clements

The v1 conversion feature will support dense unitary and subunitary Lumix layers, and it will not include Clements mesh layer conversion. This keeps the first implementation focused on the requested unitary/subunitary weight-to-inverse-design workflow while leaving mesh-specific conversion as a later extension.

**Considered Options**

- Support every optical Lumix layer with a matrix representation.
- Support dense unitary and subunitary layers first.
- Include Clements mesh layers in v1.

**Consequences**

Clements support should not block the matrix-file or dense-layer workflows. If added later, it should convert through the same Target Matrix boundary rather than introducing a separate inverse-design path.
