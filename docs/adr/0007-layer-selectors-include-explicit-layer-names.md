# Layer Selectors Include Explicit Layer Names

Whole-model conversion will use Layer Selectors rather than raw parameter paths alone, and Layer Selectors will include support for explicit user-assigned Layer Names. This avoids making Lumix-to-Tidy3D conversion depend only on fragile framework-generated paths while still allowing path-based and type-based selection when useful.

**Considered Options**

- Require slash-separated parameter paths.
- Infer convertible layers only by type.
- Support Layer Selectors with explicit Layer Names, paths, and type-based selection.

**Consequences**

Convertible Lumix layers should have a clear way to carry or expose stable names. Conversion diagnostics should report available Layer Names and ambiguous matches instead of forcing users to inspect nested parameter dictionaries manually.
