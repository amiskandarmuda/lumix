# Ambiguous Layer Selection Fails

Layer selection by name or type must fail when multiple Lumix layers match and the selector does not disambiguate them. Silent first-match behavior is too risky for checkpoint conversion because it can generate an Inverse Design Template for the wrong Target Matrix.

**Considered Options**

- Select the first matching layer.
- Select the shortest or most local matching path.
- Fail and report all matching qualified paths.

**Consequences**

Diagnostics must make ambiguity actionable by listing matching qualified paths and suggesting a more precise Layer Selector.
