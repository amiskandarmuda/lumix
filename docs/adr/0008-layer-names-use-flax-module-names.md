# Layer Names Use Flax Module Names

Layer Names for Lumix-to-Tidy3D conversion will use Flax's standard module construction `name` rather than a separate conversion-specific field such as `inverse_design_name`. This keeps layer identity aligned with the parameter tree and avoids introducing a parallel naming system that could drift from saved model state.

**Considered Options**

- Add a conversion-specific layer field.
- Wrap layers with a separate naming helper.
- Use Flax module construction `name` as the Layer Name.

**Consequences**

Layer selection should treat Flax names as first-class identifiers and provide diagnostics when names are missing, generated, or ambiguous. Existing Lumix modules that already pass `name=` to child layers align with this decision.
