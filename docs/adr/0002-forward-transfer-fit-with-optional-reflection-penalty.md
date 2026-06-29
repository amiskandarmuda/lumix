# Forward Transfer Fit With Optional Reflection Penalty

The v1 Inverse Design Template optimizes the simulated forward transfer block against the Target Matrix by default, with an optional input-reflection penalty. This keeps the Target Matrix semantics focused on the desired input-to-output map while still allowing users to discourage high-reflection devices when power efficiency or downstream cascading matters.

**Considered Options**

- Fit only the forward transfer block.
- Fit the full scattering response.
- Fit the forward transfer block and make reflection suppression optional.

**Consequences**

The default objective remains aligned with Lumix layer semantics, where a layer represents a forward map. Users who care about insertion loss, reflected power, or cascaded optical behavior should enable the reflection penalty explicitly.
