# Template Construction Is Local Only

Template Construction for Lumix-to-Tidy3D conversion must not contact Tidy3D cloud or start cost-incurring work. Cloud submission belongs to an explicit Inverse Design Run or an explicit user-called simulation submission, keeping conversion and inspection separate from execution.

**Considered Options**

- Allow Template Construction to submit validation or preflight jobs automatically.
- Keep Template Construction local-only and require explicit execution for cloud work.

**Consequences**

Users can safely convert matrices and inspect an Inverse Design Template without spending credits. Any future convenience API must preserve this boundary and make cloud execution a separate, visible action.
