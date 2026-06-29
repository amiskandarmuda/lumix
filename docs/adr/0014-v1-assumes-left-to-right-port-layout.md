# V1 Assumes Left-to-Right Port Layout

The v1 Inverse Design Template will assume a left-to-right port layout: input ports on the left side of the design region and output ports on the right. This keeps Template Construction and Preflight Inspection simple while still covering the direct matrix-realization workflow.

**Considered Options**

- Support arbitrary explicit port coordinates from the start.
- Support only left-to-right port arrays in v1.

**Consequences**

The v1 Device Design Specification should use compact port-array fields such as pitch and width, not a general port-coordinate schema. Arbitrary port placement can be added later as a separate extension.
