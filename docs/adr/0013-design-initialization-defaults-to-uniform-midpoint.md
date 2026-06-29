# Design Initialization Defaults to Uniform Midpoint

The v1 Inverse Design Template will default Design Initialization to a deterministic uniform midpoint between low and high permittivity states. Random and user-provided initial arrays remain explicit options for experiments that need stochastic starts or warm starts.

**Considered Options**

- Require users to choose an initialization every time.
- Default to random initialization.
- Default to a uniform midpoint and expose other modes explicitly.

**Consequences**

Template Construction remains deterministic by default. Random initialization must require an explicit seed or make nondeterminism clear in validation metadata.
