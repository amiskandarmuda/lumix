# Tidy3D Is an Optional Dependency

The Lumix-to-Tidy3D conversion feature will depend on Tidy3D through an optional extra rather than making Tidy3D required for every Lumix install. Lumix remains usable as a JAX/Flax optical neural network library without simulator dependencies, while users who want Inverse Design Templates can install the simulator-facing extra explicitly.

**Considered Options**

- Make Tidy3D a required project dependency.
- Put Tidy3D behind an optional extra for inverse-design workflows.

**Consequences**

The conversion module must fail with a clear missing-extra message when Tidy3D is unavailable. Core Lumix imports should not import Tidy3D eagerly.
