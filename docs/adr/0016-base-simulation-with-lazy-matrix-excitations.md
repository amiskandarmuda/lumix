# Base Simulation With Lazy Matrix Excitations

The Lumix-to-Tidy3D template should expose one Base Device Simulation for geometry inspection and a lazy Matrix Excitation Plan for matrix recovery, not an eager public tuple of per-port simulations. Tidy3D inverse design still needs independent source excitations to recover matrix columns, but Lumix should model those as run-time excitation tasks derived from one device template so the API does not imply duplicated devices or encourage memory-heavy eager materialization.

**Considered Options**

- Expose a single Tidy3D simulation with all input sources active.
- Expose an eager tuple of one simulation per input port.
- Expose one base simulation plus a lazy per-input excitation plan.

**Consequences**

A coherent all-sources simulation only gives one superposed output vector, not the individual transfer-matrix columns. The implementation may still pass multiple simulations into Tidy3D `InverseDesignMulti`, but the public Lumix object should keep one inspectable geometry artifact and materialize per-input solves only when constructing or running the inverse-design problem.
