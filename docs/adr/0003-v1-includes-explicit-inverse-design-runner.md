# V1 Includes an Explicit Inverse Design Runner

The v1 Lumix-to-Tidy3D feature will expose the objective, Base Device Simulation, and Matrix Excitation Plan, and it will also include a minimal explicit runner for the standard matrix-fit inverse-design loop. The runner is optional and separate from template construction so users can inspect the Inverse Design Template before starting any cost-incurring simulations.

**Considered Options**

- Expose only objective pieces and require users to write the optimization loop.
- Hide optimization behind template construction.
- Provide an explicit optional runner.

**Consequences**

Template construction must remain side-effect-free with respect to cloud execution. Starting an Inverse Design Run should be a separate user action with clear cost and inspection boundaries, and per-input excitation solves should be materialized only when constructing or running the Tidy3D inverse-design problem.
