# Matrix File Inputs Are Array-Only

Matrix File Inputs for Lumix-to-Tidy3D conversion will contain only the Target Matrix array. Matrix Port Counts are inferred from array shape, while physical setup belongs in the Device Design Specification so learned optical weights do not silently carry stale geometry assumptions.

**Considered Options**

- Require matrix files to include metadata such as port counts, wavelength, or design dimensions.
- Treat matrix files as array-only and keep physical metadata in the Device Design Specification.

**Consequences**

The converter should validate matrix rank, dtype, and shape directly from the loaded array. Any optional metadata found in richer formats should be treated as external context, not as the source of physical truth.
