# Lumix Tidy3D

This context defines the language for converting Lumix optical layer weights into Tidy3D inverse-design artifacts.

## Language

**Inverse Design Template**:
An inspectable object for realizing a target optical matrix as a trainable photonic device. It exposes one base device simulation for geometry inspection plus a Matrix Excitation Plan for recovering the transfer matrix; it does not imply that a cloud run has started.
_Avoid_: single runnable simulation, eager simulation tuple, cloud job

**Target Matrix**:
A complex matrix with shape `(n_output, n_input)` that defines the desired forward transfer from input ports to output ports. It may come from an explicit matrix file, a Lumix layer params subtree, or full model params plus a layer path.
_Avoid_: weights, checkpoint, operator

**Matrix File Input**:
A file input whose contents are only the array for a Target Matrix. Physical metadata belongs in the Device Design Specification, not in the matrix file.
_Avoid_: matrix config, operator manifest

**Passivity Normalization**:
A deliberate scaling of a Target Matrix by its largest singular value when that value is greater than one, producing a globally scaled passive target. It preserves the relative coupling pattern but changes the absolute amplitude scale.
_Avoid_: clipping, projection, automatic repair

**Matrix Port Counts**:
The input and output channel counts implied by a Target Matrix shape, where `(n_output, n_input)` maps to `n_input` input ports and `n_output` output ports. Explicit device specifications must agree with these counts.
_Avoid_: layer width, feature count

**Left-to-Right Port Layout**:
A port layout where input ports are placed on the left side of the design region and output ports are placed on the right side. It is the v1 layout assumption for Inverse Design Templates.
_Avoid_: arbitrary port layout, freeform layout

**Port Taper**:
An optional transition between a port waveguide and the design region interface. In v1, straight waveguide entry is the default and tapers are enabled only by explicit design specification.
_Avoid_: routing taper, coupler

**Curved Container Spec**:
An optional physical layout constraint that rounds the trainable design-region boundary, adds a fixed surrounding core ring, masks unused corner pixels, and scopes the builder-managed mesh override to the curved region's bounding box. It is part of the Device Design Specification, not a native Tidy3D topology transformation.
_Avoid_: fabrication transform, topology penalty, backend-specific fabrication stack

**Device Design Specification**:
The explicit physical specification needed to construct an Inverse Design Template, including wavelength, materials, core thickness, port layout, design region size, simulation margins, and pixel size. It prevents the converter from guessing geometry from a Target Matrix alone.
_Avoid_: dimensions, geometry hints, defaults

**Design Initialization**:
The initial trainable geometry state used by an Inverse Design Template before an Inverse Design Run. The default is a deterministic uniform midpoint, with explicit options for random or user-provided arrays.
_Avoid_: seed geometry, starting guess

**Effective-Index Template**:
An Inverse Design Template whose simulated device is represented by a 2D effective-index model rather than a full 3D structure. It is the v1 scope for Lumix-to-Tidy3D conversion.
_Avoid_: full 3D template, slab approximation

**Forward Transfer Fit**:
The objective term that compares the simulated input-to-output transfer block against the Target Matrix. It is the default realization objective for an Inverse Design Template.
_Avoid_: matrix loss, transmission loss

**Reflection Penalty**:
An optional objective term that penalizes simulated input reflection while fitting the Target Matrix. It discourages devices that match forward transfer by wasting or reflecting power.
_Avoid_: reflection fit, S-matrix objective

**Topology Region Spec**:
The user-supplied design-region controls for an Inverse Design Template, including topology transformations, topology penalties, and initialization behavior. In v1 these controls are expressed with native Tidy3D inverse-design objects rather than a Lumix-owned fabrication DSL.
_Avoid_: FabricationSpec, backend selector, Lumix fabrication stack

**Base Device Simulation**:
The single Tidy3D simulation object that represents the passive device geometry, materials, boundaries, grid, design region, ports, and monitors without selecting all matrix columns at once. It is the primary preflight and inspection artifact for an Inverse Design Template.
_Avoid_: all-input simulation, matrix simulation

**Matrix Excitation Plan**:
The explicit plan for generating one independent source excitation per input port from a Base Device Simulation so the forward transfer matrix columns can be recovered. It should be lazy in the Lumix API to avoid treating per-port solves as separately owned device templates.
_Avoid_: per-port device, eager simulations, all-sources run

**Preflight Inspection**:
An inspection step for an Inverse Design Template that renders or exposes representative geometry, mesh, design region, ports, sources, and monitors before any optimization run starts. It exists to catch physical setup errors before expensive simulation.
_Avoid_: dry run, preview, smoke test

**Inverse Design Run**:
A controlled optimization process that evaluates an Inverse Design Template against its Target Matrix and updates the trainable geometry. It is separate from template construction and may materialize per-input excitation solves for Tidy3D inverse design.
_Avoid_: simulation run, auto-run, training

**Template Construction**:
The local-only creation of an Inverse Design Template from a Target Matrix and Device Design Specification. It may validate and assemble a Base Device Simulation and Matrix Excitation Plan, but it does not start cost-incurring cloud work.
_Avoid_: run, submission, execution

**Layer Selector**:
A user-facing way to identify which Lumix layer should be converted into a Target Matrix. A selector may refer to an explicit path, a stable layer name, or a layer type with an index when multiple matches exist.
_Avoid_: layer path, checkpoint key, parameter key

**Ambiguous Layer Selection**:
A Layer Selector result where more than one Lumix layer matches the requested identifier. Ambiguous selections must fail and report the matching qualified paths rather than choosing a layer silently.
_Avoid_: first match, best effort selection

**Convertible Layer**:
A Lumix layer whose parameters can be converted into a Target Matrix for an Inverse Design Template. In v1, this means dense unitary and subunitary Lumix layers, not Clements mesh layers.
_Avoid_: optical layer, supported layer

**Layer Name**:
A stable identifier supplied through Flax module construction `name` for a Lumix layer intended to be selected for conversion. It exists so users do not have to depend only on generated parameter paths.
_Avoid_: inverse design name, conversion name, display label
