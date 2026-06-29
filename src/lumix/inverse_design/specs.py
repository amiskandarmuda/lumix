"""Public specifications for Lumix-to-Tidy3D inverse-design conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaperProfile = Literal[
    "linear",
    "quadratic",
    "raised_cosine",
    "inverted_quarter_circle",
    "local_adiabatic",
]
TaperMouthMode = Literal["fixed", "adjacent_touch"]
OptimizationScope = Literal["matrix", "taper", "matrix_and_taper"]


@dataclass(frozen=True)
class PassivityPolicy:
    """How to handle target matrices whose largest singular value exceeds one."""

    mode: Literal["reject", "normalize"] = "reject"
    atol: float = 1e-8

    def __post_init__(self) -> None:
        if self.mode not in {"reject", "normalize"}:
            raise ValueError(f"Unsupported passivity policy: {self.mode!r}")
        if self.atol < 0.0:
            raise ValueError("Passivity tolerance must be non-negative.")


@dataclass(frozen=True)
class MatrixObjectiveSpec:
    """Objective weights for matrix realization."""

    forward_weight: float = 1.0
    reflection_penalty_weight: float = 0.0
    passivity: PassivityPolicy = PassivityPolicy()

    def __post_init__(self) -> None:
        if self.forward_weight <= 0.0:
            raise ValueError("forward_weight must be positive.")
        if self.reflection_penalty_weight < 0.0:
            raise ValueError("reflection_penalty_weight must be non-negative.")


@dataclass(frozen=True)
class CurvedContainerSpec:
    """Optional rounded design container and fixed surrounding ring."""

    enabled: bool = False
    corner_radius_um: float | None = None
    corner_radius_guided_wavelengths: float | None = None
    box_thickness_um: float | None = None
    box_thickness_guided_wavelengths: float | None = None
    inner_overlap_px: int = 0
    taper_overlap_px: int = 0
    guided_wavelength_um: float | None = None

    def __post_init__(self) -> None:
        if self.corner_radius_um is not None and self.corner_radius_guided_wavelengths is not None:
            raise ValueError("Specify only one of corner_radius_um or corner_radius_guided_wavelengths.")
        if self.box_thickness_um is not None and self.box_thickness_guided_wavelengths is not None:
            raise ValueError("Specify only one of box_thickness_um or box_thickness_guided_wavelengths.")
        for name, value in (
            ("corner_radius_um", self.corner_radius_um),
            ("corner_radius_guided_wavelengths", self.corner_radius_guided_wavelengths),
            ("box_thickness_um", self.box_thickness_um),
            ("box_thickness_guided_wavelengths", self.box_thickness_guided_wavelengths),
            ("guided_wavelength_um", self.guided_wavelength_um),
        ):
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.inner_overlap_px < 0:
            raise ValueError("inner_overlap_px must be non-negative.")
        if self.taper_overlap_px < 0:
            raise ValueError("taper_overlap_px must be non-negative.")


@dataclass(frozen=True)
class PortTaperSpec:
    """Explicit polygon taper geometry between ports and the matrix design region."""

    mouth_width_um: float = 1.25
    waveguide_width_um: float | None = None
    length_um: float = 3.1
    samples: int = 101
    initial_profile: TaperProfile = "linear"
    mouth_mode: TaperMouthMode = "adjacent_touch"
    mouth_gap_um: float = 0.0

    def __post_init__(self) -> None:
        _require_positive("mouth_width_um", self.mouth_width_um)
        if self.waveguide_width_um is not None:
            _require_positive("waveguide_width_um", self.waveguide_width_um)
        _require_positive("length_um", self.length_um)
        if self.samples < 8:
            raise ValueError("samples must be at least 8.")
        if self.initial_profile not in {"linear", "quadratic", "raised_cosine", "inverted_quarter_circle", "local_adiabatic"}:
            raise ValueError(f"Unsupported initial_profile: {self.initial_profile!r}")
        if self.mouth_mode not in {"fixed", "adjacent_touch"}:
            raise ValueError(f"Unsupported mouth_mode: {self.mouth_mode!r}")
        if self.mouth_gap_um < 0.0:
            raise ValueError("mouth_gap_um must be non-negative.")


@dataclass(frozen=True)
class DesignParameterSet:
    """Matrix and polygon-taper parameters for local simulation construction."""

    scope: OptimizationScope
    matrix: Any | None = None
    input_taper_widths_um: Any | None = None
    output_taper_widths_um: Any | None = None
    optimize_matrix: bool = True
    optimize_tapers: bool = False

    def __post_init__(self) -> None:
        if self.scope not in {"matrix", "taper", "matrix_and_taper"}:
            raise ValueError(f"Unsupported optimization scope: {self.scope!r}")


@dataclass(frozen=True)
class DeviceDesignSpec:
    """Explicit physical dimensions for a v1 left-to-right matrix device."""

    wavelength_um: float
    background_eps: float
    core_eps: float
    core_thickness_um: float
    design_region_size_um: tuple[float, float]
    design_eps_bounds: tuple[float, float] | None = None
    curved_container: CurvedContainerSpec = field(default_factory=CurvedContainerSpec)
    input_pitch_um: float | None = None
    output_pitch_um: float | None = None
    input_positions_um: tuple[float, ...] | None = None
    output_positions_um: tuple[float, ...] | None = None
    device_margin_um: float = 3.0
    pixel_size_um: float = 0.05
    simulation_mode: Literal["2d_effective_index"] = "2d_effective_index"
    taper_length_um: float = 0.0
    taper_end_width_um: float | None = None
    taper_profile: TaperProfile = "raised_cosine"
    taper_samples: int = 64
    waveguide_width_um: float = 0.45
    mode_window_um: float | None = None
    monitor_distance_um: float | None = None
    source_distance_um: float | None = None
    pml_gap_um: float | None = None
    lead_length_min_wavelengths: float = 1.5
    min_steps_per_wvl: int = 12
    run_time_ps: float = 10.0
    include_input_monitors: bool = False
    port_taper: PortTaperSpec | None = None

    def __post_init__(self) -> None:
        if self.simulation_mode != "2d_effective_index":
            raise ValueError("v1 only supports simulation_mode='2d_effective_index'.")
        _require_positive("wavelength_um", self.wavelength_um)
        _require_positive("background_eps", self.background_eps)
        _require_positive("core_eps", self.core_eps)
        if self.design_eps_bounds is not None:
            if len(self.design_eps_bounds) != 2:
                raise ValueError("design_eps_bounds must be (min_eps, max_eps).")
            _require_positive("design_eps_bounds[0]", self.design_eps_bounds[0])
            _require_positive("design_eps_bounds[1]", self.design_eps_bounds[1])
            if float(self.design_eps_bounds[0]) >= float(self.design_eps_bounds[1]):
                raise ValueError("design_eps_bounds[0] must be smaller than design_eps_bounds[1].")
        _require_positive("core_thickness_um", self.core_thickness_um)
        _require_positive("pixel_size_um", self.pixel_size_um)
        _require_positive("device_margin_um", self.device_margin_um)
        _require_positive("waveguide_width_um", self.waveguide_width_um)
        if self.taper_length_um < 0.0:
            raise ValueError("taper_length_um must be non-negative.")
        if self.taper_end_width_um is not None:
            _require_positive("taper_end_width_um", self.taper_end_width_um)
        if self.taper_profile not in {"linear", "quadratic", "raised_cosine", "inverted_quarter_circle", "local_adiabatic"}:
            raise ValueError(f"Unsupported taper_profile: {self.taper_profile!r}")
        if self.taper_samples < 8:
            raise ValueError("taper_samples must be at least 8.")
        if self.port_taper is not None:
            if self.taper_length_um > 0.0 or self.taper_end_width_um is not None:
                raise ValueError("Use either port_taper or legacy taper_length_um/taper_end_width_um, not both.")
            if self.port_taper.waveguide_width_um is not None and not _almost_equal(
                self.port_taper.waveguide_width_um,
                self.waveguide_width_um,
            ):
                raise ValueError("port_taper.waveguide_width_um must match waveguide_width_um.")
        if len(self.design_region_size_um) != 2:
            raise ValueError("design_region_size_um must be (length_um, width_um).")
        _require_positive("design_region_size_um[0]", self.design_region_size_um[0])
        _require_positive("design_region_size_um[1]", self.design_region_size_um[1])
        if self.input_positions_um is None and self.input_pitch_um is None:
            raise ValueError("Specify input_pitch_um or explicit input_positions_um.")
        if self.output_positions_um is None and self.output_pitch_um is None:
            raise ValueError("Specify output_pitch_um or explicit output_positions_um.")
        if self.input_pitch_um is not None:
            _require_positive("input_pitch_um", self.input_pitch_um)
        if self.output_pitch_um is not None:
            _require_positive("output_pitch_um", self.output_pitch_um)
        _require_positive("lead_length_min_wavelengths", self.lead_length_min_wavelengths)
        if self.min_steps_per_wvl <= 0:
            raise ValueError("min_steps_per_wvl must be positive.")
        _require_positive("run_time_ps", self.run_time_ps)


@dataclass(frozen=True)
class TopologyRegionSpec:
    """Native Tidy3D topology controls passed to the generated design region."""

    transformations: tuple[Any, ...] = ()
    penalties: tuple[Any, ...] = ()
    initialization_spec: Any | None = None
    override_structure_dl: float | bool | None = None


@dataclass(frozen=True)
class PortCounts:
    """Input and output counts implied by a target matrix."""

    n_input: int
    n_output: int


def _require_positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive.")


def _almost_equal(left: float, right: float, *, atol: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= float(atol)
