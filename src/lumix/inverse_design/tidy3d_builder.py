"""Lazy Tidy3D construction for inverse-design templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lumix.inverse_design.specs import DesignParameterSet, DeviceDesignSpec, OptimizationScope, PortCounts


class MissingTidy3DDependency(ImportError):
    """Raised when Tidy3D-backed construction is requested without Tidy3D."""


@dataclass(frozen=True)
class _Port:
    name: str
    y_um: float
    width_um: float
    mode_window_um: float


def _tidy3d_modules():
    try:
        import tidy3d as td
        import tidy3d.plugins.invdes as tdi
    except ModuleNotFoundError as exc:
        raise MissingTidy3DDependency(
            "Tidy3D is required to build simulations. Install Lumix with the 'tidy3d' extra."
        ) from exc
    return td, tdi


def build_design_region(template):
    td, tdi = _tidy3d_modules()
    device = template.device
    length_um, width_um = device.design_region_size_um
    curved_spec = _resolve_curved_container_spec(device)
    if bool(curved_spec["enabled"]):
        _install_curved_topology_geometry_patch(td, tdi)
    eps_bounds = device.design_eps_bounds or (float(device.background_eps), float(device.core_eps))
    kwargs: dict[str, Any] = {
        "size": (float(length_um), float(width_um), float(device.core_thickness_um)),
        "center": (0.0, 0.0, 0.0),
        "eps_bounds": (float(eps_bounds[0]), float(eps_bounds[1])),
        "transformations": tuple(template.topology.transformations),
        "penalties": tuple(template.topology.penalties),
        "pixel_size": float(device.pixel_size_um),
        "uniform": (False, False, True),
    }
    if bool(curved_spec["enabled"]):
        kwargs["attrs"] = {"curved_geometry_spec": curved_spec}
    if template.topology.initialization_spec is not None:
        kwargs["initialization_spec"] = template.topology.initialization_spec
    return tdi.TopologyDesignRegion(**kwargs)


def build_base_simulation(template):
    return _build_simulation(template, source_index=None, include_design_structure=False)


def build_simulation_with_design_params(template, params):
    if isinstance(params, DesignParameterSet):
        return _build_simulation(
            template,
            source_index=None,
            include_design_structure=params.matrix is not None,
            design_params=params.matrix,
            taper_params=params,
        )
    return _build_simulation(template, source_index=None, include_design_structure=True, design_params=params)


def initial_design_params(design_region):
    return np.full(tuple(int(v) for v in design_region.params_shape), 0.5, dtype=np.float64)


def initial_optimization_params(template, *, scope: OptimizationScope = "matrix") -> DesignParameterSet:
    if scope not in {"matrix", "taper", "matrix_and_taper"}:
        raise ValueError(f"Unsupported optimization scope: {scope!r}")
    matrix = initial_design_params(template.design_region)
    input_tapers = None
    output_tapers = None
    if scope in {"taper", "matrix_and_taper"}:
        input_tapers, output_tapers = initial_port_taper_widths(template)
    return DesignParameterSet(
        scope=scope,
        matrix=matrix,
        input_taper_widths_um=input_tapers,
        output_taper_widths_um=output_tapers,
        optimize_matrix=scope in {"matrix", "matrix_and_taper"},
        optimize_tapers=scope in {"taper", "matrix_and_taper"},
    )


def initial_port_taper_widths(template) -> tuple[np.ndarray, np.ndarray]:
    taper = _resolve_port_taper(template.device)
    if not bool(taper["enabled"]):
        raise ValueError("Taper optimization requires an enabled port taper.")
    counts = template.port_counts
    input_ports, output_ports = _ports_for_counts(template.device, counts)
    samples = int(taper["samples"])
    xs = np.linspace(0.0, float(taper["length_um"]), samples, dtype=np.float64)
    input_mouth_widths = _taper_mouth_widths(template.device, input_ports)
    output_mouth_widths = _taper_mouth_widths(template.device, output_ports)
    input_widths = np.stack(
        [
            _taper_width_profile(
                xs=xs,
                x0=0.0,
                x1=float(taper["length_um"]),
                w0=float(taper["waveguide_width_um"]),
                w1=float(mouth_width_um),
                profile=str(taper["profile"]),
                mirror_profile=True,
            )
            for mouth_width_um in input_mouth_widths
        ],
        axis=0,
    )
    output_widths = np.stack(
        [
            _taper_width_profile(
                xs=xs,
                x0=0.0,
                x1=float(taper["length_um"]),
                w0=float(mouth_width_um),
                w1=float(taper["waveguide_width_um"]),
                profile=str(taper["profile"]),
                mirror_profile=False,
            )
            for mouth_width_um in output_mouth_widths
        ],
        axis=0,
    )
    return (input_widths, output_widths)


def apply_design_mask(template, params):
    device = template.device
    spec = _resolve_curved_container_spec(device)
    arr = np.asarray(params, dtype=np.float64)
    if not bool(spec["enabled"]):
        return np.array(arr, dtype=np.float64)
    if arr.ndim == 2:
        shape_2d = arr.shape
    elif arr.ndim == 3 and int(arr.shape[2]) == 1:
        shape_2d = arr.shape[:2]
    else:
        raise ValueError(f"params must be 2D or 3D with singleton z, got {tuple(arr.shape)}")
    xs, ys = _grid_coords(
        shape=shape_2d,
        length_um=float(device.design_region_size_um[0]),
        width_um=float(device.design_region_size_um[1]),
        pixel_size_um=float(device.pixel_size_um),
    )
    mask = _rounded_rect_mask(
        xs=xs,
        ys=ys,
        length_um=float(spec["inner_length_um"]),
        width_um=float(spec["inner_width_um"]),
        radius_um=float(spec["corner_radius_um"]),
    )
    if arr.ndim == 2:
        return np.asarray(arr * mask, dtype=np.float64)
    return np.asarray(arr * mask[:, :, None], dtype=np.float64)


class MatrixExcitationPlan:
    """Lazy per-input source simulations for matrix-column recovery."""

    def __init__(self, template):
        self._template = template

    @property
    def n_inputs(self) -> int:
        return self._template.port_counts.n_input

    @property
    def input_port_names(self) -> tuple[str, ...]:
        return tuple(f"in_mode_{index}" for index in range(self.n_inputs))

    def simulation_for_input(self, index: int):
        if index < 0 or index >= self.n_inputs:
            raise IndexError(f"Input index {index} is outside [0, {self.n_inputs}).")
        return _build_simulation(self._template, source_index=int(index), include_design_structure=False)

    def simulations(self) -> tuple[Any, ...]:
        return tuple(self.simulation_for_input(index) for index in range(self.n_inputs))


class PreflightInspection:
    """Local geometry inspection helper for an inverse-design template."""

    def __init__(self, template):
        self._template = template

    @property
    def simulation(self):
        return self._template.base_simulation_with_params(self._template.initial_design_params)

    def plot_geometry(self, **kwargs):
        return self.simulation.plot_eps(z=0.0, **kwargs)

    def assert_ports_match_target(self) -> None:
        counts = self._template.port_counts
        input_ports, output_ports = _ports_for_counts(self._template.device, counts)
        if len(input_ports) != counts.n_input or len(output_ports) != counts.n_output:
            raise AssertionError("Resolved port counts do not match the target matrix.")

    def assert_topology_region_valid(self) -> None:
        design_region = self._template.design_region
        if not tuple(int(v) for v in design_region.params_shape):
            raise AssertionError("Design region did not resolve a parameter shape.")


def to_tidy3d_inverse_design(template, *, task_name: str, verbose: bool = True):
    _, tdi = _tidy3d_modules()
    monitor_names = _output_monitor_names(template.port_counts)
    return tdi.InverseDesignMulti(
        design_region=template.design_region,
        simulations=template.excitation_plan.simulations(),
        task_name=str(task_name),
        output_monitor_names=tuple(monitor_names for _ in range(template.port_counts.n_input)),
        verbose=bool(verbose),
    )


def _build_simulation(
    template,
    *,
    source_index: int | None,
    include_design_structure: bool,
    design_params=None,
    taper_params: DesignParameterSet | None = None,
):
    td, _ = _tidy3d_modules()
    device = template.device
    counts = template.port_counts
    input_ports, output_ports = _ports_for_counts(device, counts)
    x_positions = _x_positions(device)
    sx, sy, sz = _domain_size(device, input_ports, output_ports, x_positions)
    medium_clad = td.Medium(permittivity=float(device.background_eps))
    medium_core = td.Medium(permittivity=float(device.core_eps))
    mode_spec = _mode_spec(td, medium_core, wavelength_um=device.wavelength_um)
    input_taper_widths, output_taper_widths = _validated_taper_widths(template, taper_params)
    structures = _waveguide_structures(
        td,
        device,
        input_ports,
        output_ports,
        x_positions,
        medium_core,
        input_taper_widths=input_taper_widths,
        output_taper_widths=output_taper_widths,
    )
    structures.extend(_fixed_design_structures(td, device, medium_core))
    if include_design_structure:
        params = initial_design_params(template.design_region) if design_params is None else design_params
        structures = [*structures, template.design_region.to_structure(apply_design_mask(template, params))]
    sources = []
    if source_index is not None:
        source_port = input_ports[int(source_index)]
        sources = [
            td.ModeSource(
                center=(float(x_positions["x_source"]), float(source_port.y_um), 0.0),
                size=(0.0, _source_size_y(device, input_ports), td.inf),
                source_time=td.GaussianPulse(
                    freq0=float(td.C_0 / device.wavelength_um),
                    fwidth=float(0.15 * td.C_0 / device.wavelength_um),
                ),
                direction="+",
                mode_spec=mode_spec,
                mode_index=0,
                name=f"src_mode_{source_index}",
            )
        ]
    monitors = (
        _monitor_bank(td, device, input_ports, output_ports, x_positions, mode_spec)
        if source_index is not None
        else []
    )
    return td.Simulation(
        size=(sx, sy, sz),
        center=(0.5 * float(x_positions["x_min"] + x_positions["x_max"]), 0.0, 0.0),
        medium=medium_clad,
        structures=list(structures),
        sources=list(sources),
        monitors=list(monitors),
        run_time=float(device.run_time_ps) * 1e-12,
        grid_spec=td.GridSpec.auto(
            wavelength=float(device.wavelength_um),
            min_steps_per_wvl=int(device.min_steps_per_wvl),
            override_structures=_mesh_override_structures(td, template),
        ),
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.periodic(),
        ),
    )


def _ports_for_counts(device: DeviceDesignSpec, counts: PortCounts) -> tuple[tuple[_Port, ...], tuple[_Port, ...]]:
    mode_window_um = float(device.mode_window_um or max(1.5 * device.waveguide_width_um, device.waveguide_width_um + 0.4))
    input_y = _resolve_port_positions(
        count=counts.n_input,
        pitch_um=device.input_pitch_um,
        positions_um=device.input_positions_um,
        role="input",
    )
    output_y = _resolve_port_positions(
        count=counts.n_output,
        pitch_um=device.output_pitch_um,
        positions_um=device.output_positions_um,
        role="output",
    )
    input_ports = tuple(
        _Port(name=f"in_mode_{index}", y_um=float(y_um), width_um=float(device.waveguide_width_um), mode_window_um=mode_window_um)
        for index, y_um in enumerate(input_y)
    )
    output_ports = tuple(
        _Port(name=f"out_mode_{index}", y_um=float(y_um), width_um=float(device.waveguide_width_um), mode_window_um=mode_window_um)
        for index, y_um in enumerate(output_y)
    )
    return input_ports, output_ports


def _resolve_port_positions(*, count: int, pitch_um: float | None, positions_um: tuple[float, ...] | None, role: str) -> np.ndarray:
    if positions_um is not None:
        if len(positions_um) != count:
            raise ValueError(f"Expected {count} {role} port positions, got {len(positions_um)}.")
        return np.asarray(positions_um, dtype=np.float64)
    if pitch_um is None:
        raise ValueError(f"Missing {role}_pitch_um for {count} ports.")
    indices = np.arange(int(count), dtype=np.float64)
    return (indices - 0.5 * (int(count) - 1)) * float(pitch_um)


def _x_positions(device: DeviceDesignSpec) -> dict[str, float]:
    wavelength_um = float(device.wavelength_um)
    design_length_um = float(device.design_region_size_um[0])
    x_left = -0.5 * design_length_um
    x_right = 0.5 * design_length_um
    taper_len_um = float(_resolve_port_taper(device)["length_um"])
    monitor_distance_um = float(device.monitor_distance_um or 1.5 * wavelength_um)
    source_distance_um = float(device.source_distance_um or 1.5 * wavelength_um)
    pml_gap_um = float(device.pml_gap_um or 1.5 * wavelength_um)
    lead_len_um = max(
        float(device.lead_length_min_wavelengths) * wavelength_um,
        monitor_distance_um + pml_gap_um,
        source_distance_um + pml_gap_um,
    )
    x_input_taper_start = x_left - taper_len_um
    x_output_taper_end = x_right + taper_len_um
    x_input_start = x_left - taper_len_um - lead_len_um
    x_output_end = x_right + taper_len_um + lead_len_um
    return {
        "x_min": x_input_start,
        "x_max": x_output_end,
        "x_left": x_left,
        "x_right": x_right,
        "x_source": x_input_taper_start - source_distance_um,
        "x_in_monitor": x_input_taper_start - monitor_distance_um,
        "x_out_monitor": x_output_taper_end + monitor_distance_um,
        "x_input_start": x_input_start,
        "x_output_end": x_output_end,
    }


def _domain_size(device: DeviceDesignSpec, input_ports, output_ports, x_positions: dict[str, float]) -> tuple[float, float, float]:
    sx = float(x_positions["x_max"] - x_positions["x_min"])
    y_values = np.asarray([port.y_um for port in (*input_ports, *output_ports)], dtype=np.float64)
    port_span = float(np.max(y_values) - np.min(y_values)) if y_values.size else 0.0
    sy = float(max(float(device.design_region_size_um[1]), port_span + float(device.waveguide_width_um)) + 2.0 * float(device.device_margin_um))
    return sx, sy, 0.0


def _validated_taper_widths(template, taper_params: DesignParameterSet | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if taper_params is None or not taper_params.optimize_tapers:
        return None, None
    taper = _resolve_port_taper(template.device)
    if not bool(taper["enabled"]):
        raise ValueError("Taper parameters require an enabled port taper.")
    input_ports, output_ports = _ports_for_counts(template.device, template.port_counts)
    input_widths = _validate_taper_width_array(
        taper_params.input_taper_widths_um,
        port_count=len(input_ports),
        samples=int(taper["samples"]),
        role="input",
    )
    output_widths = _validate_taper_width_array(
        taper_params.output_taper_widths_um,
        port_count=len(output_ports),
        samples=int(taper["samples"]),
        role="output",
    )
    return input_widths, output_widths


def _validate_taper_width_array(widths, *, port_count: int, samples: int, role: str) -> np.ndarray:
    if widths is None:
        raise ValueError(f"Missing {role} taper widths.")
    arr = np.asarray(widths, dtype=np.float64)
    expected_shape = (int(port_count), int(samples))
    if arr.shape != expected_shape:
        raise ValueError(f"Expected {role} taper widths with shape {expected_shape}, got {arr.shape}.")
    if np.any(arr <= 0.0):
        raise ValueError(f"{role} taper widths must be positive.")
    return arr


def _resolve_port_taper(device: DeviceDesignSpec) -> dict[str, float | int | bool | str]:
    if device.port_taper is not None:
        return {
            "enabled": True,
            "length_um": float(device.port_taper.length_um),
            "mouth_width_um": float(device.port_taper.mouth_width_um),
            "waveguide_width_um": float(device.waveguide_width_um),
            "samples": int(device.port_taper.samples),
            "profile": str(device.port_taper.initial_profile),
            "mouth_mode": str(device.port_taper.mouth_mode),
            "mouth_gap_um": float(device.port_taper.mouth_gap_um),
        }
    taper_len_um = float(device.taper_length_um)
    mouth_width_um = float(device.taper_end_width_um or device.waveguide_width_um)
    return {
        "enabled": bool(taper_len_um > 0.0 and mouth_width_um != float(device.waveguide_width_um)),
        "length_um": taper_len_um,
        "mouth_width_um": mouth_width_um,
        "waveguide_width_um": float(device.waveguide_width_um),
        "samples": int(device.taper_samples),
        "profile": str(device.taper_profile),
        "mouth_mode": "fixed",
        "mouth_gap_um": 0.0,
    }


def _taper_mouth_widths(device: DeviceDesignSpec, ports) -> np.ndarray:
    bounds = _taper_mouth_bounds(device, ports)
    return np.asarray(bounds[:, 1] - bounds[:, 0], dtype=np.float64)


def _taper_mouth_bounds(device: DeviceDesignSpec, ports) -> np.ndarray:
    taper = _resolve_port_taper(device)
    centers = np.asarray([port.y_um for port in ports], dtype=np.float64)
    if centers.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    mouth_width_um = float(taper["mouth_width_um"])
    if str(taper["mouth_mode"]) == "fixed" or centers.size == 1:
        return np.stack([centers - 0.5 * mouth_width_um, centers + 0.5 * mouth_width_um], axis=1)

    order = np.argsort(centers)
    sorted_centers = centers[order]
    center_diffs = np.diff(sorted_centers)
    if np.any(center_diffs <= 0.0):
        raise ValueError("adjacent_touch taper mouths require unique port positions.")

    gap_um = float(taper["mouth_gap_um"])
    split_lines = 0.5 * (sorted_centers[:-1] + sorted_centers[1:])
    lower = np.empty_like(sorted_centers)
    upper = np.empty_like(sorted_centers)
    lower[0] = sorted_centers[0] - 0.5 * center_diffs[0] + 0.5 * gap_um
    upper[-1] = sorted_centers[-1] + 0.5 * center_diffs[-1] - 0.5 * gap_um
    lower[1:] = split_lines + 0.5 * gap_um
    upper[:-1] = split_lines - 0.5 * gap_um
    if np.any(upper <= lower):
        raise ValueError("mouth_gap_um is too large for adjacent_touch taper mouths.")

    sorted_bounds = np.stack([lower, upper], axis=1)
    bounds = np.empty_like(sorted_bounds)
    bounds[order] = sorted_bounds
    return bounds


def _waveguide_structures(
    td,
    device: DeviceDesignSpec,
    input_ports,
    output_ports,
    x_positions: dict[str, float],
    medium_core,
    *,
    input_taper_widths: np.ndarray | None = None,
    output_taper_widths: np.ndarray | None = None,
):
    structures = []
    input_mouth_bounds = _taper_mouth_bounds(device, input_ports)
    output_mouth_bounds = _taper_mouth_bounds(device, output_ports)
    for index, port in enumerate(input_ports):
        taper_widths = None if input_taper_widths is None else input_taper_widths[index]
        structures.extend(
            _input_waveguide_structures(
                td,
                device,
                x_positions,
                port,
                medium_core,
                taper_widths=taper_widths,
                mouth_bounds=input_mouth_bounds[index],
            )
        )
    for index, port in enumerate(output_ports):
        taper_widths = None if output_taper_widths is None else output_taper_widths[index]
        structures.extend(
            _output_waveguide_structures(
                td,
                device,
                x_positions,
                port,
                medium_core,
                taper_widths=taper_widths,
                mouth_bounds=output_mouth_bounds[index],
            )
        )
    return structures


def _input_waveguide_structures(
    td,
    device: DeviceDesignSpec,
    x_positions: dict[str, float],
    port: _Port,
    medium_core,
    *,
    taper_widths: np.ndarray | None = None,
    mouth_bounds: np.ndarray | None = None,
):
    taper = _resolve_port_taper(device)
    taper_len_um = float(taper["length_um"])
    taper_mouth_bounds = _single_taper_mouth_bounds(port, taper, mouth_bounds)
    taper_end_width_um = float(taper_mouth_bounds[1] - taper_mouth_bounds[0])
    taper_end_center_um = float(0.5 * (taper_mouth_bounds[0] + taper_mouth_bounds[1]))
    if taper_len_um <= 0.0 or taper_end_width_um == float(port.width_um):
        return [_waveguide_segment(td, x_positions["x_input_start"], x_positions["x_left"], port, medium_core)]
    taper_overlap_um = _taper_overlap_um(device, taper_len_um)
    x_taper_start = float(x_positions["x_left"] - taper_len_um)
    lead = _waveguide_segment(td, x_positions["x_input_start"], x_taper_start + taper_overlap_um, port, medium_core)
    xs = np.linspace(x_taper_start, x_positions["x_left"], int(taper["samples"]), dtype=np.float64)
    widths = (
        np.asarray(taper_widths, dtype=np.float64)
        if taper_widths is not None
        else _taper_width_profile(
            xs=xs,
            x0=x_taper_start,
            x1=x_positions["x_left"],
            w0=port.width_um,
            w1=taper_end_width_um,
            profile=str(taper["profile"]),
            mirror_profile=True,
        )
    )
    centers = _taper_center_profile(
        xs=xs,
        x0=x_taper_start,
        x1=x_positions["x_left"],
        c0=port.y_um,
        c1=taper_end_center_um,
        profile=str(taper["profile"]),
        mirror_profile=True,
    )
    return [
        lead,
        _taper_structure(
            td,
            xs=xs,
            widths=widths,
            centers=centers,
            port=port,
            medium_core=medium_core,
        ),
    ]


def _output_waveguide_structures(
    td,
    device: DeviceDesignSpec,
    x_positions: dict[str, float],
    port: _Port,
    medium_core,
    *,
    taper_widths: np.ndarray | None = None,
    mouth_bounds: np.ndarray | None = None,
):
    taper = _resolve_port_taper(device)
    taper_len_um = float(taper["length_um"])
    taper_mouth_bounds = _single_taper_mouth_bounds(port, taper, mouth_bounds)
    taper_end_width_um = float(taper_mouth_bounds[1] - taper_mouth_bounds[0])
    taper_start_center_um = float(0.5 * (taper_mouth_bounds[0] + taper_mouth_bounds[1]))
    if taper_len_um <= 0.0 or taper_end_width_um == float(port.width_um):
        return [_waveguide_segment(td, x_positions["x_right"], x_positions["x_output_end"], port, medium_core)]
    taper_overlap_um = _taper_overlap_um(device, taper_len_um)
    x_taper_end = float(x_positions["x_right"] + taper_len_um)
    xs = np.linspace(x_positions["x_right"], x_taper_end, int(taper["samples"]), dtype=np.float64)
    widths = (
        np.asarray(taper_widths, dtype=np.float64)
        if taper_widths is not None
        else _taper_width_profile(
            xs=xs,
            x0=x_positions["x_right"],
            x1=x_taper_end,
            w0=taper_end_width_um,
            w1=port.width_um,
            profile=str(taper["profile"]),
            mirror_profile=False,
        )
    )
    centers = _taper_center_profile(
        xs=xs,
        x0=x_positions["x_right"],
        x1=x_taper_end,
        c0=taper_start_center_um,
        c1=port.y_um,
        profile=str(taper["profile"]),
        mirror_profile=False,
    )
    lead = _waveguide_segment(td, x_taper_end - taper_overlap_um, x_positions["x_output_end"], port, medium_core)
    return [
        _taper_structure(
            td,
            xs=xs,
            widths=widths,
            centers=centers,
            port=port,
            medium_core=medium_core,
        ),
        lead,
    ]


def _single_taper_mouth_bounds(
    port: _Port,
    taper: dict[str, float | int | bool | str],
    mouth_bounds: np.ndarray | None,
) -> np.ndarray:
    if mouth_bounds is not None:
        bounds = np.asarray(mouth_bounds, dtype=np.float64).reshape(2)
        if bounds[1] <= bounds[0]:
            raise ValueError("Taper mouth bounds must be increasing.")
        return bounds
    mouth_width_um = float(taper["mouth_width_um"])
    return np.asarray([float(port.y_um) - 0.5 * mouth_width_um, float(port.y_um) + 0.5 * mouth_width_um], dtype=np.float64)


def _waveguide_segment(td, x0: float, x1: float, port: _Port, medium_core):
    return td.Structure(
        geometry=td.Box(
            center=(0.5 * (float(x0) + float(x1)), float(port.y_um), 0.0),
            size=(float(abs(x1 - x0)), float(port.width_um), td.inf),
        ),
        medium=medium_core,
    )


def _taper_structure(
    td,
    *,
    xs: np.ndarray,
    widths: np.ndarray,
    port: _Port,
    medium_core,
    centers: np.ndarray | None = None,
):
    xarr = np.asarray(xs, dtype=np.float64).reshape(-1)
    warr = np.asarray(widths, dtype=np.float64).reshape(-1)
    carr = np.full_like(warr, float(port.y_um)) if centers is None else np.asarray(centers, dtype=np.float64).reshape(-1)
    if xarr.shape != warr.shape or xarr.shape != carr.shape:
        raise ValueError("Taper x, width, and center arrays must have the same shape.")
    y_lo = carr - 0.5 * warr
    y_hi = carr + 0.5 * warr
    vertices = np.concatenate(
        [
            np.stack([xarr, y_lo], axis=1),
            np.stack([xarr[::-1], y_hi[::-1]], axis=1),
        ],
        axis=0,
    )
    return td.Structure(
        geometry=td.PolySlab(axis=2, slab_bounds=(-td.inf, td.inf), vertices=vertices),
        medium=medium_core,
    )


def _taper_overlap_um(device: DeviceDesignSpec, taper_len_um: float) -> float:
    overlap_um = float(device.curved_container.taper_overlap_px) * float(device.pixel_size_um)
    return float(min(max(0.0, overlap_um), max(0.0, taper_len_um)))


def _taper_width_profile(
    *,
    xs: np.ndarray,
    x0: float,
    x1: float,
    w0: float,
    w1: float,
    profile: str,
    mirror_profile: bool,
) -> np.ndarray:
    base = _taper_profile_fraction(xs=xs, x0=x0, x1=x1, profile=profile, mirror_profile=mirror_profile)
    if profile == "local_adiabatic":
        widths_sq = float(w0) ** 2 + (float(w1) ** 2 - float(w0) ** 2) * base
        return np.asarray(np.sqrt(np.maximum(0.0, widths_sq)), dtype=np.float64)
    return np.asarray(float(w0) + (float(w1) - float(w0)) * base, dtype=np.float64)


def _taper_center_profile(
    *,
    xs: np.ndarray,
    x0: float,
    x1: float,
    c0: float,
    c1: float,
    profile: str,
    mirror_profile: bool,
) -> np.ndarray:
    base = _taper_profile_fraction(xs=xs, x0=x0, x1=x1, profile=profile, mirror_profile=mirror_profile)
    return np.asarray(float(c0) + (float(c1) - float(c0)) * base, dtype=np.float64)


def _taper_profile_fraction(*, xs: np.ndarray, x0: float, x1: float, profile: str, mirror_profile: bool) -> np.ndarray:
    span = max(1e-12, float(x1) - float(x0))
    u = (np.asarray(xs, dtype=np.float64) - float(x0)) / span
    base = _profile_interp(u, profile)
    if mirror_profile:
        base = 1.0 - _profile_interp(1.0 - np.clip(u, 0.0, 1.0), profile)
    return np.asarray(base, dtype=np.float64)


def _profile_interp(u: np.ndarray, profile: str) -> np.ndarray:
    uu = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    if profile == "linear":
        return uu
    if profile == "quadratic":
        return uu**2
    if profile == "raised_cosine":
        return 0.5 * (1.0 - np.cos(np.pi * uu))
    if profile == "inverted_quarter_circle":
        return np.sqrt(np.maximum(0.0, 1.0 - (1.0 - uu) ** 2))
    if profile == "local_adiabatic":
        return uu
    raise ValueError(f"Unsupported taper_profile: {profile!r}")


def _mode_spec(td, medium_core, *, wavelength_um: float):
    freq0 = float(td.C_0 / wavelength_um)
    try:
        eps_value = np.asarray(medium_core.eps_model(freq0)).reshape(-1)[0]
        target_neff = float(np.sqrt(max(float(np.real(eps_value)), 1e-12)))
    except Exception:
        target_neff = 2.4
    return td.ModeSpec(num_modes=1, target_neff=target_neff)


def _source_size_y(device: DeviceDesignSpec, ports) -> float:
    if len(ports) <= 1:
        return float(device.mode_window_um or max(1.5 * device.waveguide_width_um, device.waveguide_width_um + 0.4))
    centers = np.sort(np.asarray([port.y_um for port in ports], dtype=np.float64))
    return float(max(1e-6, np.min(np.diff(centers)) - 1e-3))


def _monitor_bank(td, device: DeviceDesignSpec, input_ports, output_ports, x_positions: dict[str, float], mode_spec):
    monitors = []
    input_width = _source_size_y(device, input_ports)
    output_width = _source_size_y(device, output_ports)
    freq0 = float(td.C_0 / device.wavelength_um)
    if bool(device.include_input_monitors):
        for index, port in enumerate(input_ports):
            monitors.append(
                td.ModeMonitor(
                    center=(float(x_positions["x_in_monitor"]), float(port.y_um), 0.0),
                    size=(0.0, input_width, td.inf),
                    freqs=[freq0],
                    mode_spec=mode_spec,
                    name=f"in_mode_{index}",
                )
            )
    for index, port in enumerate(output_ports):
        monitors.append(
            td.ModeMonitor(
                center=(float(x_positions["x_out_monitor"]), float(port.y_um), 0.0),
                size=(0.0, output_width, td.inf),
                freqs=[freq0],
                mode_spec=mode_spec,
                name=f"out_mode_{index}",
            )
        )
    return monitors


def _output_monitor_names(counts: PortCounts) -> tuple[str, ...]:
    return tuple(f"out_mode_{index}" for index in range(counts.n_output))


def _mesh_override_structures(td, template) -> list[Any]:
    override_structure_dl = template.topology.override_structure_dl
    if override_structure_dl in (None, False):
        return []
    dl = float(template.device.pixel_size_um if override_structure_dl is True else override_structure_dl)
    if dl <= 0.0:
        return []
    return [
        td.MeshOverrideStructure(
            geometry=template.design_region.geometry.bounding_box,
            dl=(dl, dl, None),
            enforce=True,
        )
    ]


def _fixed_design_structures(td, device: DeviceDesignSpec, medium_core) -> list[Any]:
    curved_spec = _resolve_curved_container_spec(device)
    if not bool(curved_spec["enabled"]) or float(curved_spec["box_thickness_um"]) <= 0.0:
        return []
    geometry = _make_curved_box_geometry(
        td,
        device=device,
        curved_spec=curved_spec,
    )
    return [td.Structure(geometry=geometry, medium=medium_core)]


def _resolve_curved_container_spec(device: DeviceDesignSpec) -> dict[str, float | int | bool]:
    curved = device.curved_container
    enabled = bool(curved.enabled)
    pixel_size_um = float(device.pixel_size_um)
    guided_wavelength_um = float(curved.guided_wavelength_um or device.wavelength_um / max(np.sqrt(float(device.core_eps)), 1e-12))
    corner_radius_um = (
        float(curved.corner_radius_um)
        if curved.corner_radius_um is not None
        else float(curved.corner_radius_guided_wavelengths or 0.0) * guided_wavelength_um
    )
    box_thickness_um = (
        float(curved.box_thickness_um)
        if curved.box_thickness_um is not None
        else float(curved.box_thickness_guided_wavelengths or 0.0) * guided_wavelength_um
    )
    inner_overlap_px = int(curved.inner_overlap_px)
    taper_overlap_px = int(curved.taper_overlap_px)
    inner_overlap_um = float(inner_overlap_px) * pixel_size_um
    box_thickness_um = _snap_to_grid(box_thickness_um, pixel_size_um)
    inner_length_um = _snap_to_grid(float(device.design_region_size_um[0]), pixel_size_um)
    inner_width_um = _snap_to_grid(float(device.design_region_size_um[1]), pixel_size_um)
    inner_radius_um = max(
        0.0,
        min(
            corner_radius_um,
            0.5 * inner_length_um - 1e-6,
            0.5 * inner_width_um - 1e-6,
        ),
    )
    outer_length_um = _snap_to_grid(inner_length_um + 2.0 * box_thickness_um, pixel_size_um)
    outer_width_um = _snap_to_grid(inner_width_um + 2.0 * box_thickness_um, pixel_size_um)
    outer_radius_um = max(
        0.0,
        min(
            inner_radius_um + box_thickness_um,
            0.5 * outer_length_um - 1e-6,
            0.5 * outer_width_um - 1e-6,
        ),
    )
    box_inner_length_um = _snap_to_grid(max(0.0, inner_length_um - 2.0 * inner_overlap_um), pixel_size_um)
    box_inner_width_um = _snap_to_grid(max(0.0, inner_width_um - 2.0 * inner_overlap_um), pixel_size_um)
    box_inner_radius_um = max(
        0.0,
        min(
            inner_radius_um - inner_overlap_um,
            0.5 * box_inner_length_um - 1e-6,
            0.5 * box_inner_width_um - 1e-6,
        ),
    )
    return {
        "enabled": enabled,
        "pixel_size_um": pixel_size_um,
        "corner_radius_um": float(inner_radius_um),
        "box_thickness_um": float(box_thickness_um),
        "inner_overlap_px": int(inner_overlap_px),
        "taper_overlap_px": int(taper_overlap_px),
        "inner_length_um": float(inner_length_um),
        "inner_width_um": float(inner_width_um),
        "outer_length_um": float(outer_length_um),
        "outer_width_um": float(outer_width_um),
        "outer_radius_um": float(outer_radius_um),
        "box_inner_length_um": float(box_inner_length_um),
        "box_inner_width_um": float(box_inner_width_um),
        "box_inner_radius_um": float(box_inner_radius_um),
    }


def _snap_to_grid(value_um: float, grid_um: float) -> float:
    if float(grid_um) <= 0.0:
        return float(value_um)
    return float(round(float(value_um) / float(grid_um)) * float(grid_um))


def _grid_coords(*, shape: tuple[int, int], length_um: float, width_um: float, pixel_size_um: float) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = int(shape[0]), int(shape[1])
    xs = np.linspace(-0.5 * float(length_um) + 0.5 * float(pixel_size_um), 0.5 * float(length_um) - 0.5 * float(pixel_size_um), nx)
    ys = np.linspace(-0.5 * float(width_um) + 0.5 * float(pixel_size_um), 0.5 * float(width_um) - 0.5 * float(pixel_size_um), ny)
    return xs, ys


def _rounded_rect_mask(*, xs: np.ndarray, ys: np.ndarray, length_um: float, width_um: float, radius_um: float) -> np.ndarray:
    xx, yy = np.meshgrid(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), indexing="ij")
    radius = float(max(0.0, radius_um))
    if radius <= 0.0:
        return ((np.abs(xx) <= 0.5 * float(length_um)) & (np.abs(yy) <= 0.5 * float(width_um))).astype(np.float64)
    hx = max(1e-9, 0.5 * float(length_um) - radius)
    hy = max(1e-9, 0.5 * float(width_um) - radius)
    qx = np.abs(xx) - hx
    qy = np.abs(yy) - hy
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    signed_distance = outside + inside - radius
    return (signed_distance <= 0.0).astype(np.float64)


def _install_curved_topology_geometry_patch(td, tdi) -> None:
    if getattr(tdi.TopologyDesignRegion, "_lumix_curved_patch", False):
        return

    def _geometry(self):
        attrs = getattr(self, "attrs", None)
        spec = dict(attrs.get("curved_geometry_spec", {}) or {}) if isinstance(attrs, dict) else {}
        if bool(spec.get("enabled", False)):
            return _make_curved_container_geometry(
                td,
                length_um=float(self.size[0]),
                width_um=float(self.size[1]),
                thickness_um=float(self.size[2]),
                radius_um=float(spec.get("corner_radius_um", 0.0)),
            )
        return td.Box(center=self.center, size=self.size)

    tdi.TopologyDesignRegion.geometry = property(_geometry)
    tdi.TopologyDesignRegion._lumix_curved_patch = True


def _make_curved_container_geometry(td, *, length_um: float, width_um: float, thickness_um: float, radius_um: float):
    import gdstk

    library = gdstk.Library()
    cell = library.new_cell("LUMIX_CURVED_CONTAINER")
    polygon = gdstk.rectangle(
        (-0.5 * float(length_um), -0.5 * float(width_um)),
        (0.5 * float(length_um), 0.5 * float(width_um)),
        layer=1,
        datatype=0,
    )
    if float(radius_um) > 0.0:
        polygon.fillet(float(radius_um))
    cell.add(polygon)
    return td.Geometry.from_gds(
        cell,
        gds_layer=1,
        gds_dtype=0,
        axis=2,
        slab_bounds=(-0.5 * float(thickness_um), 0.5 * float(thickness_um)),
        reference_plane="middle",
    )


def _make_curved_box_geometry(td, *, device: DeviceDesignSpec, curved_spec: dict[str, float | int | bool]):
    import gdstk

    library = gdstk.Library()
    cell = library.new_cell("LUMIX_CURVED_BOX_RING")
    outer = gdstk.rectangle(
        (-0.5 * float(curved_spec["outer_length_um"]), -0.5 * float(curved_spec["outer_width_um"])),
        (0.5 * float(curved_spec["outer_length_um"]), 0.5 * float(curved_spec["outer_width_um"])),
        layer=2,
        datatype=0,
    )
    inner = gdstk.rectangle(
        (-0.5 * float(curved_spec["box_inner_length_um"]), -0.5 * float(curved_spec["box_inner_width_um"])),
        (0.5 * float(curved_spec["box_inner_length_um"]), 0.5 * float(curved_spec["box_inner_width_um"])),
        layer=2,
        datatype=0,
    )
    if float(curved_spec["outer_radius_um"]) > 0.0:
        outer.fillet(float(curved_spec["outer_radius_um"]))
    if float(curved_spec["box_inner_radius_um"]) > 0.0:
        inner.fillet(float(curved_spec["box_inner_radius_um"]))
    for polygon in gdstk.boolean([outer], [inner], "not", layer=2, datatype=0):
        cell.add(polygon)
    return td.Geometry.from_gds(
        cell,
        gds_layer=2,
        gds_dtype=0,
        axis=2,
        slab_bounds=(-0.5 * float(device.core_thickness_um), 0.5 * float(device.core_thickness_um)),
        reference_plane="middle",
    )
