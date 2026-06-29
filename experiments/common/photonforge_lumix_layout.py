from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import photonforge as pf
import tidy3d as td


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORNERSTONE_STRIP_90_BEND_OFFSET = 5.0

CORNERSTONE_PDK_COMPONENT_FILES: dict[str, str] = {
    "strip_1550_grating": "SOI220nm_1550nm_TE_STRIP_Grating_Coupler",
    "strip_1550_mmi_1x2": "SOI220nm_1550nm_TE_STRIP_2x1_MMI",
    "strip_1550_bend_90": "SOI220nm_1550nm_TE_STRIP_90_Degree_Bend",
    "topm_heater": "Heater",
}

CORNERSTONE_PDK_COMPONENT_METADATA: dict[str, dict[str, Any]] = {
    "strip_1550_grating": {
        "name": "SOI220nm_1550nm_TE_STRIP_Grating_Coupler",
        "ports": [
            {"name": "o1", "port_type": "optical", "center": [0.0, 0.0], "orientation": 180.0, "cross_section": "strip_1550nm"},
            {"name": "vertical_te", "port_type": "vertical_te", "center": [369.084, 0.0], "orientation": 0.0},
        ],
    },
    "strip_1550_mmi_1x2": {
        "name": "SOI220nm_1550nm_TE_STRIP_2x1_MMI",
        "ports": [
            {"name": "o1", "port_type": "optical", "center": [-45.55, 0.0], "orientation": 180.0, "cross_section": "strip_1550nm"},
            {"name": "o2", "port_type": "optical", "center": [46.25, 1.57], "orientation": 0.0, "cross_section": "strip_1550nm"},
            {"name": "o3", "port_type": "optical", "center": [46.25, -1.57], "orientation": 0.0, "cross_section": "strip_1550nm"},
        ],
    },
    "strip_1550_bend_90": {
        "name": "SOI220nm_1550nm_TE_STRIP_90_Degree_Bend",
        "ports": [
            {"name": "o1", "port_type": "optical", "center": [0.225, 0.0], "orientation": 270.0, "cross_section": "strip_1550nm"},
            {"name": "o2", "port_type": "optical", "center": [5.225, 5.0], "orientation": 0.0, "cross_section": "strip_1550nm"},
        ],
    },
    "topm_heater": {
        "name": "Heater",
        "ports": [
            {"name": "e1", "port_type": "electrical_dc", "center": [-50.086, 93.826], "orientation": 90.0, "cross_section": "dc"},
            {"name": "e2", "port_type": "electrical_dc", "center": [49.914, 93.826], "orientation": 90.0, "cross_section": "dc"},
        ],
    },
}


CORNERSTONE_ACTIVE_LAYERS: dict[str, tuple[int, int]] = {
    "grating_duv": (6, 0),
    "wg_lf": (3, 0),
    "wg_df": (4, 0),
    "rib_slab": (5, 0),
    "P_Implant_Low_DF": (7, 0),
    "N_Implant_Low_DF": (8, 0),
    "P_Implant_Hi_DF": (9, 0),
    "N_Implant_Hi_DF": (11, 0),
    "Via_DF": (12, 0),
    "Electrode_LF": (13, 0),
    "HEATER": (39, 0),
    "PAD": (41, 0),
    "Floorplan": (99, 0),
    "Label_Etch_DF": (100, 0),
}


@dataclass(frozen=True)
class LumixLayoutConfig:
    width: int = 16
    layers: int = 3
    internal_pitch: float = 250.0
    grating_pitch: float = 250.0
    waveguide_width: float = 0.45
    bend_radius: float = CORNERSTONE_STRIP_90_BEND_OFFSET
    grating_length: float = 392.0
    grating_width: float = 11.0
    modulator_length: float = 260.0
    modulator_slab_width: float = 235.004
    splitter_length: float = 91.8
    splitter_width: float = 6.0
    splitter_output_pitch: float = 3.14
    splitter_stage_gap: float = 80.0
    splitter_to_modulator_gap: float = 100.0
    inverse_design_length: float = 21.0
    inverse_design_width: float = 21.0
    inverse_design_port_pitch: float = 1.25
    port_taper_length: float = 3.1
    port_taper_mouth_width: float = 1.25
    port_taper_samples: int = 101
    input_fanout_length: float = 320.0
    output_fanout_length: float = 320.0
    inter_block_gap: float = 1500.0
    pad_clearance: float = 7.0
    die_length: float = 11470.0
    die_width: float = 4900.0


@dataclass(frozen=True)
class RouteRecord:
    name: str
    group: str
    component: pf.Component
    points: tuple[tuple[float, float], ...]
    length_um: float
    bend_radius_um: float
    bend_count: int
    straight_count: int


@dataclass(frozen=True)
class LumixLayout:
    component: pf.Component
    technology: pf.Technology
    config: LumixLayoutConfig
    summary: dict[str, Any]
    route_components: tuple[pf.Component, ...]
    route_records: tuple[RouteRecord, ...]


@dataclass(frozen=True)
class LumixLayoutArtifacts:
    output_dir: Path
    gds_path: Path
    oas_path: Path
    summary_path: Path
    connectivity_path: Path
    pdk_drc_summary_path: Path
    pdk_sparameter_manifest_path: Path | None
    preview_path: Path | None
    circuit_sweep_path: Path | None = None
    pdk_sparameter_sweep_path: Path | None = None


@dataclass(frozen=True)
class LumixCompactModelConfig:
    wavelength_um: float = 1.55
    effective_index: float = 2.4
    grating_coupling_loss_db: float = 3.0
    splitter_excess_loss_db: float = 0.2
    route_loss_db_per_cm: float = 1.0
    bend_loss_db: float = 0.02
    phase_modulator_vpi_v: float = 1.0
    phase_modulator_insertion_loss_db: float = 1.0
    phase_modulator_phase_offset_rad: float = 0.0
    inverse_design_default_loss_db: float = 0.0


STRICT_PDK_SPARAMETER_SOURCE_TYPES = frozenset({"pdk_measured", "measured", "em_derived"})

LUMIX_PDK_SPARAMETER_ROLES: dict[str, str] = {
    "grating_coupler": "Input/output grating-coupler S-parameters.",
    "splitter_1x2": "CORNERSTONE 1x2 MMI splitter S-parameters.",
    "phase_modulator": "Phase-modulator optical S-parameters at the modeled bias state.",
    "inverse_design_region": "Fabricated or EM-derived inverse-design-region S-parameters.",
    "strip_straight": "Strip-waveguide straight section S-parameters or calibrated per-length model data.",
    "strip_bend_90": "CORNERSTONE strip 90-degree bend S-parameters.",
}


class MissingPdkSParameterError(RuntimeError):
    """Raised when strict PDK/EM S-parameter data is missing or not trustworthy."""


@dataclass(frozen=True)
class PdkSParameterSpec:
    role: str
    source_type: str
    path: Path | None = None
    component: str | None = None
    description: str | None = None
    reference_length_um: float | None = None
    ports: tuple[str, ...] | None = None

    @classmethod
    def from_entry(cls, entry: dict[str, Any], *, base_dir: Path | None = None) -> "PdkSParameterSpec":
        if "role" not in entry:
            raise ValueError("S-parameter manifest entry is missing 'role'.")
        if "source_type" not in entry:
            raise ValueError(f"S-parameter manifest entry {entry['role']!r} is missing 'source_type'.")
        raw_path = entry.get("path")
        path = None
        if raw_path is not None:
            path = Path(raw_path)
            if not path.is_absolute() and base_dir is not None:
                path = base_dir / path
        return cls(
            role=str(entry["role"]),
            source_type=str(entry["source_type"]),
            path=path,
            component=str(entry["component"]) if entry.get("component") is not None else None,
            description=str(entry["description"]) if entry.get("description") is not None else None,
            reference_length_um=(
                float(entry["reference_length_um"]) if entry.get("reference_length_um") is not None else None
            ),
            ports=tuple(str(port) for port in entry["ports"]) if entry.get("ports") is not None else None,
        )

    @property
    def is_strict_source(self) -> bool:
        return self.source_type in STRICT_PDK_SPARAMETER_SOURCE_TYPES

    def as_report(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "source_type": self.source_type,
            "strict_source": self.is_strict_source,
            "path": str(self.path) if self.path is not None else None,
            "path_exists": bool(self.path is not None and self.path.exists()),
            "component": self.component,
            "description": self.description,
            "reference_length_um": self.reference_length_um,
            "ports": list(self.ports) if self.ports is not None else None,
        }


class PdkSParameterLibrary:
    """Manifest-backed library of measured or EM-derived component S-parameters."""

    def __init__(self, specs: dict[str, PdkSParameterSpec]) -> None:
        self._specs = dict(specs)

    @classmethod
    def from_entries(
        cls,
        entries: list[dict[str, Any] | PdkSParameterSpec],
        *,
        base_dir: Path | None = None,
    ) -> "PdkSParameterLibrary":
        specs: dict[str, PdkSParameterSpec] = {}
        for entry in entries:
            spec = entry if isinstance(entry, PdkSParameterSpec) else PdkSParameterSpec.from_entry(entry, base_dir=base_dir)
            specs[spec.role] = spec
        return cls(specs)

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> "PdkSParameterLibrary":
        payload = json.loads(Path(manifest_path).read_text())
        version = int(payload.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported S-parameter manifest version {version}.")
        return cls.from_entries(list(payload.get("models", [])), base_dir=Path(manifest_path).parent)

    def get(self, role: str) -> PdkSParameterSpec:
        try:
            return self._specs[role]
        except KeyError as exc:
            raise KeyError(f"No S-parameter model registered for role {role!r}.") from exc

    def data_model(self, role: str) -> pf.DataModel:
        s_array, frequencies, ports = self.load_sparameters(role)
        return pf.DataModel(s_array=s_array, frequencies=frequencies, ports=ports)

    def load_sparameters(self, role: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        spec = self.get(role)
        if spec.path is None:
            raise FileNotFoundError(f"S-parameter model {role!r} has no data path.")
        if not spec.path.exists():
            raise FileNotFoundError(f"S-parameter model {role!r} data file does not exist: {spec.path}")
        if _is_touchstone_path(spec.path):
            return _load_touchstone_sparameters(spec.path, ports=spec.ports)
        with np.load(spec.path, allow_pickle=False) as data:
            if "s_array" not in data:
                raise ValueError(f"S-parameter model {role!r} is missing 's_array'.")
            frequency_key = "frequencies_hz" if "frequencies_hz" in data else "frequencies"
            if frequency_key not in data:
                raise ValueError(f"S-parameter model {role!r} is missing 'frequencies_hz'.")
            if "ports" not in data:
                raise ValueError(f"S-parameter model {role!r} is missing 'ports'.")
            s_array = np.asarray(data["s_array"], dtype=np.complex128)
            frequencies = np.asarray(data[frequency_key], dtype=float)
            ports = [str(port) for port in np.asarray(data["ports"]).tolist()]
        if s_array.ndim != 3:
            raise ValueError(f"S-parameter model {role!r} s_array must have shape (frequency, port, port).")
        if s_array.shape[0] != frequencies.shape[0]:
            raise ValueError(
                f"S-parameter model {role!r} has {s_array.shape[0]} frequency slices "
                f"but {frequencies.shape[0]} frequencies."
            )
        if s_array.shape[1:] != (len(ports), len(ports)):
            raise ValueError(
                f"S-parameter model {role!r} has port axes {s_array.shape[1:]}, expected "
                f"{(len(ports), len(ports))} for ports {ports!r}."
            )
        return s_array, frequencies, ports

    def insertion_loss_db(self, role: str, *, port0: str = "P0", port1: str = "P1") -> float:
        s_array, _, ports = self.load_sparameters(role)
        try:
            index0 = ports.index(port0)
            index1 = ports.index(port1)
        except ValueError as exc:
            raise ValueError(f"S-parameter model {role!r} does not expose ports {port0!r}/{port1!r}.") from exc
        forward = np.asarray(s_array[:, index0, index1], dtype=np.complex128)
        backward = np.asarray(s_array[:, index1, index0], dtype=np.complex128)
        powers = np.concatenate([np.abs(forward) ** 2, np.abs(backward) ** 2])
        powers = powers[powers > 0.0]
        if powers.size == 0:
            return math.inf
        return _db_from_power_ratio(float(np.mean(powers)))

    def audit_for(self, layout: LumixLayout) -> dict[str, Any]:
        return audit_lumix_pdk_sparameter_library(layout, self)

    def assert_ready_for(self, layout: LumixLayout) -> None:
        report = self.audit_for(layout)
        if report["status"] == "ready":
            return
        problems: list[str] = []
        if report["missing_roles"]:
            problems.append("missing roles: " + ", ".join(report["missing_roles"]))
        if report["non_strict_roles"]:
            formatted = ", ".join(
                f"{role}={source_type}" for role, source_type in report["non_strict_roles"].items()
            )
            problems.append("non-PDK/EM source types: " + formatted)
        if report["missing_files"]:
            problems.append("missing files: " + ", ".join(report["missing_files"]))
        raise MissingPdkSParameterError("; ".join(problems))


def _is_touchstone_path(path: Path) -> bool:
    return re.fullmatch(r"\.s\d+p", path.suffix.lower()) is not None


def _load_touchstone_sparameters(
    path: Path,
    *,
    ports: tuple[str, ...] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    match = re.fullmatch(r"\.s(\d+)p", path.suffix.lower())
    if match is None:
        raise ValueError(f"Not a Touchstone path: {path}")
    port_count = int(match.group(1))
    port_names = list(ports) if ports is not None else [f"P{index + 1}" for index in range(port_count)]
    if len(port_names) != port_count:
        raise ValueError(
            f"Touchstone file {path} has {port_count} ports from extension, "
            f"but manifest supplied {len(port_names)} port names."
        )

    frequency_scale = 1e9
    data_format = "ma"
    values: list[float] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            frequency_scale, data_format = _parse_touchstone_option_line(line)
            continue
        values.extend(float(token) for token in line.split())

    values_per_frequency = 1 + 2 * port_count * port_count
    if len(values) % values_per_frequency != 0:
        raise ValueError(
            f"Touchstone file {path} has {len(values)} numeric values; expected a multiple of "
            f"{values_per_frequency} for {port_count} ports."
        )
    frequency_count = len(values) // values_per_frequency
    frequencies = np.zeros((frequency_count,), dtype=float)
    s_array = np.zeros((frequency_count, port_count, port_count), dtype=np.complex128)
    order = _touchstone_parameter_order(port_count)
    for frequency_index in range(frequency_count):
        offset = frequency_index * values_per_frequency
        frequencies[frequency_index] = float(values[offset]) * frequency_scale
        pair_values = values[offset + 1 : offset + values_per_frequency]
        for parameter_index, (output_port, input_port) in enumerate(order):
            first = pair_values[2 * parameter_index]
            second = pair_values[2 * parameter_index + 1]
            value = _touchstone_complex_value(first, second, data_format)
            s_array[frequency_index, input_port, output_port] = value
    return s_array, frequencies, port_names


def _parse_touchstone_option_line(line: str) -> tuple[float, str]:
    tokens = line[1:].strip().lower().split()
    frequency_units = {
        "hz": 1.0,
        "khz": 1e3,
        "mhz": 1e6,
        "ghz": 1e9,
        "thz": 1e12,
    }
    frequency_scale = 1e9
    data_format = "ma"
    if tokens and tokens[0] in frequency_units:
        frequency_scale = frequency_units[tokens[0]]
    if "s" in tokens:
        pass
    for token in tokens:
        if token in {"ri", "ma", "db"}:
            data_format = token
            break
    return frequency_scale, data_format


def _touchstone_parameter_order(port_count: int) -> list[tuple[int, int]]:
    if port_count == 2:
        return [(0, 0), (1, 0), (0, 1), (1, 1)]
    return [
        (output_port, input_port)
        for output_port in range(port_count)
        for input_port in range(port_count)
    ]


def _touchstone_complex_value(first: float, second: float, data_format: str) -> complex:
    if data_format == "ri":
        return complex(first, second)
    if data_format == "ma":
        return complex(first * np.exp(1j * np.deg2rad(second)))
    if data_format == "db":
        return complex((10 ** (first / 20.0)) * np.exp(1j * np.deg2rad(second)))
    raise ValueError(f"Unsupported Touchstone data format {data_format!r}.")


class VoltagePhaseModulatorModel(pf.Model):
    """Analytic two-port EO phase shifter with voltage-updatable phase."""

    def __init__(
        self,
        voltage_v: float = 0.0,
        vpi_v: float = 1.0,
        insertion_loss_db: float = 1.0,
        phase_offset_rad: float = 0.0,
        ports: tuple[str, str] = ("P0", "P1"),
    ) -> None:
        if float(vpi_v) == 0.0:
            raise ValueError("vpi_v must be nonzero.")
        super().__init__(
            voltage_v=float(voltage_v),
            vpi_v=float(vpi_v),
            insertion_loss_db=float(insertion_loss_db),
            phase_offset_rad=float(phase_offset_rad),
            ports=tuple(ports),
        )
        self.voltage_v = float(voltage_v)
        self.vpi_v = float(vpi_v)
        self.insertion_loss_db = float(insertion_loss_db)
        self.phase_offset_rad = float(phase_offset_rad)
        self.ports = tuple(ports)

    def __copy__(self) -> "VoltagePhaseModulatorModel":
        return type(self)(
            voltage_v=self.voltage_v,
            vpi_v=self.vpi_v,
            insertion_loss_db=self.insertion_loss_db,
            phase_offset_rad=self.phase_offset_rad,
            ports=self.ports,
        )

    def __deepcopy__(self, memo: dict | None = None) -> "VoltagePhaseModulatorModel":
        return self.__copy__()

    def __str__(self) -> str:
        return "VoltagePhaseModulatorModel"

    def __repr__(self) -> str:
        return (
            "VoltagePhaseModulatorModel("
            f"voltage_v={self.voltage_v!r}, "
            f"vpi_v={self.vpi_v!r}, "
            f"insertion_loss_db={self.insertion_loss_db!r}, "
            f"phase_offset_rad={self.phase_offset_rad!r}, "
            f"ports={self.ports!r})"
        )

    def start(self, component: pf.Component, frequencies: Any, **kwargs: Any) -> pf.ModelResult:
        frequency_array = np.asarray(frequencies, dtype=float).reshape(-1)
        if len(self.ports) != 2:
            raise RuntimeError("VoltagePhaseModulatorModel requires exactly two ports.")
        if not all(port_name in component.ports for port_name in self.ports):
            raise RuntimeError(
                f"VoltagePhaseModulatorModel ports {self.ports!r} are not present "
                f"on component {component.name!r}."
            )
        port0, port1 = self.ports
        component_ports = {
            port0: component.ports[port0].copy(True),
            port1: component.ports[port1].copy(True),
        }
        transmission = complex(
            eo_phase_transmission(
                self.voltage_v,
                vpi_v=self.vpi_v,
                insertion_loss_db=self.insertion_loss_db,
                phase_offset_rad=self.phase_offset_rad,
            )
        )
        values = np.full(frequency_array.shape, transmission, dtype=np.complex128)
        elements = {
            (f"{port0}@0", f"{port1}@0"): values,
            (f"{port1}@0", f"{port0}@0"): values,
        }
        return pf.ModelResult(pf.SMatrix(frequency_array, elements, component_ports))

    @property
    def as_bytes(self) -> bytes:
        return json.dumps(
            {
                "version": 1,
                "voltage_v": self.voltage_v,
                "vpi_v": self.vpi_v,
                "insertion_loss_db": self.insertion_loss_db,
                "phase_offset_rad": self.phase_offset_rad,
                "ports": list(self.ports),
            },
            sort_keys=True,
        ).encode("utf8")

    @classmethod
    def from_bytes(cls, byte_repr: bytes) -> "VoltagePhaseModulatorModel":
        payload = json.loads(byte_repr.decode("utf8"))
        if payload.get("version") != 1:
            raise RuntimeError("Unsupported VoltagePhaseModulatorModel version.")
        return cls(
            voltage_v=payload["voltage_v"],
            vpi_v=payload["vpi_v"],
            insertion_loss_db=payload["insertion_loss_db"],
            phase_offset_rad=payload["phase_offset_rad"],
            ports=tuple(payload["ports"]),
        )


pf.register_model_class(VoltagePhaseModulatorModel)


def cornerstone_soi220_active_technology() -> pf.Technology:
    layers = {
        "grating_duv": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["grating_duv"], "DUV grating etch", "#2f80ed18", "\\"),
        "wg_lf": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["wg_lf"], "Si waveguide LF protect", "#d6272818", "//"),
        "wg_df": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["wg_df"], "Si waveguide DF etch", "#ff989618", "."),
        "rib_slab": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["rib_slab"], "Rib slab protect", "#2ca02c18", "xx"),
        "P_Implant_Low_DF": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["P_Implant_Low_DF"], "Low-dose P implant", "#9467bd18", "\\"),
        "N_Implant_Low_DF": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["N_Implant_Low_DF"], "Low-dose N implant", "#17becf18", "/"),
        "P_Implant_Hi_DF": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["P_Implant_Hi_DF"], "High-dose P implant", "#8c564b18", "x"),
        "N_Implant_Hi_DF": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["N_Implant_Hi_DF"], "High-dose N implant", "#1f77b418", "+"),
        "Via_DF": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["Via_DF"], "Ohmic contact via", "#7f7f7f40", ":"),
        "Electrode_LF": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["Electrode_LF"], "Metal electrode", "#bcbd2240", "//"),
        "HEATER": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["HEATER"], "cspdk TiN heater", "#ebc63440", "//"),
        "PAD": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["PAD"], "cspdk metal pad", "#00808040", "\\"),
        "Floorplan": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["Floorplan"], "Cell outline", "#00000010", "-"),
        "Label_Etch_DF": pf.LayerSpec(CORNERSTONE_ACTIVE_LAYERS["Label_Etch_DF"], "Etched label", "#11111118", "."),
    }
    oxide = td.Medium(permittivity=1.44**2)
    silicon = td.Medium(permittivity=3.48**2)
    metal = td.PECMedium()
    extrusion_specs = [
        pf.ExtrusionSpec(pf.MaskSpec(), oxide, (-3.0, 2.5)),
        pf.ExtrusionSpec(pf.MaskSpec(CORNERSTONE_ACTIVE_LAYERS["wg_lf"]), silicon, (0.0, 0.22)),
        pf.ExtrusionSpec(pf.MaskSpec(CORNERSTONE_ACTIVE_LAYERS["rib_slab"]), silicon, (0.0, 0.10)),
        pf.ExtrusionSpec(pf.MaskSpec(CORNERSTONE_ACTIVE_LAYERS["Electrode_LF"]), metal, (2.0, 3.6)),
        pf.ExtrusionSpec(pf.MaskSpec(CORNERSTONE_ACTIVE_LAYERS["HEATER"]), metal, (1.1, 1.25)),
        pf.ExtrusionSpec(pf.MaskSpec(CORNERSTONE_ACTIVE_LAYERS["PAD"]), metal, (1.32, 1.54)),
    ]
    ports = {
        "strip_1550nm": pf.PortSpec(
            "CORNERSTONE SOI 220 nm strip, 1550 nm",
            0.45,
            (-0.6, 0.9),
            polarization="TE",
            target_neff=2.4,
            path_profiles=[(0.45, 0.0, CORNERSTONE_ACTIVE_LAYERS["wg_lf"])],
            default_radius=10.0,
        ),
        "vertical_te_1550nm": pf.PortSpec(
            "CORNERSTONE vertical TE fiber marker, 1550 nm",
            10.0,
            (-3.0, 3.0),
            polarization="TE",
            path_profiles=[(10.0, 0.0, CORNERSTONE_ACTIVE_LAYERS["grating_duv"])],
            default_radius=0.0,
        ),
    }
    return pf.Technology("CORNERSTONE Si 220 nm active", "layout-placeholder-1", layers, extrusion_specs, ports, oxide)


def _strip_port_spec(technology: pf.Technology) -> pf.PortSpec:
    return technology.ports["strip_1550nm"]


def _cornerstone_pdk_component(
    component_key: str,
    *,
    port_aliases: dict[str, str],
) -> pf.Component:
    metadata, gds_path = _cornerstone_pdk_component_data(component_key)
    tech = pf.config.default_technology
    component = pf.Component(f"PDK {metadata['name']}", technology=tech)
    _add_pdk_gds_polygons(component, gds_path)
    ports_by_name = {str(port["name"]): port for port in metadata.get("ports", [])}
    for alias, pdk_name in port_aliases.items():
        port_metadata = ports_by_name[pdk_name]
        spec = _pdk_port_spec(tech, port_metadata)
        center = tuple(float(value) for value in port_metadata["center"])
        outward_orientation = float(port_metadata["orientation"])
        component.add_port(
            pf.Port(center, (outward_orientation + 180.0) % 360.0, spec),
            port_name=alias,
        )
    return component


def _cornerstone_pdk_component_data(component_key: str) -> tuple[dict[str, Any], Path]:
    component_name = CORNERSTONE_PDK_COMPONENT_FILES[component_key]
    gds_dir = _cornerstone_pdk_gds_dir()
    gds_path = gds_dir / f"{component_name}.gds"
    if not gds_path.exists():
        raise FileNotFoundError(f"Missing cspdk GDS component {component_name} in {gds_dir}.")
    return CORNERSTONE_PDK_COMPONENT_METADATA[component_key], gds_path


def _cornerstone_pdk_gds_dir() -> Path:
    env_root = os.environ.get("CSPDK_ROOT") or os.environ.get("CORNERSTONE_PDK_ROOT")
    candidates = [
        _installed_cspdk_gds_dir(),
        Path(env_root) / "cspdk" / "si220" / "cband" / "gds" if env_root else None,
        Path(env_root) / "si220" / "cband" / "gds" if env_root else None,
        PROJECT_ROOT / "experiments" / "vendor" / "cspdk" / "cspdk" / "si220" / "cband" / "gds",
        PROJECT_ROOT.parent / "cspdk" / "cspdk" / "si220" / "cband" / "gds",
        Path("/tmp/cspdk/cspdk/si220/cband/gds"),
    ]
    for gds_dir in candidates:
        if gds_dir is None:
            continue
        if gds_dir.exists():
            return gds_dir
    raise FileNotFoundError(
        "cspdk C-band GDS directory not found. Install with `uv pip install cspdk --upgrade` "
        "or set CSPDK_ROOT to a checkout of https://github.com/gdsfactory/cspdk."
    )


def _installed_cspdk_gds_dir() -> Path | None:
    try:
        from cspdk.si220.cband.config import PATH as CSPDK_PATH
    except ImportError:
        return None
    return Path(CSPDK_PATH.gds)


def _add_pdk_gds_polygons(
    component: pf.Component,
    gds_path: Path,
    *,
    transform: Any | None = None,
) -> None:
    import gdstk

    library = gdstk.read_gds(gds_path)
    top_cells = library.top_level()
    if len(top_cells) != 1:
        raise ValueError(f"Expected one top-level cell in {gds_path}, found {len(top_cells)}.")
    for polygon in top_cells[0].get_polygons(apply_repetitions=True):
        vertices = [(float(x), float(y)) for x, y in polygon.points]
        if transform is not None:
            vertices = [transform(vertex) for vertex in vertices]
        component.add((int(polygon.layer), int(polygon.datatype)), pf.Polygon(vertices))


def _pdk_port_spec(technology: pf.Technology, port_metadata: dict[str, Any]) -> pf.PortSpec:
    port_type = str(port_metadata.get("port_type", ""))
    cross_section = port_metadata.get("cross_section")
    if isinstance(cross_section, str) and cross_section in technology.ports:
        return technology.ports[cross_section]
    if port_type == "vertical_te":
        return technology.ports["vertical_te_1550nm"]
    raise ValueError(f"Unsupported PDK port metadata: {port_metadata}")


def _add_centered_pdk_heater(component: pf.Component, *, length: float) -> None:
    metadata, gds_path = _cornerstone_pdk_component_data("topm_heater")
    bounds = _pdk_gds_bounds(gds_path)
    lower, upper = bounds
    heater_width = float(upper[0] - lower[0])
    heater_height = float(upper[1] - lower[1])
    x_offset = 0.5 * (length - heater_width) - float(lower[0])
    y_offset = -0.5 * heater_height - float(lower[1])

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        return (point[0] + x_offset, point[1] + y_offset)

    _add_pdk_gds_polygons(component, gds_path, transform=transform)
    ports_by_name = {str(port["name"]): port for port in metadata.get("ports", [])}
    for terminal_name, pdk_name in (("SIG", "e1"), ("GND", "e2")):
        port_metadata = ports_by_name[pdk_name]
        center = transform(tuple(float(value) for value in port_metadata["center"]))
        terminal = pf.Terminal(
            "PAD",
            pf.Rectangle(center=center, size=(80.0, 20.0)),
        )
        component.add_terminal(terminal, terminal_name=terminal_name)


def _pdk_gds_bounds(gds_path: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    import gdstk

    library = gdstk.read_gds(gds_path)
    top_cells = library.top_level()
    if len(top_cells) != 1:
        raise ValueError(f"Expected one top-level cell in {gds_path}, found {len(top_cells)}.")
    lower, upper = top_cells[0].bounding_box()
    return (float(lower[0]), float(lower[1])), (float(upper[0]), float(upper[1]))


@pf.parametric_component
def input_grating_coupler(
    *,
    length: float = 65.0,
    width: float = 18.0,
    port_spec: str = "strip_1550nm",
) -> pf.Component:
    return _cornerstone_pdk_component(
        "strip_1550_grating",
        port_aliases={"P0": "o1", "Fiber": "vertical_te"},
    )


@pf.parametric_component
def output_grating_coupler(
    *,
    length: float = 65.0,
    width: float = 18.0,
    port_spec: str = "strip_1550nm",
) -> pf.Component:
    return _cornerstone_pdk_component(
        "strip_1550_grating",
        port_aliases={"P0": "o1", "Fiber": "vertical_te"},
    )


@pf.parametric_component
def power_splitter_1x2(
    *,
    length: float = 55.0,
    width: float = 12.0,
    output_pitch: float = 10.0,
    port_spec: str = "strip_1550nm",
) -> pf.Component:
    return _cornerstone_pdk_component(
        "strip_1550_mmi_1x2",
        port_aliases={"P0": "o1", "P_HI": "o2", "P_LO": "o3"},
    )


@pf.parametric_component
def phase_modulator(
    *,
    length: float = 260.0,
    waveguide_width: float = 0.45,
    slab_width: float = 8.0,
    port_spec: str = "strip_1550nm",
) -> pf.Component:
    tech = pf.config.default_technology
    spec = tech.ports[port_spec]
    component = pf.Component("Lumix phase modulator with PDK TOPM heater", technology=tech)
    _add_centered_pdk_heater(component, length=float(length))
    component.add("wg_lf", pf.Rectangle(corner1=(-5.0, -waveguide_width / 2), corner2=(length + 5.0, waveguide_width / 2)))
    component.add_port(pf.Port((0.0, 0.0), 0, spec), port_name="P0")
    component.add_port(pf.Port((length, 0.0), 180, spec), port_name="P1")
    return component


@pf.parametric_component
def cornerstone_strip_90_bend(
    *,
    turn: float = 90.0,
    port_spec: str = "strip_1550nm",
) -> pf.Component:
    if float(turn) not in {-90.0, 90.0}:
        raise ValueError("The Cornerstone strip bend wrapper only supports +/-90 degree turns.")
    metadata, gds_path = _cornerstone_pdk_component_data("strip_1550_bend_90")
    tech = pf.config.default_technology
    spec = tech.ports[port_spec] if isinstance(port_spec, str) else port_spec
    component = pf.Component(f"PDK {metadata['name']} turn {turn:+.0f}", technology=tech)
    ports_by_name = {str(port["name"]): port for port in metadata.get("ports", [])}
    p0 = tuple(float(value) for value in ports_by_name["o1"]["center"])
    offset = CORNERSTONE_STRIP_90_BEND_OFFSET
    if float(turn) > 0.0:
        transform = lambda point: (point[1] - p0[1], point[0] - p0[0])
        p1 = (offset, offset)
        p1_direction = 270.0
    else:
        transform = lambda point: (point[1] - p0[1], -(point[0] - p0[0]))
        p1 = (offset, -offset)
        p1_direction = 90.0
    _add_pdk_gds_polygons(component, gds_path, transform=transform)
    component.add_port(pf.Port((0.0, 0.0), 0.0, spec), port_name="P0")
    component.add_port(pf.Port(p1, p1_direction, spec, inverted=True), port_name="P1")
    return component


@pf.parametric_component
def inverse_design_placeholder(
    *,
    width: int = 16,
    region_length: float = 21.0,
    region_width: float = 21.0,
    port_pitch: float = 1.25,
    taper_length: float = 3.1,
    taper_mouth_width: float = 1.25,
    taper_samples: int = 101,
    port_spec: str = "strip_1550nm",
) -> pf.Component:
    tech = pf.config.default_technology
    spec = tech.ports[port_spec]
    component = pf.Component(f"Lumix passive {width}x{width} ID placeholder", technology=tech)
    y_positions = _lane_positions(width, port_pitch)
    design_x0 = float(taper_length)
    design_x1 = float(taper_length + region_length)
    component_x1 = float(2 * taper_length + region_length)
    component.add("wg_lf", pf.Rectangle(corner1=(design_x0, -region_width / 2), corner2=(design_x1, region_width / 2)))
    for index, y in enumerate(y_positions):
        component.add(
            "wg_lf",
            _linear_taper_polygon(
                x0=0.0,
                x1=design_x0,
                y0=float(y),
                y1=float(y),
                w0=float(spec.width),
                w1=float(taper_mouth_width),
                samples=int(taper_samples),
            ),
        )
        component.add(
            "wg_lf",
            _linear_taper_polygon(
                x0=design_x1,
                x1=component_x1,
                y0=float(y),
                y1=float(y),
                w0=float(taper_mouth_width),
                w1=float(spec.width),
                samples=int(taper_samples),
            ),
        )
        component.add_port(pf.Port((0.0, y), 0, spec), port_name=f"W{index:02d}")
        component.add_port(pf.Port((component_x1, y), 180, spec), port_name=f"E{index:02d}")
    return component


def build_lumix_module(config: LumixLayoutConfig | None = None) -> LumixLayout:
    config = config or LumixLayoutConfig()
    tree_depth = _splitter_tree_depth(config.width)
    technology = cornerstone_soi220_active_technology()
    pf.config.default_technology = technology
    component = pf.Component(f"Lumix {config.layers}L x {config.width} circuit", technology=technology)

    input_grating = input_grating_coupler(length=config.grating_length, width=config.grating_width)
    output_grating = output_grating_coupler(length=config.grating_length, width=config.grating_width)
    splitter = power_splitter_1x2(
        length=config.splitter_length,
        width=config.splitter_width,
        output_pitch=config.splitter_output_pitch,
    )
    modulator = phase_modulator(
        length=config.modulator_length,
        waveguide_width=config.waveguide_width,
        slab_width=config.modulator_slab_width,
    )
    id_region = inverse_design_placeholder(
        width=config.width,
        region_length=config.inverse_design_length,
        region_width=config.inverse_design_width,
        port_pitch=config.inverse_design_port_pitch,
        taper_length=config.port_taper_length,
        taper_mouth_width=config.port_taper_mouth_width,
        taper_samples=config.port_taper_samples,
    )

    lane_y = _lane_positions(config.width, config.internal_pitch)
    id_y = _lane_positions(config.width, config.inverse_design_port_pitch)
    grating_y = _lane_positions(config.width, config.grating_pitch)
    input_ref = component.add_reference(input_grating)
    input_ref.rotate(180.0, center=(0.0, 0.0))
    input_ref.translate((0.0, 0.0))
    component.add_port(input_ref["Fiber"], port_name="fiber_in")
    output_refs: list[pf.Reference] = []
    splitter_refs: list[list[pf.Reference]] = []
    mod_refs: list[list[pf.Reference]] = []
    id_refs: list[pf.Reference] = []
    route_records: list[RouteRecord] = []

    splitter_stage_pitch = config.splitter_length + config.splitter_stage_gap
    x_splitter0 = config.grating_length + config.input_fanout_length
    for stage in range(tree_depth):
        stage_refs: list[pf.Reference] = []
        nodes = 2**stage
        group_size = config.width // nodes
        x_stage = x_splitter0 + stage * splitter_stage_pitch
        for node in range(nodes):
            first = node * group_size
            last = first + group_size - 1
            y = 0.5 * (lane_y[first] + lane_y[last])
            ref = component.add_reference(splitter)
            ref.translate((x_stage, y))
            stage_refs.append(ref)
        splitter_refs.append(stage_refs)

    x_mod0 = (
        x_splitter0
        + (tree_depth - 1) * splitter_stage_pitch
        + config.splitter_length
        + config.splitter_to_modulator_gap
    )
    id_component_length = config.inverse_design_length + 2 * config.port_taper_length
    bank_pitch_x = config.modulator_length + id_component_length + 2 * config.inter_block_gap
    for layer_index in range(config.layers):
        x_mod = x_mod0 + layer_index * bank_pitch_x
        layer_mod_refs: list[pf.Reference] = []
        for channel, y in enumerate(lane_y):
            ref = component.add_reference(modulator)
            ref.translate((x_mod, y))
            layer_mod_refs.append(ref)
            component.add_terminal(ref["SIG"], terminal_name=f"L{layer_index + 1}_CH{channel:02d}_SIG")
            component.add_terminal(ref["GND"], terminal_name=f"L{layer_index + 1}_CH{channel:02d}_GND")
        mod_refs.append(layer_mod_refs)

        id_ref = component.add_reference(id_region)
        id_ref.translate((x_mod + config.modulator_length + config.inter_block_gap, 0.0))
        id_refs.append(id_ref)

    x_output = (
        x_mod0
        + (config.layers - 1) * bank_pitch_x
        + config.modulator_length
        + config.inter_block_gap
        + id_component_length
        + config.output_fanout_length
    )
    for channel, y in enumerate(grating_y):
        ref = component.add_reference(output_grating)
        ref.translate((x_output, y))
        output_refs.append(ref)
        component.add_port(ref["Fiber"], port_name=f"fiber_out_{channel:02d}")

    route_records.extend(
        _add_splitter_tree_routes(
            component=component,
            input_ref=input_ref,
            splitter_refs=splitter_refs,
            mod_refs=mod_refs[0],
            technology=technology,
        )
    )

    for layer_index in range(config.layers):
        route_records.extend(
            _add_balanced_interface_routes(
                component=component,
                starts=[mod_refs[layer_index][channel]["P1"] for channel in range(config.width)],
                ends=[id_refs[layer_index][f"W{channel:02d}"] for channel in range(config.width)],
                technology=technology,
                group=f"L{layer_index + 1}_mod_to_id",
                fan="in",
            )
        )
        if layer_index < config.layers - 1:
            route_records.extend(
                _add_balanced_interface_routes(
                    component=component,
                    starts=[id_refs[layer_index][f"E{channel:02d}"] for channel in range(config.width)],
                    ends=[mod_refs[layer_index + 1][channel]["P0"] for channel in range(config.width)],
                    technology=technology,
                    group=f"L{layer_index + 1}_id_to_L{layer_index + 2}_mod",
                    fan="out",
                )
            )

    route_records.extend(
        _add_ordered_interface_routes(
            component=component,
            starts=[id_refs[-1][f"E{channel:02d}"] for channel in range(config.width)],
            ends=[output_refs[channel]["P0"] for channel in range(config.width)],
            technology=technology,
            group="output_fanout",
            fan="out",
        )
    )

    _add_floorplan(component, length=config.die_length, width=config.die_width)
    route_components = tuple(record.component for record in route_records)
    route_overlaps = find_route_overlaps(
        LumixLayout(
            component=component,
            technology=technology,
            config=config,
            summary={},
            route_components=route_components,
            route_records=tuple(route_records),
        )
    )
    diagonal_segments = find_diagonal_route_segments(tuple(route_records))
    phase_groups = _phase_balanced_group_report(tuple(route_records))
    route_bend_report = _route_bend_report(tuple(route_records))

    summary = {
        **asdict(config),
        "inverse_design_region_size_um": [float(config.inverse_design_length), float(config.inverse_design_width)],
        "inverse_design_component_size_um": [float(id_component_length), float(config.inverse_design_width)],
        "inverse_design_port_pitch_um": float(config.inverse_design_port_pitch),
        "port_taper_length_um": float(config.port_taper_length),
        "port_taper_mouth_width_um": float(config.port_taper_mouth_width),
        "input_gratings": 1,
        "output_gratings": config.width,
        "splitter_tree_depth": tree_depth,
        "power_splitters_1x2": 2**tree_depth - 1,
        "phase_modulators": config.width * config.layers,
        "inverse_design_regions": config.layers,
        "optical_routes": len(route_records),
        "route_overlap_count": len(route_overlaps),
        "diagonal_route_segment_count": len(diagonal_segments),
        "bend_aware_routes": route_bend_report["bend_aware_routes"],
        "route_bend_radius_um": float(config.bend_radius),
        "min_route_bend_radius_um": route_bend_report["min_bend_radius_um"],
        "total_route_bends": route_bend_report["total_bends"],
        "manual_path_routes": route_bend_report["manual_path_routes"],
        "phase_balanced_route_groups": phase_groups,
        "fiber_ports": len(component.ports),
        "electrical_terminals": len(component.terminals),
        "die_size_um": [float(config.die_length), float(config.die_width)],
        "bounds_um": _bounds_to_list(component.bounds()),
        "process": "CORNERSTONE Si_220nm_active",
        "cross_section": "strip_1550nm",
        "pdk_source": "gdsfactory/cspdk",
        "pdk_version": _cspdk_version(),
        "pdk_gds_dir": str(_cornerstone_pdk_gds_dir()),
        "pdk_components_used": {
            "input_grating": CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_grating"],
            "output_grating": CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_grating"],
            "splitter_1x2": CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_mmi_1x2"],
            "route_bend_90": CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_bend_90"],
            "phase_modulator_heater": CORNERSTONE_PDK_COMPONENT_FILES["topm_heater"],
        },
        "pdk_components_not_used": {
            "SOI220nm_1550nm_TE_MZI_Modulator": (
                "available in the PDK, but not used because it exposes vertical grating ports "
                "instead of compact in-line strip ports and does not match the repeated phase "
                "encoding topology."
            )
        },
    }
    return LumixLayout(
        component=component,
        technology=technology,
        config=config,
        summary=summary,
        route_components=route_components,
        route_records=tuple(route_records),
    )


def eo_phase_transmission(
    voltage_v: Any,
    *,
    vpi_v: float = 1.0,
    insertion_loss_db: float = 0.0,
    phase_offset_rad: float = 0.0,
) -> Any:
    if float(vpi_v) == 0.0:
        raise ValueError("vpi_v must be nonzero.")
    voltage = np.asarray(voltage_v, dtype=float)
    amplitude = _amplitude_from_loss_db(float(insertion_loss_db))
    phase = float(phase_offset_rad) + np.pi * voltage / float(vpi_v)
    transmission = amplitude * np.exp(1j * phase)
    return transmission.item() if transmission.shape == () else transmission


def attach_lumix_compact_models(
    layout: LumixLayout,
    compact_config: LumixCompactModelConfig | None = None,
    *,
    inverse_design_matrices: Any = None,
) -> dict[str, Any]:
    compact_config = compact_config or LumixCompactModelConfig()
    _set_compact_circuit_port_limits(layout)
    matrices, matrix_source = _normalize_inverse_design_matrices(
        inverse_design_matrices,
        width=layout.config.width,
        layers=layout.config.layers,
        default_loss_db=compact_config.inverse_design_default_loss_db,
    )
    frequency = _frequency_from_wavelength_um(compact_config.wavelength_um)
    reference_counts = {
        "grating_couplers": 0,
        "power_splitters_1x2": 0,
        "phase_modulators": 0,
        "inverse_design_regions": 0,
        "routes": len(layout.route_records),
    }
    attached_components: set[int] = set()
    inverse_design_reference_index = 0

    for record in layout.route_records:
        transmission = _route_transmission(record, compact_config)
        _replace_model(
            record.component,
            "Compact",
            pf.TwoPortModel(t=transmission, ports=("P0", "P1")),
        )

    for ref in layout.component.references:
        component = ref.component
        component_id = id(component)
        if _is_grating_component(component):
            reference_counts["grating_couplers"] += 1
            if component_id not in attached_components:
                _replace_model(
                    component,
                    "Compact",
                    pf.TwoPortModel(
                        t=_amplitude_from_loss_db(compact_config.grating_coupling_loss_db),
                        ports=("Fiber", "P0"),
                    ),
                )
                attached_components.add(component_id)
            continue
        if _is_splitter_component(component):
            reference_counts["power_splitters_1x2"] += 1
            if component_id not in attached_components:
                _replace_model(
                    component,
                    "Compact",
                    pf.PowerSplitterModel(
                        t=_amplitude_from_loss_db(compact_config.splitter_excess_loss_db) / math.sqrt(2.0),
                        ports=("P0", "P_HI", "P_LO"),
                    ),
                )
                attached_components.add(component_id)
            continue
        if _is_phase_modulator_component(component):
            reference_counts["phase_modulators"] += 1
            if component_id not in attached_components:
                _replace_model(
                    component,
                    "Compact",
                    VoltagePhaseModulatorModel(
                        voltage_v=0.0,
                        vpi_v=compact_config.phase_modulator_vpi_v,
                        insertion_loss_db=compact_config.phase_modulator_insertion_loss_db,
                        phase_offset_rad=compact_config.phase_modulator_phase_offset_rad,
                    ),
                )
                attached_components.add(component_id)
            continue
        if _is_inverse_design_component(component):
            reference_counts["inverse_design_regions"] += 1
            if component_id not in attached_components:
                matrix = matrices[min(inverse_design_reference_index, len(matrices) - 1)]
                _replace_model(
                    component,
                    "Compact",
                    _inverse_design_data_model(
                        matrix,
                        width=layout.config.width,
                        frequency=frequency,
                    ),
                )
                attached_components.add(component_id)
            inverse_design_reference_index += 1

    _replace_model(layout.component, "Circuit", pf.CircuitModel(verbose=False))
    report = {
        "top_level_model": "CircuitModel",
        "component_reference_counts": reference_counts,
        "wavelength_um": float(compact_config.wavelength_um),
        "frequency_hz": float(frequency),
        "effective_index": float(compact_config.effective_index),
        "inverse_design_matrix_source": matrix_source,
        "phase_modulator_model": "VoltagePhaseModulatorModel",
        "voltage_update_mode": "per_reference_model_updates",
        "port_model": "1d_compact_ports",
    }
    layout.summary["compact_models"] = report
    return report


def attach_lumix_pdk_sparameter_models(
    layout: LumixLayout,
    library: PdkSParameterLibrary,
) -> dict[str, Any]:
    library.assert_ready_for(layout)
    _set_compact_circuit_port_limits(layout)
    reference_counts = {
        "grating_couplers": 0,
        "power_splitters_1x2": 0,
        "phase_modulators": 0,
        "inverse_design_regions": 0,
        "routes": len(layout.route_records),
    }
    attached_components: set[int] = set()

    for record in layout.route_records:
        _replace_model(
            record.component,
            "PDK",
            _route_data_model_from_pdk_sparameters(record, library),
        )

    for ref in layout.component.references:
        component = ref.component
        component_id = id(component)
        role = None
        if _is_grating_component(component):
            reference_counts["grating_couplers"] += 1
            role = "grating_coupler"
        elif _is_splitter_component(component):
            reference_counts["power_splitters_1x2"] += 1
            role = "splitter_1x2"
        elif _is_phase_modulator_component(component):
            reference_counts["phase_modulators"] += 1
            role = "phase_modulator"
        elif _is_inverse_design_component(component):
            reference_counts["inverse_design_regions"] += 1
            role = "inverse_design_region"

        if role is not None and component_id not in attached_components:
            _replace_model(component, "PDK", library.data_model(role))
            attached_components.add(component_id)

    _replace_model(layout.component, "PDK", pf.CircuitModel(verbose=False))
    report = {
        "top_level_model": "CircuitModel",
        "model_source": "pdk_sparameters",
        "fallback_allowed": False,
        "component_reference_counts": reference_counts,
        "route_model": "cascaded_strip_straight_and_bend_sparameters",
        "block_models": {
            role: library.get(role).as_report()
            for role in LUMIX_PDK_SPARAMETER_ROLES
            if role in library._specs
        },
    }
    layout.summary["pdk_sparameter_models"] = report
    return report


def required_lumix_pdk_sparameter_roles(layout: LumixLayout) -> dict[str, dict[str, Any]]:
    splitter_stages = _splitter_tree_depth(layout.config.width)
    route_straights = int(sum(record.straight_count for record in layout.route_records))
    route_bends = int(sum(record.bend_count for record in layout.route_records))
    return {
        "grating_coupler": {
            "count": 1 + int(layout.config.width),
            "component": CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_grating"],
            "description": LUMIX_PDK_SPARAMETER_ROLES["grating_coupler"],
            "ports": ["Fiber", "P0"],
        },
        "splitter_1x2": {
            "count": 2**splitter_stages - 1,
            "component": CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_mmi_1x2"],
            "description": LUMIX_PDK_SPARAMETER_ROLES["splitter_1x2"],
            "ports": ["P0", "P_HI", "P_LO"],
        },
        "phase_modulator": {
            "count": int(layout.config.width * layout.config.layers),
            "component": "Lumix phase modulator with active phase-shifter geometry",
            "description": LUMIX_PDK_SPARAMETER_ROLES["phase_modulator"],
            "ports": ["P0", "P1"],
        },
        "inverse_design_region": {
            "count": int(layout.config.layers),
            "component": f"Lumix passive {layout.config.width}x{layout.config.width} inverse-design region",
            "description": LUMIX_PDK_SPARAMETER_ROLES["inverse_design_region"],
            "ports": _inverse_design_ports(layout.config.width),
        },
        "strip_straight": {
            "count": route_straights,
            "component": "CORNERSTONE SOI 220 nm strip waveguide",
            "description": LUMIX_PDK_SPARAMETER_ROLES["strip_straight"],
            "ports": ["P0", "P1"],
        },
        "strip_bend_90": {
            "count": route_bends,
            "component": CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_bend_90"],
            "description": LUMIX_PDK_SPARAMETER_ROLES["strip_bend_90"],
            "ports": ["P0", "P1"],
        },
    }


def audit_lumix_pdk_sparameter_library(
    layout: LumixLayout,
    library: PdkSParameterLibrary,
) -> dict[str, Any]:
    required_roles = required_lumix_pdk_sparameter_roles(layout)
    missing_roles: list[str] = []
    non_strict_roles: dict[str, str] = {}
    missing_files: list[str] = []
    models: dict[str, dict[str, Any]] = {}
    for role in required_roles:
        spec = library._specs.get(role)
        if spec is None:
            missing_roles.append(role)
            continue
        models[role] = spec.as_report()
        if not spec.is_strict_source:
            non_strict_roles[role] = spec.source_type
        if spec.path is None or not spec.path.exists():
            missing_files.append(f"{role}:{spec.path}")

    status = "ready"
    if missing_roles:
        status = "missing_models"
    elif non_strict_roles:
        status = "non_strict_sources"
    elif missing_files:
        status = "missing_files"

    return {
        "status": status,
        "strict_source_types": sorted(STRICT_PDK_SPARAMETER_SOURCE_TYPES),
        "required_roles": required_roles,
        "models": models,
        "missing_roles": missing_roles,
        "non_strict_roles": non_strict_roles,
        "missing_files": missing_files,
    }


def write_lumix_pdk_sparameter_manifest_template(
    layout: LumixLayout,
    manifest_path: Path,
) -> Path:
    required_roles = required_lumix_pdk_sparameter_roles(layout)
    models = []
    for role, requirement in required_roles.items():
        models.append(
            {
                "role": role,
                "source_type": "missing",
                "path": f"{role}.npz",
                "component": requirement["component"],
                "description": requirement["description"],
                "expected_count_in_layout": requirement["count"],
                "ports": requirement["ports"],
            }
        )
    payload = {
        "version": 1,
        "process": "CORNERSTONE Si_220nm_active",
        "pdk_source": "gdsfactory/cspdk",
        "pdk_version": layout.summary.get("pdk_version"),
        "wavelength_um": 1.55,
        "accepted_source_types": sorted(STRICT_PDK_SPARAMETER_SOURCE_TYPES),
        "models": models,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def pdk_route_loss_report(layout: LumixLayout, library: PdkSParameterLibrary) -> dict[str, Any]:
    straight_spec = _strict_pdk_role(library, "strip_straight")
    _strict_pdk_role(library, "strip_bend_90")
    if straight_spec.reference_length_um is None or straight_spec.reference_length_um <= 0.0:
        raise MissingPdkSParameterError("strip_straight requires a positive reference_length_um.")

    straight_loss_db = library.insertion_loss_db("strip_straight", port0="P0", port1="P1")
    bend_loss_db = library.insertion_loss_db("strip_bend_90", port0="P0", port1="P1")
    straight_loss_db_per_um = float(straight_loss_db / straight_spec.reference_length_um)

    route_reports = []
    grouped_losses: dict[str, list[float]] = {}
    for record in layout.route_records:
        bend_arc_length_um = float(record.bend_count) * 0.5 * math.pi * float(record.bend_radius_um)
        straight_length_um = max(float(record.length_um) - bend_arc_length_um, 0.0)
        loss_db = float(straight_length_um * straight_loss_db_per_um + int(record.bend_count) * bend_loss_db)
        grouped_losses.setdefault(record.group, []).append(loss_db)
        route_reports.append(
            {
                "name": record.name,
                "group": record.group,
                "length_um": float(record.length_um),
                "straight_length_um": straight_length_um,
                "bend_arc_length_um": bend_arc_length_um,
                "bend_count": int(record.bend_count),
                "loss_db": loss_db,
            }
        )

    route_losses = [route["loss_db"] for route in route_reports]
    groups = {}
    for group, losses in sorted(grouped_losses.items()):
        groups[group] = {
            "count": len(losses),
            "mean_loss_db": float(sum(losses) / len(losses)),
            "min_loss_db": float(min(losses)),
            "max_loss_db": float(max(losses)),
            "loss_spread_db": float(max(losses) - min(losses)),
        }

    return {
        "source": "pdk_sparameters",
        "strip_straight": {
            "source_type": straight_spec.source_type,
            "reference_length_um": float(straight_spec.reference_length_um),
            "loss_db_per_reference": float(straight_loss_db),
            "loss_db_per_um": straight_loss_db_per_um,
        },
        "strip_bend_90": {
            "source_type": library.get("strip_bend_90").source_type,
            "loss_db_per_bend": float(bend_loss_db),
        },
        "routes": route_reports,
        "groups": groups,
        "total_route_loss_db": float(sum(route_losses)) if route_losses else 0.0,
        "mean_route_loss_db": float(sum(route_losses) / len(route_losses)) if route_losses else 0.0,
        "min_route_loss_db": float(min(route_losses)) if route_losses else 0.0,
        "max_route_loss_db": float(max(route_losses)) if route_losses else 0.0,
        "loss_balance_objective_db": float(max(route_losses) - min(route_losses)) if route_losses else 0.0,
    }


def pdk_route_balance_objective(
    layout: LumixLayout,
    library: PdkSParameterLibrary,
    *,
    loss_tolerance_db: float = 1e-6,
    length_tolerance_um: float = 1e-6,
) -> dict[str, Any]:
    route_loss_report = pdk_route_loss_report(layout, library)
    routes_by_group: dict[str, list[dict[str, Any]]] = {}
    for route in route_loss_report["routes"]:
        routes_by_group.setdefault(str(route["group"]), []).append(route)

    phase_balanced_groups = set(layout.summary.get("phase_balanced_route_groups", {}))
    group_reports = {}
    phase_group_length_spreads = []
    phase_group_loss_spreads = []
    for group, routes in sorted(routes_by_group.items()):
        lengths = [float(route["length_um"]) for route in routes]
        losses = [float(route["loss_db"]) for route in routes]
        length_spread = float(max(lengths) - min(lengths)) if lengths else 0.0
        loss_spread = float(max(losses) - min(losses)) if losses else 0.0
        is_phase_balanced_group = group in phase_balanced_groups
        if is_phase_balanced_group:
            phase_group_length_spreads.append(length_spread)
            phase_group_loss_spreads.append(loss_spread)
        if is_phase_balanced_group and length_spread <= length_tolerance_um and loss_spread > loss_tolerance_db:
            status = "loss_balancing_required"
        elif is_phase_balanced_group and length_spread > length_tolerance_um:
            status = "phase_balancing_required"
        else:
            status = "balanced" if loss_spread <= loss_tolerance_db else "not_phase_constrained"
        group_reports[group] = {
            "count": len(routes),
            "phase_constrained": is_phase_balanced_group,
            "length_spread_um": length_spread,
            "loss_spread_db": loss_spread,
            "status": status,
        }

    return {
        "source": "pdk_sparameters",
        "phase_constrained_groups": sorted(phase_balanced_groups),
        "phase_balance_objective_um": float(max(phase_group_length_spreads)) if phase_group_length_spreads else 0.0,
        "loss_balance_objective_db": float(max(phase_group_loss_spreads)) if phase_group_loss_spreads else 0.0,
        "groups": group_reports,
    }


def pdk_block_loss_report(
    layout: LumixLayout,
    library: PdkSParameterLibrary,
    *,
    route_loss_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    splitter_total_power, splitter_output_powers = _role_total_transmission_power(
        library,
        "splitter_1x2",
        input_ports=["P0"],
        output_ports=["P_HI", "P_LO"],
    )
    inverse_total_power, _ = _role_total_transmission_power(
        library,
        "inverse_design_region",
        input_ports=[f"W{index:02d}" for index in range(layout.config.width)],
        output_ports=[f"E{index:02d}" for index in range(layout.config.width)],
    )
    route_loss_report = route_loss_report or pdk_route_loss_report(layout, library)
    return {
        "source": "pdk_sparameters",
        "grating_coupler": {
            "count": 1 + int(layout.config.width),
            "per_instance_loss_db": library.insertion_loss_db("grating_coupler", port0="Fiber", port1="P0"),
        },
        "splitter_1x2": {
            "count": 2 ** _splitter_tree_depth(layout.config.width) - 1,
            "total_output_power": float(splitter_total_power),
            "excess_loss_db": _db_from_power_ratio(float(splitter_total_power)),
            "output_imbalance_db": _output_uniformity_db([float(power) for power in splitter_output_powers]),
        },
        "phase_modulator": {
            "count": int(layout.config.width * layout.config.layers),
            "per_instance_loss_db": library.insertion_loss_db("phase_modulator", port0="P0", port1="P1"),
        },
        "inverse_design_region": {
            "count": int(layout.config.layers),
            "mean_output_power_per_input": float(inverse_total_power),
            "mean_input_loss_db": _db_from_power_ratio(float(inverse_total_power)),
        },
        "routes": route_loss_report,
    }


def run_lumix_pdk_sparameter_sweep(
    layout: LumixLayout,
    library: PdkSParameterLibrary,
    *,
    input_port: str = "fiber_in",
) -> dict[str, Any]:
    attach_report = attach_lumix_pdk_sparameter_models(layout, library)
    frequency = _pdk_library_frequency(library)
    output_ports = [f"fiber_out_{channel:02d}" for channel in range(layout.config.width)]
    s_matrix = layout.component.s_matrix([frequency], show_progress=False)
    output_fields = [
        complex(s_matrix.elements.get((f"{input_port}@0", f"{port}@0"), np.array([0.0j]))[0])
        for port in output_ports
    ]
    output_powers = [float(abs(value) ** 2) for value in output_fields]
    total_output_power = float(sum(output_powers))
    route_loss_report = pdk_route_loss_report(layout, library)
    return {
        "frequency_hz": float(frequency),
        "input_port": input_port,
        "output_ports": output_ports,
        "measurements": {
            "input_power_w": 1.0,
            "field_amplitude_normalization": "unit input field amplitude",
            "reported_powers": "|S_output,input|^2 for a 1 W normalized input",
        },
        "model_report": attach_report,
        "route_loss_report": route_loss_report,
        "route_balance_objective": pdk_route_balance_objective(layout, library),
        "block_loss_report": pdk_block_loss_report(layout, library, route_loss_report=route_loss_report),
        "points": [
            {
                "index": 0,
                "output_fields": [_complex_to_json(value) for value in output_fields],
                "output_powers": output_powers,
                "total_output_power": total_output_power,
                "total_insertion_loss_db": _db_from_power_ratio(total_output_power),
                "output_uniformity_db": _output_uniformity_db(output_powers),
                "s_parameters": _s_parameter_records(
                    source=input_port,
                    output_ports=output_ports,
                    output_fields=output_fields,
                ),
            }
        ],
    }


def write_lumix_pdk_sparameter_sweep(
    layout: LumixLayout,
    output_path: Path,
    library: PdkSParameterLibrary,
) -> dict[str, Any]:
    sweep = run_lumix_pdk_sparameter_sweep(layout, library)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sweep, indent=2) + "\n")
    return sweep


def _strict_pdk_role(library: PdkSParameterLibrary, role: str) -> PdkSParameterSpec:
    try:
        spec = library.get(role)
    except KeyError as exc:
        raise MissingPdkSParameterError(f"Missing required S-parameter role {role!r}.") from exc
    if not spec.is_strict_source:
        raise MissingPdkSParameterError(
            f"S-parameter role {role!r} uses source_type={spec.source_type!r}; "
            f"expected one of {sorted(STRICT_PDK_SPARAMETER_SOURCE_TYPES)!r}."
        )
    if spec.path is None or not spec.path.exists():
        raise MissingPdkSParameterError(f"S-parameter role {role!r} is missing a data file: {spec.path}")
    return spec


def _role_total_transmission_power(
    library: PdkSParameterLibrary,
    role: str,
    *,
    input_ports: list[str],
    output_ports: list[str],
) -> tuple[float, list[float]]:
    s_array, _, ports = library.load_sparameters(role)
    port_index = {port: index for index, port in enumerate(ports)}
    missing = [port for port in input_ports + output_ports if port not in port_index]
    if missing:
        raise MissingPdkSParameterError(f"S-parameter role {role!r} is missing ports: {missing!r}.")
    output_powers = []
    input_totals = []
    for input_port in input_ports:
        powers_for_input = []
        for output_port in output_ports:
            values = np.asarray(
                s_array[:, port_index[input_port], port_index[output_port]],
                dtype=np.complex128,
            )
            power = float(np.mean(np.abs(values) ** 2))
            output_powers.append(power)
            powers_for_input.append(power)
        input_totals.append(float(sum(powers_for_input)))
    return float(sum(input_totals) / len(input_totals)), output_powers


def _route_data_model_from_pdk_sparameters(
    record: RouteRecord,
    library: PdkSParameterLibrary,
) -> pf.DataModel:
    straight_spec = _strict_pdk_role(library, "strip_straight")
    _strict_pdk_role(library, "strip_bend_90")
    if straight_spec.reference_length_um is None or straight_spec.reference_length_um <= 0.0:
        raise MissingPdkSParameterError("strip_straight requires a positive reference_length_um.")

    straight_s, frequencies, straight_ports = library.load_sparameters("strip_straight")
    bend_s, bend_frequencies, bend_ports = library.load_sparameters("strip_bend_90")
    if frequencies.shape != bend_frequencies.shape or not np.allclose(frequencies, bend_frequencies):
        raise MissingPdkSParameterError("strip_straight and strip_bend_90 frequencies must match.")

    straight_t = _sparameter_complex_transmission(straight_s, straight_ports, "P0", "P1")
    bend_t = _sparameter_complex_transmission(bend_s, bend_ports, "P0", "P1")
    straight_scale = _route_straight_length_um(record) / float(straight_spec.reference_length_um)
    route_t = _scale_complex_transmission(straight_t, straight_scale) * np.power(bend_t, int(record.bend_count))
    s_array = np.zeros((len(frequencies), 2, 2), dtype=np.complex128)
    s_array[:, 0, 1] = route_t
    s_array[:, 1, 0] = route_t
    return pf.DataModel(s_array=s_array, frequencies=frequencies, ports=["P0", "P1"])


def _pdk_library_frequency(library: PdkSParameterLibrary) -> float:
    _, frequencies, _ = library.load_sparameters("grating_coupler")
    if frequencies.size == 0:
        raise MissingPdkSParameterError("grating_coupler S-parameter data has no frequencies.")
    return float(frequencies[0])


def _sparameter_complex_transmission(
    s_array: np.ndarray,
    ports: list[str],
    port0: str,
    port1: str,
) -> np.ndarray:
    try:
        index0 = ports.index(port0)
        index1 = ports.index(port1)
    except ValueError as exc:
        raise ValueError(f"S-parameter data does not expose ports {port0!r}/{port1!r}.") from exc
    forward = np.asarray(s_array[:, index0, index1], dtype=np.complex128)
    backward = np.asarray(s_array[:, index1, index0], dtype=np.complex128)
    use_backward = np.abs(forward) == 0.0
    return np.where(use_backward, backward, forward)


def _scale_complex_transmission(values: np.ndarray, scale: float) -> np.ndarray:
    magnitudes = np.power(np.abs(values), float(scale))
    phases = np.unwrap(np.angle(values)) * float(scale)
    return magnitudes * np.exp(1j * phases)


def _route_straight_length_um(record: RouteRecord) -> float:
    bend_arc_length_um = float(record.bend_count) * 0.5 * math.pi * float(record.bend_radius_um)
    return max(float(record.length_um) - bend_arc_length_um, 0.0)


def run_lumix_circuit_sweep(
    layout: LumixLayout,
    *,
    voltage_masks: Any,
    compact_config: LumixCompactModelConfig | None = None,
    inverse_design_matrices: Any = None,
    input_port: str = "fiber_in",
) -> dict[str, Any]:
    compact_config = compact_config or LumixCompactModelConfig()
    matrices, matrix_source = _normalize_inverse_design_matrices(
        inverse_design_matrices,
        width=layout.config.width,
        layers=layout.config.layers,
        default_loss_db=compact_config.inverse_design_default_loss_db,
    )
    attach_report = attach_lumix_compact_models(
        layout,
        compact_config,
        inverse_design_matrices=matrices,
    )
    masks = _normalize_voltage_masks(
        voltage_masks,
        layers=layout.config.layers,
        width=layout.config.width,
    )
    frequency = _frequency_from_wavelength_um(compact_config.wavelength_um)
    output_ports = [f"fiber_out_{channel:02d}" for channel in range(layout.config.width)]
    points: list[dict[str, Any]] = []
    modeled_losses = _modeled_loss_report(layout, compact_config)
    for point_index, voltages in enumerate(masks):
        updates = _circuit_model_updates(
            layout,
            voltages,
            matrices=matrices,
            compact_config=compact_config,
            frequency=frequency,
        )
        s_matrix = layout.component.s_matrix(
            [frequency],
            show_progress=False,
            model_kwargs={"updates": updates},
        )
        output_fields = [
            complex(s_matrix.elements.get((f"{input_port}@0", f"{port}@0"), np.array([0.0j]))[0])
            for port in output_ports
        ]
        output_powers = [float(abs(value) ** 2) for value in output_fields]
        total_output_power = float(sum(output_powers))
        points.append(
            {
                "index": point_index,
                "voltages": voltages.tolist(),
                "output_fields": [_complex_to_json(value) for value in output_fields],
                "output_powers": output_powers,
                "total_output_power": total_output_power,
                "total_insertion_loss_db": _db_from_power_ratio(total_output_power),
                "output_uniformity_db": _output_uniformity_db(output_powers),
                "s_parameters": _s_parameter_records(
                    source=input_port,
                    output_ports=output_ports,
                    output_fields=output_fields,
                ),
            }
        )
    return {
        "wavelength_um": float(compact_config.wavelength_um),
        "frequency_hz": float(frequency),
        "input_port": input_port,
        "output_ports": output_ports,
        "measurements": {
            "input_power_w": 1.0,
            "field_amplitude_normalization": "unit input field amplitude",
            "reported_powers": "|S_output,input|^2 for a 1 W normalized input",
        },
        "modeled_losses": modeled_losses,
        "model_report": attach_report,
        "inverse_design_matrix_source": matrix_source,
        "points": points,
    }


def write_lumix_circuit_sweep(
    layout: LumixLayout,
    output_path: Path,
    *,
    voltage_masks: Any,
    compact_config: LumixCompactModelConfig | None = None,
    inverse_design_matrices: Any = None,
) -> dict[str, Any]:
    sweep = run_lumix_circuit_sweep(
        layout,
        voltage_masks=voltage_masks,
        compact_config=compact_config,
        inverse_design_matrices=inverse_design_matrices,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sweep, indent=2) + "\n")
    return sweep


def _frequency_from_wavelength_um(wavelength_um: float) -> float:
    if float(wavelength_um) <= 0.0:
        raise ValueError("wavelength_um must be positive.")
    return float(pf.C_0 / float(wavelength_um))


def _amplitude_from_loss_db(loss_db: float) -> float:
    return float(10 ** (-float(loss_db) / 20.0))


def _route_transmission(record: RouteRecord, compact_config: LumixCompactModelConfig) -> complex:
    loss_db = (
        float(compact_config.route_loss_db_per_cm) * float(record.length_um) / 10_000.0
        + float(compact_config.bend_loss_db) * int(record.bend_count)
    )
    phase_rad = 2.0 * math.pi * float(compact_config.effective_index) * float(record.length_um) / float(
        compact_config.wavelength_um
    )
    return complex(_amplitude_from_loss_db(loss_db) * np.exp(1j * phase_rad))


def _route_loss_db(record: RouteRecord, compact_config: LumixCompactModelConfig) -> float:
    return float(
        float(compact_config.route_loss_db_per_cm) * float(record.length_um) / 10_000.0
        + float(compact_config.bend_loss_db) * int(record.bend_count)
    )


def _modeled_loss_report(layout: LumixLayout, compact_config: LumixCompactModelConfig) -> dict[str, Any]:
    route_losses = [_route_loss_db(record, compact_config) for record in layout.route_records]
    grouped_routes: dict[str, list[tuple[RouteRecord, float]]] = {}
    for record, loss_db in zip(layout.route_records, route_losses):
        grouped_routes.setdefault(record.group, []).append((record, loss_db))

    route_groups = {}
    for group, records in sorted(grouped_routes.items()):
        losses = [loss_db for _, loss_db in records]
        lengths = [record.length_um for record, _ in records]
        bends = [record.bend_count for record, _ in records]
        route_groups[group] = {
            "count": len(records),
            "total_length_um": float(sum(lengths)),
            "mean_length_um": float(sum(lengths) / len(lengths)),
            "total_bends": int(sum(bends)),
            "total_loss_db": float(sum(losses)),
            "mean_loss_db": float(sum(losses) / len(losses)),
            "max_loss_db": float(max(losses)),
            "min_loss_db": float(min(losses)),
        }

    splitter_stages = _splitter_tree_depth(layout.config.width)
    ideal_split_loss_db = 10.0 * math.log10(layout.config.width)
    grating_path_loss_db = 2.0 * float(compact_config.grating_coupling_loss_db)
    splitter_excess_path_loss_db = splitter_stages * float(compact_config.splitter_excess_loss_db)
    phase_path_loss_db = layout.config.layers * float(compact_config.phase_modulator_insertion_loss_db)
    inverse_design_path_loss_db = layout.config.layers * float(compact_config.inverse_design_default_loss_db)
    nominal_path_loss_without_routes_db = (
        grating_path_loss_db
        + ideal_split_loss_db
        + splitter_excess_path_loss_db
        + phase_path_loss_db
        + inverse_design_path_loss_db
    )
    return {
        "grating_couplers": {
            "count": 1 + int(layout.config.width),
            "per_device_loss_db": float(compact_config.grating_coupling_loss_db),
            "input_to_output_path_loss_db": grating_path_loss_db,
        },
        "splitter_tree": {
            "count": 2**splitter_stages - 1,
            "stages_per_path": splitter_stages,
            "excess_loss_db_per_stage": float(compact_config.splitter_excess_loss_db),
            "excess_loss_db_per_path": splitter_excess_path_loss_db,
            "ideal_power_division_db_per_output": ideal_split_loss_db,
        },
        "phase_modulators": {
            "count": int(layout.config.width * layout.config.layers),
            "layers_per_path": int(layout.config.layers),
            "per_device_loss_db": float(compact_config.phase_modulator_insertion_loss_db),
            "input_to_output_path_loss_db": phase_path_loss_db,
        },
        "inverse_design_regions": {
            "count": int(layout.config.layers),
            "per_region_default_loss_db": float(compact_config.inverse_design_default_loss_db),
            "input_to_output_path_loss_db": inverse_design_path_loss_db,
            "matrix_source": "identity" if compact_config.inverse_design_default_loss_db == 0.0 else "identity_with_loss",
        },
        "routes": {
            "count": len(layout.route_records),
            "total_layout_length_um": float(sum(record.length_um for record in layout.route_records)),
            "total_layout_bends": int(sum(record.bend_count for record in layout.route_records)),
            "loss_db_per_cm": float(compact_config.route_loss_db_per_cm),
            "bend_loss_db": float(compact_config.bend_loss_db),
            "total_layout_loss_db": float(sum(route_losses)),
            "mean_route_loss_db": float(sum(route_losses) / len(route_losses)) if route_losses else 0.0,
            "max_route_loss_db": float(max(route_losses)) if route_losses else 0.0,
            "groups": route_groups,
        },
        "nominal_single_output_path_loss_without_route_imbalance_db": float(nominal_path_loss_without_routes_db),
    }


def _db_from_power_ratio(power_ratio: float) -> float:
    if power_ratio <= 0.0:
        return math.inf
    return float(-10.0 * math.log10(power_ratio))


def _transmission_db(power_ratio: float) -> float:
    if power_ratio <= 0.0:
        return -math.inf
    return float(10.0 * math.log10(power_ratio))


def _output_uniformity_db(output_powers: list[float]) -> float:
    if not output_powers:
        return math.inf
    min_power = min(output_powers)
    max_power = max(output_powers)
    if min_power <= 0.0:
        return math.inf
    return float(10.0 * math.log10(max_power / min_power))


def _s_parameter_records(
    *,
    source: str,
    output_ports: list[str],
    output_fields: list[complex],
) -> list[dict[str, Any]]:
    records = []
    for output, field in zip(output_ports, output_fields):
        power = float(abs(field) ** 2)
        phase_rad = float(np.angle(field))
        records.append(
            {
                "source": source,
                "output": output,
                "field": _complex_to_json(field),
                "magnitude": float(abs(field)),
                "power": power,
                "phase_rad": phase_rad,
                "phase_deg": float(np.degrees(phase_rad)),
                "transmission_db": _transmission_db(power),
            }
        )
    return records


def _set_compact_circuit_port_limits(layout: LumixLayout) -> None:
    _set_component_ports_to_1d(layout.component)
    for ref in layout.component.references:
        _set_component_ports_to_1d(ref.component)
    for route in layout.route_components:
        _set_component_ports_to_1d(route)


def _set_component_ports_to_1d(component: pf.Component) -> None:
    for port in component.ports.values():
        if not hasattr(port, "spec"):
            continue
        if float(port.spec.limits[0]) == float(port.spec.limits[1]):
            continue
        port.spec = port.spec.copy()
        port.spec.limits = (0.0, 0.0)


def _replace_model(component: pf.Component, model_name: str, model: pf.Model) -> None:
    if model_name in component.models:
        component.remove_model(model_name)
    component.add_model(model, model_name=model_name, set_active=True)


def _is_grating_component(component: pf.Component) -> bool:
    return CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_grating"] in component.name


def _is_splitter_component(component: pf.Component) -> bool:
    return CORNERSTONE_PDK_COMPONENT_FILES["strip_1550_mmi_1x2"] in component.name


def _is_phase_modulator_component(component: pf.Component) -> bool:
    return component.name.startswith("Lumix phase modulator")


def _is_inverse_design_component(component: pf.Component) -> bool:
    return "ID placeholder" in component.name


def _normalize_inverse_design_matrices(
    inverse_design_matrices: Any,
    *,
    width: int,
    layers: int,
    default_loss_db: float,
) -> tuple[list[np.ndarray], str]:
    if inverse_design_matrices is None:
        matrix = _amplitude_from_loss_db(default_loss_db) * np.eye(width, dtype=np.complex128)
        return [matrix.copy() for _ in range(layers)], "identity"
    if isinstance(inverse_design_matrices, dict):
        raw_matrices = [inverse_design_matrices[name] for name in sorted(inverse_design_matrices)]
    else:
        raw_matrices = list(inverse_design_matrices)
    if len(raw_matrices) != layers:
        raise ValueError(f"Expected {layers} inverse-design matrices, got {len(raw_matrices)}.")
    matrices = [np.asarray(matrix, dtype=np.complex128) for matrix in raw_matrices]
    for index, matrix in enumerate(matrices):
        if matrix.shape != (width, width):
            raise ValueError(
                f"Inverse-design matrix {index} has shape {matrix.shape}, expected {(width, width)}."
            )
        singular_max = float(np.linalg.svd(matrix, compute_uv=False)[0])
        if singular_max > 1.0 + 1e-9:
            raise ValueError(
                f"Inverse-design matrix {index} is not passive; max singular value is {singular_max:.6f}."
            )
    return matrices, "provided"


def _inverse_design_data_model(
    matrix: np.ndarray,
    *,
    width: int,
    frequency: float,
) -> pf.DataModel:
    ports = _inverse_design_ports(width)
    return pf.DataModel(
        s_array=_inverse_design_s_array(matrix, width=width),
        frequencies=np.asarray([frequency], dtype=float),
        ports=ports,
    )


def _inverse_design_model_updates(
    matrix: np.ndarray,
    *,
    width: int,
    frequency: float,
) -> dict[str, Any]:
    return {
        "s_array": _inverse_design_s_array(matrix, width=width),
        "frequencies": np.asarray([frequency], dtype=float),
        "ports": _inverse_design_ports(width),
    }


def _inverse_design_ports(width: int) -> list[str]:
    return [f"W{index:02d}" for index in range(width)] + [f"E{index:02d}" for index in range(width)]


def _inverse_design_s_array(matrix: np.ndarray, *, width: int) -> np.ndarray:
    s_array = np.zeros((1, 2 * width, 2 * width), dtype=np.complex128)
    for output_channel in range(width):
        for input_channel in range(width):
            value = complex(matrix[output_channel, input_channel])
            s_array[0, width + output_channel, input_channel] = value
            s_array[0, input_channel, width + output_channel] = value
    return s_array


def _normalize_voltage_masks(voltage_masks: Any, *, layers: int, width: int) -> np.ndarray:
    masks = np.asarray(voltage_masks, dtype=float)
    if masks.shape == (layers, width):
        masks = masks.reshape(1, layers, width)
    if masks.ndim != 3 or masks.shape[1:] != (layers, width):
        raise ValueError(
            f"voltage_masks must have shape (sweep, {layers}, {width}) "
            f"or ({layers}, {width}); got {masks.shape}."
        )
    return masks


def _circuit_model_updates(
    layout: LumixLayout,
    voltages: np.ndarray,
    *,
    matrices: list[np.ndarray],
    compact_config: LumixCompactModelConfig,
    frequency: float,
) -> dict[tuple[Any, ...], dict[str, dict[str, Any]]]:
    updates: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    phase_modulator_name = "^Lumix phase modulator.*"
    inverse_design_name = f"^Lumix passive {layout.config.width}x{layout.config.width} ID placeholder$"
    for layer_index in range(layout.config.layers):
        for channel in range(layout.config.width):
            modulator_reference_index = layer_index * layout.config.width + channel
            updates[(phase_modulator_name, modulator_reference_index)] = {
                "model_updates": {
                    "voltage_v": float(voltages[layer_index, channel]),
                    "vpi_v": float(compact_config.phase_modulator_vpi_v),
                    "insertion_loss_db": float(compact_config.phase_modulator_insertion_loss_db),
                    "phase_offset_rad": float(compact_config.phase_modulator_phase_offset_rad),
                }
            }
        updates[(inverse_design_name, layer_index)] = {
            "model_updates": _inverse_design_model_updates(
                matrices[layer_index],
                width=layout.config.width,
                frequency=frequency,
            )
        }
    return updates


def _complex_to_json(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def _cspdk_version() -> str:
    try:
        from importlib.metadata import version

        return version("cspdk")
    except Exception:
        return "source-checkout"


def write_lumix_layout_artifacts(
    layout: LumixLayout,
    output_dir: Path,
    *,
    write_preview: bool = True,
    write_pdk_sparameter_manifest: bool = True,
    write_circuit_sweep: bool = False,
    write_pdk_sparameter_sweep: bool = False,
    circuit_voltage_masks: Any = None,
    compact_config: LumixCompactModelConfig | None = None,
    inverse_design_matrices: Any = None,
    pdk_sparameter_library: PdkSParameterLibrary | None = None,
) -> LumixLayoutArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"lumix_{layout.config.layers}layer_w{layout.config.width}_cornerstone_soi220"
    gds_path = output_dir / f"{stem}.gds"
    oas_path = output_dir / f"{stem}.oas"
    summary_path = output_dir / "layout_summary.json"
    connectivity_path = output_dir / "connectivity.json"
    pdk_drc_summary_path = output_dir / "cspdk_drc_summary.json"
    pdk_sparameter_manifest_path = output_dir / "pdk_sparameters_manifest.json"
    preview_path = output_dir / "layout_preview.png"
    circuit_sweep_path = output_dir / "circuit_sweep.json"
    pdk_sparameter_sweep_path = output_dir / "pdk_sparameter_sweep.json"

    layout.component.write_gds(gds_path)
    layout.component.write_oas(oas_path)
    generated_circuit_sweep_path = None
    if write_circuit_sweep:
        if circuit_voltage_masks is None:
            circuit_voltage_masks = np.zeros((1, layout.config.layers, layout.config.width), dtype=float)
        write_lumix_circuit_sweep(
            layout,
            circuit_sweep_path,
            voltage_masks=circuit_voltage_masks,
            compact_config=compact_config,
            inverse_design_matrices=inverse_design_matrices,
        )
        generated_circuit_sweep_path = circuit_sweep_path
    generated_pdk_sparameter_sweep_path = None
    if write_pdk_sparameter_sweep:
        if pdk_sparameter_library is None:
            raise MissingPdkSParameterError("write_pdk_sparameter_sweep requires pdk_sparameter_library.")
        write_lumix_pdk_sparameter_sweep(
            layout,
            pdk_sparameter_sweep_path,
            pdk_sparameter_library,
        )
        generated_pdk_sparameter_sweep_path = pdk_sparameter_sweep_path
    summary_path.write_text(json.dumps(layout.summary, indent=2) + "\n")
    connectivity_path.write_text(json.dumps(_netlist_report(layout.component), indent=2) + "\n")
    pdk_drc_summary_path.write_text(json.dumps(_cspdk_drc_report(layout), indent=2) + "\n")
    generated_pdk_sparameter_manifest_path = (
        write_lumix_pdk_sparameter_manifest_template(layout, pdk_sparameter_manifest_path)
        if write_pdk_sparameter_manifest
        else None
    )
    generated_preview = _write_preview_from_gds(gds_path, preview_path) if write_preview else None
    return LumixLayoutArtifacts(
        output_dir=output_dir,
        gds_path=gds_path,
        oas_path=oas_path,
        summary_path=summary_path,
        connectivity_path=connectivity_path,
        pdk_drc_summary_path=pdk_drc_summary_path,
        pdk_sparameter_manifest_path=generated_pdk_sparameter_manifest_path,
        preview_path=generated_preview,
        circuit_sweep_path=generated_circuit_sweep_path,
        pdk_sparameter_sweep_path=generated_pdk_sparameter_sweep_path,
    )


def _cspdk_drc_report(layout: LumixLayout) -> dict[str, Any]:
    local_deck = _find_cspdk_lydrc()
    return {
        "pdk_source": layout.summary.get("pdk_source", "gdsfactory/cspdk"),
        "pdk_version": layout.summary.get("pdk_version", _cspdk_version()),
        "pdk_gds_dir": layout.summary.get("pdk_gds_dir", str(_cornerstone_pdk_gds_dir())),
        "local_klayout_drc_deck": str(local_deck) if local_deck is not None else None,
        "status": "local_lydrc_available_not_run" if local_deck is not None else "no_local_lydrc_found",
        "note": (
            "The installed cspdk package provides KLayout technology files and fixed-cell GDS files. "
            "No local .lydrc deck was found in the installed package; use the upstream gdsfactory-plus "
            "DRC workflow or foundry signoff deck for final rule checking."
            if local_deck is None
            else "A local cspdk .lydrc deck was found, but this artifact writer does not execute DRC."
        ),
    }


def _find_cspdk_lydrc() -> Path | None:
    roots: list[Path] = []
    try:
        import cspdk

        roots.append(Path(cspdk.__file__).resolve().parent)
    except Exception:
        pass
    try:
        from cspdk.si220.cband.config import PATH as CSPDK_PATH

        roots.append(Path(CSPDK_PATH.klayout).resolve())
    except Exception:
        pass
    for root in dict.fromkeys(roots):
        if not root.exists():
            continue
        matches = sorted(root.rglob("*.lydrc"))
        if matches:
            return matches[0]
    return None


def _lane_positions(width: int, pitch: float) -> list[float]:
    center = (width - 1) / 2
    return [(index - center) * pitch for index in range(width)]


def _splitter_tree_depth(width: int) -> int:
    if width < 2 or width & (width - 1):
        raise ValueError("The 50/50 splitter tree requires a power-of-two width of at least 2.")
    return int(math.log2(width))


def _linear_taper_polygon(
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    w0: float,
    w1: float,
    samples: int,
) -> pf.Polygon:
    xs = [
        float(x0) + (float(x1) - float(x0)) * index / (int(samples) - 1)
        for index in range(int(samples))
    ]
    centers = [
        float(y0) + (float(y1) - float(y0)) * index / (int(samples) - 1)
        for index in range(int(samples))
    ]
    widths = [
        float(w0) + (float(w1) - float(w0)) * index / (int(samples) - 1)
        for index in range(int(samples))
    ]
    lower = [(x, center - 0.5 * width) for x, center, width in zip(xs, centers, widths)]
    upper = [(x, center + 0.5 * width) for x, center, width in zip(reversed(xs), reversed(centers), reversed(widths))]
    return pf.Polygon([*lower, *upper])


def _add_splitter_tree_routes(
    *,
    component: pf.Component,
    input_ref: pf.Reference,
    splitter_refs: list[list[pf.Reference]],
    mod_refs: list[pf.Reference],
    technology: pf.Technology,
) -> list[RouteRecord]:
    route_records: list[RouteRecord] = []
    root = splitter_refs[0][0]
    route_records.append(
        _add_manhattan_route(
            component,
            input_ref["P0"],
            root["P0"],
            technology,
            group="input_splitter_tree",
            name="input_grating_to_splitter_root",
            points=_straight_or_manhattan_points(input_ref["P0"].center, root["P0"].center),
        )
    )
    for stage, parent_refs in enumerate(splitter_refs[:-1]):
        child_refs = splitter_refs[stage + 1]
        for node, parent_ref in enumerate(parent_refs):
            lower_child = child_refs[2 * node]
            upper_child = child_refs[2 * node + 1]
            route_records.append(
                _add_manhattan_route(
                    component,
                    parent_ref["P_LO"],
                    lower_child["P0"],
                    technology,
                    group="input_splitter_tree",
                    name=f"splitter_s{stage}_n{node}_lo_to_child",
                    points=_tree_link_points(parent_ref["P_LO"].center, lower_child["P0"].center),
                )
            )
            route_records.append(
                _add_manhattan_route(
                    component,
                    parent_ref["P_HI"],
                    upper_child["P0"],
                    technology,
                    group="input_splitter_tree",
                    name=f"splitter_s{stage}_n{node}_hi_to_child",
                    points=_tree_link_points(parent_ref["P_HI"].center, upper_child["P0"].center),
                )
            )

    for node, splitter_ref in enumerate(splitter_refs[-1]):
        lower_channel = 2 * node
        upper_channel = lower_channel + 1
        route_records.append(
            _add_manhattan_route(
                component,
                splitter_ref["P_LO"],
                mod_refs[lower_channel]["P0"],
                technology,
                group="input_splitter_tree",
                name=f"splitter_leaf_n{node}_lo_to_mod",
                points=_tree_link_points(splitter_ref["P_LO"].center, mod_refs[lower_channel]["P0"].center),
            )
        )
        route_records.append(
            _add_manhattan_route(
                component,
                splitter_ref["P_HI"],
                mod_refs[upper_channel]["P0"],
                technology,
                group="input_splitter_tree",
                name=f"splitter_leaf_n{node}_hi_to_mod",
                points=_tree_link_points(splitter_ref["P_HI"].center, mod_refs[upper_channel]["P0"].center),
            )
        )
    return route_records


def _add_ordered_interface_routes(
    *,
    component: pf.Component,
    starts: list[pf.Port],
    ends: list[pf.Port],
    technology: pf.Technology,
    group: str,
    fan: str,
) -> list[RouteRecord]:
    if len(starts) != len(ends):
        raise ValueError("Ordered interface routing requires matching start/end port counts.")
    width = len(starts)
    route_records: list[RouteRecord] = []
    for channel, (start, end) in enumerate(zip(starts, ends)):
        points = _ordered_interface_points(
            start=tuple(float(v) for v in start.center),
            end=tuple(float(v) for v in end.center),
            channel=channel,
            width=width,
            fan=fan,
        )
        route_records.append(
            _add_manhattan_route(
                component,
                start,
                end,
                technology,
                group=group,
                name=f"{group}_ch{channel:02d}",
                points=points,
            )
        )
    return route_records


def _add_balanced_interface_routes(
    *,
    component: pf.Component,
    starts: list[pf.Port],
    ends: list[pf.Port],
    technology: pf.Technology,
    group: str,
    fan: str,
) -> list[RouteRecord]:
    if len(starts) != len(ends):
        raise ValueError("Balanced interface routing requires matching start/end port counts.")
    width = len(starts)
    vertical_deltas = [abs(float(start.center[1]) - float(end.center[1])) for start, end in zip(starts, ends)]
    max_vertical_delta = max(vertical_deltas)
    radius = CORNERSTONE_STRIP_90_BEND_OFFSET
    wide_ports = starts if fan == "in" else ends
    max_loop_height = (
        max(radius * 2, 0.5 * abs(float(wide_ports[1].center[1] - wide_ports[0].center[1])) - 5.0)
        if len(wide_ports) > 1
        else 3.5 * radius
    )
    min_loop_extra = _delay_loop_extra(2 * radius, radius)
    max_required_extra = min_loop_extra + max(max_vertical_delta - delta for delta in vertical_deltas)
    max_loop_count = _delay_loop_count(max_required_extra, radius, max_loop_height)
    delay_region_length = max_loop_count * _delay_loop_period(radius) + 4 * radius
    route_records: list[RouteRecord] = []
    for channel, (start, end, vertical_delta) in enumerate(zip(starts, ends, vertical_deltas)):
        points = _balanced_interface_points(
            start=tuple(float(v) for v in start.center),
            end=tuple(float(v) for v in end.center),
            channel=channel,
            width=width,
            max_vertical_delta=float(max_vertical_delta),
            required_extra_um=float(min_loop_extra + max_vertical_delta - vertical_delta),
            delay_region_length=float(delay_region_length),
            radius=radius,
            max_loop_height=float(max_loop_height),
            fan=fan,
        )
        route_records.append(
            _add_manhattan_route(
                component,
                start,
                end,
                technology,
                group=group,
                name=f"{group}_ch{channel:02d}",
                points=points,
            )
        )
    return route_records


def _add_manhattan_route(
    component: pf.Component,
    port1: pf.Port,
    port2: pf.Port,
    technology: pf.Technology,
    *,
    group: str,
    name: str,
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> RouteRecord:
    clean_points = _dedupe_points(tuple((float(x), float(y)) for x, y in points))
    _validate_manhattan_points(clean_points)
    bend_radius = CORNERSTONE_STRIP_90_BEND_OFFSET
    try:
        route = _rounded_manhattan_route_component(
            port1=port1,
            port2=port2,
            points=clean_points,
            radius=bend_radius,
            technology=technology,
            name=f"route_{_route_slug(name)}",
        )
    except ValueError:
        if group != "input_splitter_tree":
            raise
        route = pf.parametric.route_s_bend(
            port1=port1,
            port2=port2,
            euler_fraction=1.0,
            technology=technology,
            name=f"route_{_route_slug(name)}",
        )
    component.add_reference(route)
    route_length, straight_count, bend_count = _photonforge_route_length(route)
    return RouteRecord(
        name=name,
        group=group,
        component=route,
        points=clean_points,
        length_um=route_length,
        bend_radius_um=bend_radius,
        bend_count=bend_count,
        straight_count=straight_count,
    )


def _rounded_manhattan_route_component(
    *,
    port1: pf.Port,
    port2: pf.Port,
    points: tuple[tuple[float, float], ...],
    radius: float,
    technology: pf.Technology,
    name: str,
) -> pf.Component:
    spec = port1.spec
    route = pf.Component(name, technology=technology)
    route.add_port(
        pf.Port(points[0], (float(port1.input_direction) + 180.0) % 360.0, spec, inverted=not port1.inverted),
        port_name="P0",
    )
    route.add_port(
        pf.Port(points[-1], (float(port2.input_direction) + 180.0) % 360.0, port2.spec, inverted=not port2.inverted),
        port_name="P1",
    )
    if len(points) == 2:
        _add_straight_reference(route, points[0], _segment_direction(points[0], points[1]), _segment_length(points[0], points[1]), spec)
        return route

    segment_lengths = [_segment_length(start, end) for start, end in zip(points, points[1:])]
    directions = [_segment_direction(start, end) for start, end in zip(points, points[1:])]
    for index, length in enumerate(segment_lengths):
        required = (radius if index > 0 else 0.0) + (radius if index < len(segment_lengths) - 1 else 0.0)
        if length < required - 1e-9:
            raise ValueError(
                f"Route {name} has segment {index} length {length:.3f} um, "
                f"which cannot fit {radius:.3f} um bends."
            )

    for index, (start, end) in enumerate(zip(points, points[1:])):
        direction = directions[index]
        length = segment_lengths[index]
        start_trim = radius if index > 0 else 0.0
        end_trim = radius if index < len(segment_lengths) - 1 else 0.0
        straight_length = length - start_trim - end_trim
        straight_start = _advance_point(start, direction, start_trim)
        if straight_length > 1e-9:
            _add_straight_reference(route, straight_start, direction, straight_length, spec)
        if index < len(segment_lengths) - 1:
            turn = _signed_turn(direction, directions[index + 1])
            if abs(turn) != 90:
                raise ValueError(f"Route {name} has unsupported turn angle {turn}.")
            bend_start = _advance_point(end, direction, -radius)
            _add_bend_reference(route, bend_start, direction, turn, radius, spec)
    return route


def _add_straight_reference(
    component: pf.Component,
    start: tuple[float, float],
    direction: float,
    length: float,
    spec: pf.PortSpec,
) -> None:
    straight = pf.parametric.straight(port_spec=spec, length=float(length), technology=component.technology)
    ref = component.add_reference(straight)
    ref.rotate(float(direction), center=(0.0, 0.0))
    ref.translate(start)


def _add_bend_reference(
    component: pf.Component,
    start: tuple[float, float],
    direction: float,
    turn: float,
    radius: float,
    spec: pf.PortSpec,
) -> None:
    bend = cornerstone_strip_90_bend(turn=float(turn), port_spec=spec)
    ref = component.add_reference(bend)
    ref.rotate(float(direction), center=(0.0, 0.0))
    ref.translate(start)


def _segment_length(start: tuple[float, float], end: tuple[float, float]) -> float:
    return abs(float(end[0] - start[0])) + abs(float(end[1] - start[1]))


def _segment_direction(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if dx > 0.0 and dy == 0.0:
        return 0.0
    if dx < 0.0 and dy == 0.0:
        return 180.0
    if dy > 0.0 and dx == 0.0:
        return 90.0
    if dy < 0.0 and dx == 0.0:
        return 270.0
    raise ValueError(f"Non-Manhattan segment from {start} to {end}.")


def _advance_point(point: tuple[float, float], direction: float, distance: float) -> tuple[float, float]:
    radians = math.radians(direction)
    return (
        float(point[0]) + float(distance) * round(math.cos(radians)),
        float(point[1]) + float(distance) * round(math.sin(radians)),
    )


def _signed_turn(direction: float, next_direction: float) -> float:
    return (next_direction - direction + 180.0) % 360.0 - 180.0


def _straight_or_manhattan_points(start: Any, end: Any) -> list[tuple[float, float]]:
    start_xy = tuple(float(v) for v in start)
    end_xy = tuple(float(v) for v in end)
    if start_xy[0] == end_xy[0] or start_xy[1] == end_xy[1]:
        return [start_xy, end_xy]
    x_mid = 0.5 * (start_xy[0] + end_xy[0])
    return [start_xy, (x_mid, start_xy[1]), (x_mid, end_xy[1]), end_xy]


def _tree_link_points(start: Any, end: Any) -> list[tuple[float, float]]:
    start_xy = tuple(float(v) for v in start)
    end_xy = tuple(float(v) for v in end)
    x_mid = 0.5 * (start_xy[0] + end_xy[0])
    return [start_xy, (x_mid, start_xy[1]), (x_mid, end_xy[1]), end_xy]


def _balanced_interface_points(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    channel: int,
    width: int,
    max_vertical_delta: float,
    required_extra_um: float,
    delay_region_length: float,
    radius: float,
    max_loop_height: float,
    fan: str,
) -> list[tuple[float, float]]:
    x0, y0 = start
    x1, y1 = end
    if x1 <= x0:
        raise ValueError("Balanced interface routing expects left-to-right ports.")
    span = x1 - x0
    if fan not in {"in", "out"}:
        raise ValueError("fan must be 'in' or 'out'.")
    if span <= delay_region_length + 6 * radius:
        raise ValueError("Balanced route span is too short for bend-aware delay equalization.")
    loop_sign = _toward_center_loop_sign(y0 if fan == "in" else y1)
    if fan == "in":
        delay_points = _delay_loop_points(
            start=(x0, y0),
            required_extra_um=required_extra_um,
            sign=loop_sign,
            radius=radius,
            max_loop_height=max_loop_height,
        )
        x_transition = _ordered_transition_x(x0 + delay_region_length, x1, channel, width, fan)
        return [*delay_points, (x_transition, y0), (x_transition, y1), (x1, y1)]

    x_delay_start = x1 - delay_region_length
    x_transition = _ordered_transition_x(x0, x_delay_start, channel, width, fan)
    delay_points = _delay_loop_points(
        start=(x_delay_start, y1),
        required_extra_um=required_extra_um,
        sign=loop_sign,
        radius=radius,
        max_loop_height=max_loop_height,
    )
    return [(x0, y0), (x_transition, y0), (x_transition, y1), *delay_points, (x1, y1)]


def _ordered_interface_points(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    channel: int,
    width: int,
    fan: str,
) -> list[tuple[float, float]]:
    x0, y0 = start
    x1, y1 = end
    if x1 <= x0:
        raise ValueError("Ordered interface routing expects left-to-right ports.")
    x_transition = _ordered_transition_x(x0, x1, channel, width, fan)
    return [(x0, y0), (x_transition, y0), (x_transition, y1), (x1, y1)]


def _ordered_transition_x(x0: float, x1: float, channel: int, width: int, fan: str) -> float:
    span = x1 - x0
    if width == 1:
        return x0 + 0.5 * span
    center = 0.5 * (width - 1)
    outer_fraction = abs(channel - center) / center
    if fan == "in":
        fraction = 0.25 + 0.5 * outer_fraction
    elif fan == "out":
        fraction = 0.25 + 0.5 * (1.0 - outer_fraction)
    else:
        raise ValueError("fan must be 'in' or 'out'.")
    return x0 + fraction * span


def _serpentine_horizontal_points(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    extra_length: float,
    sign: float,
    omit_start: bool = False,
) -> list[tuple[float, float]]:
    x0, y0 = start
    x1, y1 = end
    if y0 != y1:
        raise ValueError("Serpentine length matching is only defined along one horizontal lane.")
    if x1 < x0:
        raise ValueError("Serpentine length matching expects left-to-right horizontal space.")
    points: list[tuple[float, float]] = [] if omit_start else [(x0, y0)]
    if extra_length <= 1e-9:
        points.append((x1, y1))
        return points
    total_excursion = 0.5 * extra_length
    max_excursion = 8.0
    doglegs = max(1, math.ceil(total_excursion / max_excursion))
    base_excursion = total_excursion / doglegs
    span = x1 - x0
    if span <= 0.0:
        raise ValueError("Serpentine length matching needs positive horizontal space.")
    step = span / doglegs
    for dogleg in range(doglegs):
        x_left = x0 + dogleg * step + 0.25 * step
        x_right = x0 + dogleg * step + 0.75 * step
        y_excursion = y0 + sign * base_excursion
        points.extend(
            [
                (x_left, y0),
                (x_left, y_excursion),
                (x_right, y_excursion),
                (x_right, y0),
            ]
        )
    points.append((x1, y1))
    return points


def _outward_serpentine_sign(y: float) -> float:
    return -1.0 if y < 0.0 else 1.0


def _toward_center_loop_sign(y: float) -> float:
    return 1.0 if y < 0.0 else -1.0


def _delay_loop_points(
    *,
    start: tuple[float, float],
    required_extra_um: float,
    sign: float,
    radius: float,
    max_loop_height: float,
) -> list[tuple[float, float]]:
    min_height = 2 * radius
    if required_extra_um <= 1e-9:
        return [start]
    loop_count = _delay_loop_count(required_extra_um, radius, max_loop_height)
    per_loop_extra = required_extra_um / loop_count
    height = _delay_loop_height(per_loop_extra, radius)
    if height < min_height - 1e-9 or height > max_loop_height + 1e-9:
        raise ValueError(
            f"Cannot realize {required_extra_um:.6f} um delay with {loop_count} loops "
            f"and {radius:.3f} um bend radius."
        )
    x, y = start
    points = [start]
    period = _delay_loop_period(radius)
    for _ in range(loop_count):
        points.extend(
            [
                (x + 2 * radius, y),
                (x + 2 * radius, y + sign * height),
                (x + 4 * radius, y + sign * height),
                (x + 4 * radius, y),
            ]
        )
        x += period
    points.append((x, y))
    return points


def _delay_loop_period(radius: float) -> float:
    return 6 * radius


def _delay_loop_extra(height: float, radius: float) -> float:
    return 2 * height + 4 * (0.5 * math.pi * radius - 2 * radius)


def _delay_loop_height(extra_um: float, radius: float) -> float:
    return 0.5 * (extra_um - 4 * (0.5 * math.pi * radius - 2 * radius))


def _delay_loop_count(required_extra_um: float, radius: float, max_loop_height: float) -> int:
    min_extra = _delay_loop_extra(2 * radius, radius)
    max_extra = _delay_loop_extra(max_loop_height, radius)
    if required_extra_um < min_extra - 1e-9:
        return 1
    for count in range(1, 128):
        if count * min_extra - 1e-9 <= required_extra_um <= count * max_extra + 1e-9:
            return count
    raise ValueError(f"Required delay {required_extra_um:.6f} um exceeds delay loop capacity.")


def _dedupe_points(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    clean: list[tuple[float, float]] = []
    for point in points:
        if not clean or point != clean[-1]:
            clean.append(point)
            while len(clean) >= 3 and _points_are_collinear(clean[-3], clean[-2], clean[-1]):
                last = clean.pop()
                clean.pop()
                clean.append(last)
    return tuple(clean)


def _points_are_collinear(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> bool:
    return (first[0] == second[0] == third[0]) or (first[1] == second[1] == third[1])


def _validate_manhattan_points(points: tuple[tuple[float, float], ...]) -> None:
    if len(points) < 2:
        raise ValueError("A route needs at least two points.")
    for first, second in zip(points, points[1:]):
        if first[0] != second[0] and first[1] != second[1]:
            raise ValueError(f"Diagonal route segment found between {first} and {second}.")


def _polyline_length(points: tuple[tuple[float, float], ...]) -> float:
    return sum(
        abs(second[0] - first[0]) + abs(second[1] - first[1])
        for first, second in zip(points, points[1:])
    )


def _photonforge_route_length(route: pf.Component) -> tuple[float, int, int]:
    function = getattr(route, "parametric_function", None)
    function_name = str(function).rsplit(".", maxsplit=1)[-1]
    if function_name in {"s_bend", "route_s_bend"}:
        waveguide_model = route.models.get("Waveguide")
        length = float(waveguide_model.length) if waveguide_model is not None and waveguide_model.length is not None else 0.0
        return length, 0, 2
    if function_name == "straight":
        return float(route.parametric_kwargs["length"]), 1, 0
    if function_name == "bend":
        radius = float(route.parametric_kwargs["radius"])
        angle = abs(float(route.parametric_kwargs["angle"]))
        return math.radians(angle) * radius, 0, 1
    if function_name == "cornerstone_strip_90_bend":
        return 0.5 * math.pi * CORNERSTONE_STRIP_90_BEND_OFFSET, 0, 1

    total = 0.0
    straight_count = 0
    bend_count = 0
    for ref in route.references:
        function = getattr(ref.component, "parametric_function", None)
        function_name = str(function).rsplit(".", maxsplit=1)[-1]
        kwargs = getattr(ref.component, "parametric_kwargs", {})
        if function_name == "straight":
            total += float(kwargs["length"])
            straight_count += 1
            continue
        if function_name == "bend":
            radius = float(kwargs["radius"])
            angle = abs(float(kwargs["angle"]))
            total += math.radians(angle) * radius
            bend_count += 1
            continue
        if function_name == "cornerstone_strip_90_bend":
            total += 0.5 * math.pi * CORNERSTONE_STRIP_90_BEND_OFFSET
            bend_count += 1
            continue
        if function_name == "s_bend":
            waveguide_model = ref.component.models.get("Waveguide")
            if waveguide_model is not None and waveguide_model.length is not None:
                total += float(waveguide_model.length)
            else:
                total += _reference_port_distance(ref)
            bend_count += 2
            continue
        total += _reference_port_distance(ref)
    return total, straight_count, bend_count


def _reference_port_distance(ref: pf.Reference) -> float:
    ports = list(ref.component.ports.values())
    if len(ports) < 2:
        return 0.0
    start = ports[0].center
    end = ports[1].center
    return float(math.hypot(float(end[0] - start[0]), float(end[1] - start[1])))


def _route_slug(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name)


def find_diagonal_route_segments(route_records: tuple[RouteRecord, ...]) -> list[tuple[str, int]]:
    diagonal_segments: list[tuple[str, int]] = []
    for route in route_records:
        for segment_index, (first, second) in enumerate(zip(route.points, route.points[1:])):
            if first[0] != second[0] and first[1] != second[1]:
                diagonal_segments.append((route.name, segment_index))
    return diagonal_segments


def _phase_balanced_group_report(route_records: tuple[RouteRecord, ...]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    for route in route_records:
        if "_mod_to_id" in route.group or "_id_to_L" in route.group:
            groups.setdefault(route.group, []).append(route.length_um)
    report: dict[str, dict[str, float | int]] = {}
    for group, lengths in groups.items():
        reference = lengths[0]
        max_delta = max(abs(length - reference) for length in lengths)
        report[group] = {
            "count": len(lengths),
            "length_um": round(reference, 6),
            "max_delta_um": round(max_delta, 9),
        }
    return report


def _route_bend_report(route_records: tuple[RouteRecord, ...]) -> dict[str, float | int]:
    bend_radii = [record.bend_radius_um for record in route_records if record.bend_count > 0]
    manual_path_routes = sum(1 for record in route_records if record.bend_count == 0 and len(record.points) > 2)
    return {
        "bend_aware_routes": sum(1 for record in route_records if record.bend_count > 0),
        "manual_path_routes": manual_path_routes,
        "min_bend_radius_um": round(min(bend_radii), 6) if bend_radii else 0.0,
        "total_bends": sum(record.bend_count for record in route_records),
    }


def find_route_overlaps(layout: LumixLayout) -> list[tuple[int, int]]:
    route_bounds = [route.component.bounds() for route in layout.route_records]
    route_structures = [
        route.get_structures("wg_lf", depth=-1)
        for route in layout.route_components
    ]
    overlaps: list[tuple[int, int]] = []
    for first_index, first_structures in enumerate(route_structures):
        if not first_structures:
            continue
        for second_index in range(first_index + 1, len(route_structures)):
            if not _bounds_intersect(route_bounds[first_index], route_bounds[second_index]):
                continue
            second_structures = route_structures[second_index]
            if second_structures and pf.boolean(first_structures, second_structures, "*"):
                overlaps.append((first_index, second_index))
    return overlaps


def _bounds_intersect(first: tuple[Any, Any], second: tuple[Any, Any]) -> bool:
    first_lower, first_upper = first
    second_lower, second_upper = second
    return not (
        float(first_upper[0]) < float(second_lower[0])
        or float(second_upper[0]) < float(first_lower[0])
        or float(first_upper[1]) < float(second_lower[1])
        or float(second_upper[1]) < float(first_lower[1])
    )


def _add_floorplan(component: pf.Component, *, length: float, width: float) -> None:
    lower, upper = component.bounds()
    span_x = float(upper[0] - lower[0])
    span_y = float(upper[1] - lower[1])
    if span_x > length or span_y > width:
        raise ValueError(
            f"Layout span {span_x:.3f} x {span_y:.3f} um does not fit "
            f"inside the {length:.3f} x {width:.3f} um Cornerstone full-block floorplan."
        )
    center_x = 0.5 * float(lower[0] + upper[0])
    center_y = 0.5 * float(lower[1] + upper[1])
    component.add(
        "Floorplan",
        pf.Rectangle(
            corner1=(center_x - 0.5 * length, center_y - 0.5 * width),
            corner2=(center_x + 0.5 * length, center_y + 0.5 * width),
        ),
    )


def _bounds_to_list(bounds: tuple[Any, Any]) -> list[list[float]]:
    lower, upper = bounds
    return [[float(lower[0]), float(lower[1])], [float(upper[0]), float(upper[1])]]


def _netlist_report(component: pf.Component) -> dict[str, Any]:
    netlist = component.get_netlist()
    return {
        "instance_count": len(netlist["instances"]),
        "port_count": len(netlist["ports"]),
        "connection_count": len(netlist["connections"]),
        "virtual_connection_count": len(netlist["virtual connections"]),
        "ports": {repr(key): value for key, value in netlist["ports"].items()},
        "connections": [repr(connection) for connection in netlist["connections"]],
        "virtual_connections": [repr(connection) for connection in netlist["virtual connections"]],
    }


def _write_preview_from_gds(gds_path: Path, preview_path: Path) -> Path | None:
    try:
        import gdstk
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ImportError:
        return None

    library = gdstk.read_gds(gds_path)
    cell = library.top_level()[0]
    polygons_by_spec: dict[tuple[int, int], list[Any]] = {}
    for polygon in cell.get_polygons(apply_repetitions=True):
        polygons_by_spec.setdefault((polygon.layer, polygon.datatype), []).append(polygon.points)

    if not polygons_by_spec:
        return None

    colors = {
        CORNERSTONE_ACTIVE_LAYERS["wg_lf"]: "#d62728",
        CORNERSTONE_ACTIVE_LAYERS["grating_duv"]: "#1f77b4",
        CORNERSTONE_ACTIVE_LAYERS["rib_slab"]: "#2ca02c",
        CORNERSTONE_ACTIVE_LAYERS["Electrode_LF"]: "#bcbd22",
        CORNERSTONE_ACTIVE_LAYERS["Floorplan"]: "#555555",
    }
    fig, ax = plt.subplots(figsize=(14, 7))
    for layer, polygons in polygons_by_spec.items():
        collection = PolyCollection(
            polygons,
            facecolors=colors.get(layer, "#999999"),
            edgecolors=colors.get(layer, "#666666"),
            linewidths=0.2,
            alpha=0.55 if layer != CORNERSTONE_ACTIVE_LAYERS["Floorplan"] else 0.10,
        )
        ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title("Lumix 3-layer module, CORNERSTONE SOI 220 nm active")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=180)
    plt.close(fig)
    return preview_path
