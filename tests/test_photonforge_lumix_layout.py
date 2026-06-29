import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("photonforge")
import photonforge as pf

from experiments.common.photonforge_lumix_layout import (
    CORNERSTONE_ACTIVE_LAYERS,
    LUMIX_PDK_SPARAMETER_ROLES,
    LumixCompactModelConfig,
    LumixLayoutConfig,
    MissingPdkSParameterError,
    PdkSParameterLibrary,
    attach_lumix_compact_models,
    attach_lumix_pdk_sparameter_models,
    audit_lumix_pdk_sparameter_library,
    build_lumix_module,
    cornerstone_soi220_active_technology,
    eo_phase_transmission,
    find_diagonal_route_segments,
    find_route_overlaps,
    pdk_route_balance_objective,
    pdk_route_loss_report,
    run_lumix_pdk_sparameter_sweep,
    run_lumix_circuit_sweep,
    write_lumix_pdk_sparameter_manifest_template,
    write_lumix_layout_artifacts,
)


def test_cornerstone_technology_uses_active_soi220_layers():
    technology = cornerstone_soi220_active_technology()

    assert technology.layers["wg_lf"].layer == CORNERSTONE_ACTIVE_LAYERS["wg_lf"]
    assert technology.layers["grating_duv"].layer == CORNERSTONE_ACTIVE_LAYERS["grating_duv"]
    assert technology.layers["rib_slab"].layer == CORNERSTONE_ACTIVE_LAYERS["rib_slab"]
    assert technology.layers["Electrode_LF"].layer == CORNERSTONE_ACTIVE_LAYERS["Electrode_LF"]
    assert technology.layers["HEATER"].layer == (39, 0)
    assert technology.layers["PAD"].layer == (41, 0)
    assert technology.ports["strip_1550nm"].width == 0.45
    assert technology.ports["strip_1550nm"].default_radius == 10.0


def test_three_layer_lumix_module_has_expected_physical_contract():
    layout = build_lumix_module(LumixLayoutConfig(width=16, layers=3))

    assert layout.summary["width"] == 16
    assert layout.summary["layers"] == 3
    assert layout.summary["input_gratings"] == 1
    assert layout.summary["output_gratings"] == 16
    assert layout.summary["splitter_tree_depth"] == 4
    assert layout.summary["power_splitters_1x2"] == 15
    assert layout.summary["phase_modulators"] == 48
    assert layout.summary["inverse_design_regions"] == 3
    assert layout.summary["inverse_design_region_size_um"] == [21.0, 21.0]
    assert layout.summary["inverse_design_component_size_um"] == [27.2, 21.0]
    assert layout.summary["inverse_design_port_pitch_um"] == 1.25
    assert layout.summary["port_taper_length_um"] == 3.1
    assert layout.summary["port_taper_mouth_width_um"] == 1.25
    assert layout.summary["optical_routes"] == 127
    assert layout.summary["route_overlap_count"] == 0
    assert layout.summary["diagonal_route_segment_count"] == 0
    assert layout.summary["bend_aware_routes"] == 126
    assert layout.summary["manual_path_routes"] == 0
    assert layout.summary["route_bend_radius_um"] == 5.0
    assert layout.summary["min_route_bend_radius_um"] == 5.0
    assert layout.summary["total_route_bends"] == 1692
    assert layout.summary["die_size_um"] == [11470.0, 4900.0]
    assert layout.summary["pdk_source"] == "gdsfactory/cspdk"
    assert layout.summary["pdk_version"] == _expected_cspdk_version()
    assert Path(layout.summary["pdk_gds_dir"]).exists()
    assert Path(layout.summary["pdk_gds_dir"]).name == "gds"
    assert layout.summary["pdk_components_used"] == {
        "input_grating": "SOI220nm_1550nm_TE_STRIP_Grating_Coupler",
        "output_grating": "SOI220nm_1550nm_TE_STRIP_Grating_Coupler",
        "splitter_1x2": "SOI220nm_1550nm_TE_STRIP_2x1_MMI",
        "route_bend_90": "SOI220nm_1550nm_TE_STRIP_90_Degree_Bend",
        "phase_modulator_heater": "Heater",
    }
    assert "SOI220nm_1550nm_TE_MZI_Modulator" in layout.summary["pdk_components_not_used"]
    assert layout.summary["fiber_ports"] == 17
    assert layout.summary["electrical_terminals"] == 96

    assert len(layout.component.ports) == 17
    assert len(layout.component.terminals) == 96
    assert "fiber_in" in layout.component.ports
    assert "fiber_out_15" in layout.component.ports
    assert "L1_CH00_SIG" in layout.component.terminals
    assert "L3_CH15_GND" in layout.component.terminals
    assert find_diagonal_route_segments(layout.route_records) == []

    phase_groups = layout.summary["phase_balanced_route_groups"]
    assert set(phase_groups) == {
        "L1_mod_to_id",
        "L1_id_to_L2_mod",
        "L2_mod_to_id",
        "L2_id_to_L3_mod",
        "L3_mod_to_id",
    }
    for report in phase_groups.values():
        assert report["count"] == 16
        assert report["length_um"] == 3372.74889
        assert report["max_delta_um"] == 0.0

    netlist = layout.component.get_netlist()
    assert netlist["virtual connections"] == []
    assert find_route_overlaps(layout) == []


def test_route_turns_use_cspdk_bend_gds_wrappers():
    layout = build_lumix_module(LumixLayoutConfig(width=16, layers=3))

    bend_reference_count = 0
    for record in layout.route_records:
        bend_references = [
            ref
            for ref in record.component.references
            if "SOI220nm_1550nm_TE_STRIP_90_Degree_Bend" in ref.component.name
        ]
        assert len(bend_references) == record.bend_count, record.name
        bend_reference_count += len(bend_references)

    assert bend_reference_count == 1692
    assert bend_reference_count == layout.summary["total_route_bends"]


def test_lumix_layout_artifact_writer_exports_gds_oas_and_reports(tmp_path):
    layout = build_lumix_module(LumixLayoutConfig(width=4, layers=2))

    artifacts = write_lumix_layout_artifacts(layout, tmp_path, write_preview=False)

    assert artifacts.gds_path == tmp_path / "lumix_2layer_w4_cornerstone_soi220.gds"
    assert artifacts.oas_path == tmp_path / "lumix_2layer_w4_cornerstone_soi220.oas"
    assert artifacts.summary_path == tmp_path / "layout_summary.json"
    assert artifacts.connectivity_path == tmp_path / "connectivity.json"
    assert artifacts.pdk_drc_summary_path == tmp_path / "cspdk_drc_summary.json"
    assert artifacts.gds_path.stat().st_size > 0
    assert artifacts.oas_path.stat().st_size > 0
    assert artifacts.summary_path.stat().st_size > 0
    assert artifacts.connectivity_path.stat().st_size > 0
    assert artifacts.pdk_drc_summary_path.stat().st_size > 0
    assert artifacts.pdk_sparameter_manifest_path == tmp_path / "pdk_sparameters_manifest.json"
    assert artifacts.pdk_sparameter_manifest_path.stat().st_size > 0

    pdk_drc_summary = json.loads(artifacts.pdk_drc_summary_path.read_text())
    assert pdk_drc_summary["pdk_source"] == "gdsfactory/cspdk"
    assert pdk_drc_summary["pdk_version"] == layout.summary["pdk_version"]
    assert pdk_drc_summary["status"] in {"local_lydrc_available_not_run", "no_local_lydrc_found"}

    manifest = json.loads(artifacts.pdk_sparameter_manifest_path.read_text())
    assert manifest["accepted_source_types"] == ["em_derived", "measured", "pdk_measured"]
    assert {entry["role"] for entry in manifest["models"]} == set(LUMIX_PDK_SPARAMETER_ROLES)
    ports_by_role = {entry["role"]: entry["ports"] for entry in manifest["models"]}
    assert ports_by_role["grating_coupler"] == ["Fiber", "P0"]
    assert ports_by_role["splitter_1x2"] == ["P0", "P_HI", "P_LO"]
    assert ports_by_role["phase_modulator"] == ["P0", "P1"]
    assert ports_by_role["strip_straight"] == ["P0", "P1"]
    assert ports_by_role["strip_bend_90"] == ["P0", "P1"]
    assert ports_by_role["inverse_design_region"][:2] == ["W00", "W01"]


def test_lumix_layout_artifact_writer_can_emit_circuit_sweep(tmp_path):
    layout = build_lumix_module(LumixLayoutConfig(width=2, layers=1))

    artifacts = write_lumix_layout_artifacts(
        layout,
        tmp_path,
        write_preview=False,
        write_circuit_sweep=True,
        circuit_voltage_masks=np.zeros((1, 1, 2), dtype=float),
        compact_config=LumixCompactModelConfig(
            grating_coupling_loss_db=0.0,
            splitter_excess_loss_db=0.0,
            route_loss_db_per_cm=0.0,
            bend_loss_db=0.0,
            phase_modulator_insertion_loss_db=0.0,
        ),
        inverse_design_matrices=[np.eye(2, dtype=np.complex128)],
    )

    assert artifacts.circuit_sweep_path == tmp_path / "circuit_sweep.json"
    circuit_sweep = json.loads(artifacts.circuit_sweep_path.read_text())
    assert circuit_sweep["input_port"] == "fiber_in"
    assert circuit_sweep["output_ports"] == ["fiber_out_00", "fiber_out_01"]
    assert circuit_sweep["points"][0]["total_output_power"] > 0.0


def test_lumix_layout_artifact_writer_can_emit_strict_pdk_sparameter_sweep(tmp_path):
    layout = build_lumix_module(LumixLayoutConfig(width=2, layers=1))

    with pytest.raises(MissingPdkSParameterError, match="pdk_sparameter_library"):
        write_lumix_layout_artifacts(
            layout,
            tmp_path / "missing_library",
            write_preview=False,
            write_pdk_sparameter_sweep=True,
        )

    artifacts = write_lumix_layout_artifacts(
        layout,
        tmp_path,
        write_preview=False,
        write_pdk_sparameter_sweep=True,
        pdk_sparameter_library=_write_lossless_pdk_sparameter_library(tmp_path / "sparams", width=2),
    )

    assert artifacts.pdk_sparameter_sweep_path == tmp_path / "pdk_sparameter_sweep.json"
    pdk_sweep = json.loads(artifacts.pdk_sparameter_sweep_path.read_text())
    assert pdk_sweep["model_report"]["model_source"] == "pdk_sparameters"
    assert pdk_sweep["route_loss_report"]["source"] == "pdk_sparameters"
    assert pdk_sweep["points"][0]["total_output_power"] == pytest.approx(1.0)


def test_lumix_compact_models_attach_to_every_circuit_block():
    layout = build_lumix_module(LumixLayoutConfig(width=4, layers=2))

    report = attach_lumix_compact_models(layout)

    assert "Circuit" in layout.component.models
    assert report["top_level_model"] == "CircuitModel"
    assert report["component_reference_counts"] == {
        "grating_couplers": 5,
        "power_splitters_1x2": 3,
        "phase_modulators": 8,
        "inverse_design_regions": 2,
        "routes": 23,
    }
    assert report["inverse_design_matrix_source"] == "identity"
    assert report["phase_modulator_model"] == "VoltagePhaseModulatorModel"
    assert report["voltage_update_mode"] == "per_reference_model_updates"

    for route in layout.route_components:
        assert "Compact" in route.models
        assert route.active_model is route.models["Compact"]

    modeled_reference_components = [
        ref.component
        for ref in layout.component.references
        if ref.component.name != "route_input_grating_to_splitter_root"
    ]
    assert modeled_reference_components
    assert all(component.active_model is not None for component in modeled_reference_components)


def test_pdk_sparameter_audit_fails_closed_without_em_or_measured_models(tmp_path):
    layout = build_lumix_module(LumixLayoutConfig(width=2, layers=1))
    library = PdkSParameterLibrary.from_entries([])

    report = audit_lumix_pdk_sparameter_library(layout, library)

    assert report["status"] == "missing_models"
    assert report["required_roles"]["grating_coupler"]["count"] == 3
    assert report["required_roles"]["splitter_1x2"]["count"] == 1
    assert report["required_roles"]["phase_modulator"]["count"] == 2
    assert report["required_roles"]["inverse_design_region"]["count"] == 1
    assert report["required_roles"]["strip_bend_90"]["count"] == layout.summary["total_route_bends"]
    assert set(report["missing_roles"]) == {
        "grating_coupler",
        "splitter_1x2",
        "phase_modulator",
        "inverse_design_region",
        "strip_straight",
        "strip_bend_90",
    }

    manifest_path = write_lumix_pdk_sparameter_manifest_template(layout, tmp_path / "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["models"]:
        entry["source_type"] = "em_derived"
        entry["path"] = f"{entry['role']}.npz"
    manifest["models"][0]["source_type"] = "analytic"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    library = PdkSParameterLibrary.from_manifest(manifest_path)
    with pytest.raises(MissingPdkSParameterError, match="analytic"):
        library.assert_ready_for(layout)


def test_pdk_sparameter_library_loads_em_npz_as_datamodel(tmp_path):
    frequency = pf.C_0 / 1.55
    s_array = np.zeros((1, 2, 2), dtype=np.complex128)
    s_array[0, 0, 1] = 0.8
    s_array[0, 1, 0] = 0.8
    model_path = tmp_path / "grating_coupler.npz"
    np.savez_compressed(
        model_path,
        frequencies_hz=np.asarray([frequency], dtype=float),
        ports=np.asarray(["Fiber", "P0"]),
        s_array=s_array,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "role": "grating_coupler",
                        "source_type": "em_derived",
                        "path": model_path.name,
                        "component": "SOI220nm_1550nm_TE_STRIP_Grating_Coupler",
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )

    library = PdkSParameterLibrary.from_manifest(manifest_path)
    spec = library.get("grating_coupler")
    model = library.data_model("grating_coupler")

    assert spec.source_type == "em_derived"
    assert spec.path == model_path
    assert isinstance(model, pf.DataModel)
    pf.config.default_technology = cornerstone_soi220_active_technology()
    component = pf.Component("probe")
    component.add_port(pf.Port((0, 0), 0, pf.config.default_technology.ports["strip_1550nm"]), port_name="Fiber")
    component.add_port(pf.Port((1, 0), 180, pf.config.default_technology.ports["strip_1550nm"]), port_name="P0")
    component.add_model(model, "PDK", set_active=True)
    s_matrix = component.s_matrix([frequency], show_progress=False)
    assert s_matrix.elements[("Fiber@0", "P0@0")][0] == pytest.approx(0.8)


def test_pdk_sparameter_library_loads_touchstone_s2p_with_manifest_ports(tmp_path):
    model_path = tmp_path / "grating_coupler.s2p"
    model_path.write_text(
        "\n".join(
            [
                "! grating coupler measured S-parameters",
                "# GHz S MA R 50",
                "193.414489 0.0 0.0 0.7 30.0 0.8 -20.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": [
                    {
                        "role": "grating_coupler",
                        "source_type": "pdk_measured",
                        "path": model_path.name,
                        "ports": ["Fiber", "P0"],
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )

    library = PdkSParameterLibrary.from_manifest(manifest_path)
    s_array, frequencies, ports = library.load_sparameters("grating_coupler")

    assert ports == ["Fiber", "P0"]
    assert frequencies[0] == pytest.approx(193.414489e9)
    assert s_array[0, 0, 1] == pytest.approx(0.7 * np.exp(1j * np.deg2rad(30.0)))
    assert s_array[0, 1, 0] == pytest.approx(0.8 * np.exp(1j * np.deg2rad(-20.0)))


@pytest.mark.parametrize(
    ("format_name", "first", "second", "expected"),
    [
        ("RI", 0.3, 0.4, 0.3 + 0.4j),
        ("DB", -6.0, 45.0, (10 ** (-6.0 / 20.0)) * np.exp(1j * np.deg2rad(45.0))),
    ],
)
def test_pdk_sparameter_library_loads_touchstone_ri_and_db_formats(
    tmp_path,
    format_name,
    first,
    second,
    expected,
):
    model_path = tmp_path / f"phase_modulator_{format_name.lower()}.s2p"
    model_path.write_text(
        "\n".join(
            [
                f"# Hz S {format_name} R 50",
                f"193414489000 0 0 {first} {second} {first} {second} 0 0",
            ]
        )
        + "\n"
    )
    library = PdkSParameterLibrary.from_entries(
        [
            {
                "role": "phase_modulator",
                "source_type": "pdk_measured",
                "path": model_path.name,
                "ports": ["P0", "P1"],
            }
        ],
        base_dir=tmp_path,
    )

    s_array, frequencies, ports = library.load_sparameters("phase_modulator")

    assert ports == ["P0", "P1"]
    assert frequencies[0] == pytest.approx(193414489000.0)
    assert s_array[0, 0, 1] == pytest.approx(expected)


def test_pdk_route_loss_report_uses_straight_and_bend_sparameters(tmp_path):
    layout = build_lumix_module(LumixLayoutConfig(width=4, layers=1))
    frequency = pf.C_0 / 1.55

    def write_two_port(path: Path, loss_db: float) -> None:
        transmission = 10 ** (-loss_db / 20.0)
        s_array = np.zeros((1, 2, 2), dtype=np.complex128)
        s_array[0, 0, 1] = transmission
        s_array[0, 1, 0] = transmission
        np.savez_compressed(
            path,
            frequencies_hz=np.asarray([frequency]),
            ports=np.asarray(["P0", "P1"]),
            s_array=s_array,
        )

    write_two_port(tmp_path / "strip_straight.npz", loss_db=0.1)
    write_two_port(tmp_path / "strip_bend_90.npz", loss_db=0.25)
    library = PdkSParameterLibrary.from_entries(
        [
            {
                "role": "strip_straight",
                "source_type": "em_derived",
                "path": "strip_straight.npz",
                "reference_length_um": 1000.0,
            },
            {
                "role": "strip_bend_90",
                "source_type": "em_derived",
                "path": "strip_bend_90.npz",
            },
        ],
        base_dir=tmp_path,
    )

    report = pdk_route_loss_report(layout, library)

    assert report["source"] == "pdk_sparameters"
    assert report["strip_straight"]["loss_db_per_um"] == pytest.approx(0.1 / 1000.0)
    assert report["strip_bend_90"]["loss_db_per_bend"] == pytest.approx(0.25)
    assert report["total_route_loss_db"] > 0.0
    assert report["max_route_loss_db"] > report["min_route_loss_db"]
    assert report["groups"]["output_fanout"]["loss_spread_db"] > 0.0
    first_route = report["routes"][0]
    assert first_route["loss_db"] == pytest.approx(
        first_route["straight_length_um"] * 0.1 / 1000.0
        + first_route["bend_count"] * 0.25
    )


def test_pdk_route_balance_objective_identifies_loss_imbalance_in_phase_balanced_groups(tmp_path):
    layout = build_lumix_module(LumixLayoutConfig(width=4, layers=1))
    frequency = pf.C_0 / 1.55

    def write_two_port(path: Path, loss_db: float) -> None:
        transmission = 10 ** (-loss_db / 20.0)
        s_array = np.zeros((1, 2, 2), dtype=np.complex128)
        s_array[0, 0, 1] = transmission
        s_array[0, 1, 0] = transmission
        np.savez_compressed(
            path,
            frequencies_hz=np.asarray([frequency]),
            ports=np.asarray(["P0", "P1"]),
            s_array=s_array,
        )

    write_two_port(tmp_path / "strip_straight.npz", loss_db=0.0)
    write_two_port(tmp_path / "strip_bend_90.npz", loss_db=0.25)
    library = PdkSParameterLibrary.from_entries(
        [
            {
                "role": "strip_straight",
                "source_type": "em_derived",
                "path": "strip_straight.npz",
                "reference_length_um": 1000.0,
            },
            {
                "role": "strip_bend_90",
                "source_type": "em_derived",
                "path": "strip_bend_90.npz",
            },
        ],
        base_dir=tmp_path,
    )

    objective = pdk_route_balance_objective(layout, library)

    group = objective["groups"]["L1_mod_to_id"]
    assert group["length_spread_um"] == pytest.approx(0.0)
    assert group["loss_spread_db"] > 0.0
    assert group["status"] == "loss_balancing_required"
    assert objective["phase_balance_objective_um"] == pytest.approx(0.0)
    assert objective["loss_balance_objective_db"] > 0.0


def test_lumix_pdk_sparameter_models_attach_to_every_circuit_block(tmp_path):
    layout = build_lumix_module(LumixLayoutConfig(width=2, layers=1))
    library = _write_lossless_pdk_sparameter_library(tmp_path, width=2)

    report = attach_lumix_pdk_sparameter_models(layout, library)

    assert "PDK" in layout.component.models
    assert report["model_source"] == "pdk_sparameters"
    assert report["fallback_allowed"] is False
    assert report["component_reference_counts"] == {
        "grating_couplers": 3,
        "power_splitters_1x2": 1,
        "phase_modulators": 2,
        "inverse_design_regions": 1,
        "routes": len(layout.route_records),
    }
    assert report["route_model"] == "cascaded_strip_straight_and_bend_sparameters"
    assert report["block_models"]["grating_coupler"]["source_type"] == "em_derived"

    for route in layout.route_components:
        assert "PDK" in route.models
        assert route.active_model is route.models["PDK"]

    modeled_reference_components = [
        ref.component
        for ref in layout.component.references
        if ref.component.name != "route_input_grating_to_splitter_root"
    ]
    assert modeled_reference_components
    assert all(component.active_model is not None for component in modeled_reference_components)
    assert all(component.active_model is component.models["PDK"] for component in modeled_reference_components)


def test_lumix_pdk_sparameter_sweep_uses_strict_models_and_reports_loss(tmp_path):
    layout = build_lumix_module(LumixLayoutConfig(width=2, layers=1))
    library = _write_lossless_pdk_sparameter_library(tmp_path, width=2)

    sweep = run_lumix_pdk_sparameter_sweep(layout, library)

    assert sweep["model_report"]["model_source"] == "pdk_sparameters"
    assert sweep["measurements"]["input_power_w"] == 1.0
    assert sweep["route_loss_report"]["source"] == "pdk_sparameters"
    assert sweep["route_balance_objective"]["source"] == "pdk_sparameters"
    assert sweep["block_loss_report"]["grating_coupler"]["per_instance_loss_db"] == pytest.approx(0.0)
    assert sweep["block_loss_report"]["splitter_1x2"]["excess_loss_db"] == pytest.approx(0.0)
    assert sweep["block_loss_report"]["phase_modulator"]["per_instance_loss_db"] == pytest.approx(0.0)
    assert sweep["block_loss_report"]["inverse_design_region"]["mean_input_loss_db"] == pytest.approx(0.0)
    assert sweep["block_loss_report"]["routes"]["loss_balance_objective_db"] == pytest.approx(0.0)
    assert sweep["points"][0]["total_output_power"] == pytest.approx(1.0)
    assert sweep["points"][0]["output_uniformity_db"] == pytest.approx(0.0)
    assert [record["power"] for record in sweep["points"][0]["s_parameters"]] == pytest.approx([0.5, 0.5])
    assert sweep["points"][0]["total_insertion_loss_db"] == pytest.approx(0.0)


def test_inverse_design_compact_model_uses_supplied_16x16_matrix():
    width = 4
    frequency = pf.C_0 / 1.55
    matrix = np.diag([0.9, 0.8j, -0.7, -0.6j]).astype(np.complex128)
    layout = build_lumix_module(LumixLayoutConfig(width=width, layers=1))

    attach_lumix_compact_models(layout, inverse_design_matrices=[matrix])

    id_component = next(
        ref.component
        for ref in layout.component.references
        if "ID placeholder" in ref.component.name
    )
    s_matrix = id_component.s_matrix([frequency], show_progress=False)

    assert s_matrix.elements[("W00@0", "E00@0")][0] == pytest.approx(matrix[0, 0])
    assert s_matrix.elements[("W01@0", "E01@0")][0] == pytest.approx(matrix[1, 1])
    assert s_matrix.elements[("W00@0", "E01@0")][0] == pytest.approx(0.0)


def test_eo_phase_transmission_encodes_voltage_loss_and_pi_phase():
    transmission = eo_phase_transmission(
        np.array([0.0, 0.5, 1.0]),
        vpi_v=1.0,
        insertion_loss_db=3.0,
    )

    assert np.abs(transmission) == pytest.approx(np.full(3, 10 ** (-3.0 / 20.0)))
    assert transmission[0] == pytest.approx(10 ** (-3.0 / 20.0))
    assert transmission[1] == pytest.approx(1j * 10 ** (-3.0 / 20.0))
    assert transmission[2] == pytest.approx(-10 ** (-3.0 / 20.0))


def test_lumix_circuit_sweep_runs_with_per_channel_voltage_masks():
    width = 2
    layers = 1
    layout = build_lumix_module(LumixLayoutConfig(width=width, layers=layers))
    voltage_masks = np.array(
        [
            [[0.0, 0.0]],
            [[0.5, 1.0]],
        ],
        dtype=float,
    )

    sweep = run_lumix_circuit_sweep(
        layout,
        voltage_masks=voltage_masks,
        compact_config=LumixCompactModelConfig(
            grating_coupling_loss_db=0.0,
            splitter_excess_loss_db=0.0,
            route_loss_db_per_cm=0.0,
            bend_loss_db=0.0,
            phase_modulator_insertion_loss_db=0.0,
        ),
        inverse_design_matrices=[np.eye(width, dtype=np.complex128)],
    )

    assert sweep["wavelength_um"] == 1.55
    assert sweep["input_port"] == "fiber_in"
    assert sweep["output_ports"] == ["fiber_out_00", "fiber_out_01"]
    assert sweep["measurements"]["input_power_w"] == 1.0
    assert sweep["modeled_losses"]["grating_couplers"] == {
        "count": 3,
        "per_device_loss_db": 0.0,
        "input_to_output_path_loss_db": 0.0,
    }
    assert sweep["modeled_losses"]["splitter_tree"]["stages_per_path"] == 1
    assert sweep["modeled_losses"]["splitter_tree"]["ideal_power_division_db_per_output"] == pytest.approx(10 * np.log10(width))
    assert sweep["modeled_losses"]["phase_modulators"]["layers_per_path"] == layers
    assert sweep["modeled_losses"]["routes"]["total_layout_loss_db"] == pytest.approx(0.0)
    assert len(sweep["points"]) == 2
    assert sweep["points"][0]["voltages"] == [[0.0, 0.0]]
    assert sweep["points"][1]["voltages"] == [[0.5, 1.0]]
    assert len(sweep["points"][0]["output_powers"]) == width
    assert sweep["points"][0]["total_output_power"] > 0.0
    assert sweep["points"][0]["total_insertion_loss_db"] == pytest.approx(-10 * np.log10(sweep["points"][0]["total_output_power"]))
    assert sweep["points"][0]["total_insertion_loss_db"] == pytest.approx(0.0)
    assert sweep["points"][0]["output_uniformity_db"] == pytest.approx(0.0)
    assert len(sweep["points"][0]["s_parameters"]) == width
    assert sweep["points"][0]["s_parameters"][0]["source"] == "fiber_in"
    assert sweep["points"][0]["s_parameters"][0]["output"] == "fiber_out_00"
    assert sweep["points"][0]["s_parameters"][0]["power"] == pytest.approx(0.5)
    assert sweep["points"][0]["s_parameters"][0]["transmission_db"] == pytest.approx(-10 * np.log10(width))
    assert sweep["points"][1]["total_output_power"] == pytest.approx(
        sweep["points"][0]["total_output_power"]
    )


def _write_lossless_pdk_sparameter_library(tmp_path: Path, *, width: int) -> PdkSParameterLibrary:
    tmp_path.mkdir(parents=True, exist_ok=True)
    frequency = pf.C_0 / 1.55

    def write_model(name: str, ports: list[str], pairs: dict[tuple[str, str], complex]) -> None:
        s_array = np.zeros((1, len(ports), len(ports)), dtype=np.complex128)
        index = {port: i for i, port in enumerate(ports)}
        for (port0, port1), value in pairs.items():
            s_array[0, index[port0], index[port1]] = value
        np.savez_compressed(
            tmp_path / f"{name}.npz",
            frequencies_hz=np.asarray([frequency], dtype=float),
            ports=np.asarray(ports),
            s_array=s_array,
        )

    write_model("grating_coupler", ["Fiber", "P0"], {("Fiber", "P0"): 1.0, ("P0", "Fiber"): 1.0})
    splitter_t = 1.0 / np.sqrt(2.0)
    write_model(
        "splitter_1x2",
        ["P0", "P_HI", "P_LO"],
        {
            ("P0", "P_HI"): splitter_t,
            ("P_HI", "P0"): splitter_t,
            ("P0", "P_LO"): splitter_t,
            ("P_LO", "P0"): splitter_t,
        },
    )
    write_model("phase_modulator", ["P0", "P1"], {("P0", "P1"): 1.0, ("P1", "P0"): 1.0})
    inverse_ports = [f"W{index:02d}" for index in range(width)] + [f"E{index:02d}" for index in range(width)]
    inverse_pairs = {}
    for channel in range(width):
        inverse_pairs[(f"W{channel:02d}", f"E{channel:02d}")] = 1.0
        inverse_pairs[(f"E{channel:02d}", f"W{channel:02d}")] = 1.0
    write_model("inverse_design_region", inverse_ports, inverse_pairs)
    write_model("strip_straight", ["P0", "P1"], {("P0", "P1"): 1.0, ("P1", "P0"): 1.0})
    write_model("strip_bend_90", ["P0", "P1"], {("P0", "P1"): 1.0, ("P1", "P0"): 1.0})

    return PdkSParameterLibrary.from_entries(
        [
            {"role": "grating_coupler", "source_type": "em_derived", "path": "grating_coupler.npz"},
            {"role": "splitter_1x2", "source_type": "em_derived", "path": "splitter_1x2.npz"},
            {"role": "phase_modulator", "source_type": "em_derived", "path": "phase_modulator.npz"},
            {
                "role": "inverse_design_region",
                "source_type": "em_derived",
                "path": "inverse_design_region.npz",
            },
            {
                "role": "strip_straight",
                "source_type": "em_derived",
                "path": "strip_straight.npz",
                "reference_length_um": 1000.0,
            },
            {"role": "strip_bend_90", "source_type": "em_derived", "path": "strip_bend_90.npz"},
        ],
        base_dir=tmp_path,
    )


def _expected_cspdk_version() -> str:
    try:
        return version("cspdk")
    except PackageNotFoundError:
        return "source-checkout"
