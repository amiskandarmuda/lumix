from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization

from lumix.functional.subunitary import insertion_loss_bounds, subunitary_matrix
from lumix.functional.unitary import combine_complex_parts
from lumix.linen.subunitary import SubUnitaryLinear


def _device_spec():
    import lumix.inverse_design as lid

    return lid.DeviceDesignSpec(
        simulation_mode="2d_effective_index",
        wavelength_um=1.55,
        background_eps=1.44**2,
        core_eps=3.48**2,
        core_thickness_um=0.22,
        design_region_size_um=(8.0, 6.0),
        input_pitch_um=0.9,
        output_pitch_um=0.9,
        device_margin_um=3.0,
        pixel_size_um=0.1,
    )


def _curved_device_spec():
    import lumix.inverse_design as lid

    return lid.DeviceDesignSpec(
        simulation_mode="2d_effective_index",
        wavelength_um=1.55,
        background_eps=1.44**2,
        core_eps=3.48**2,
        core_thickness_um=0.22,
        design_region_size_um=(8.0, 6.0),
        input_pitch_um=0.9,
        output_pitch_um=0.9,
        device_margin_um=3.0,
        pixel_size_um=0.1,
        curved_container=lid.CurvedContainerSpec(
            enabled=True,
            corner_radius_um=1.0,
            box_thickness_um=0.3,
            inner_overlap_px=1,
            taper_overlap_px=1,
        ),
    )


def _linear_taper_16x16_device_spec():
    import lumix.inverse_design as lid

    return lid.DeviceDesignSpec(
        simulation_mode="2d_effective_index",
        wavelength_um=1.55,
        background_eps=1.44**2,
        core_eps=3.48**2,
        core_thickness_um=0.22,
        design_region_size_um=(21.0, 21.0),
        input_pitch_um=1.25,
        output_pitch_um=1.25,
        device_margin_um=3.0,
        pixel_size_um=0.05,
        waveguide_width_um=0.45,
        curved_container=lid.CurvedContainerSpec(
            enabled=True,
            corner_radius_um=0.8,
            box_thickness_um=0.5 * 0.55,
            inner_overlap_px=2,
            taper_overlap_px=2,
        ),
        port_taper=lid.PortTaperSpec(
            mouth_width_um=1.25,
            waveguide_width_um=0.45,
            length_um=3.1,
            samples=101,
            initial_profile="linear",
            mouth_mode="adjacent_touch",
            mouth_gap_um=0.0,
        ),
    )


def _profile_taper_device_spec(profile: str):
    import lumix.inverse_design as lid

    return lid.DeviceDesignSpec(
        simulation_mode="2d_effective_index",
        wavelength_um=1.55,
        background_eps=1.44**2,
        core_eps=3.48**2,
        core_thickness_um=0.22,
        design_region_size_um=(21.0, 21.0),
        input_pitch_um=1.25,
        output_pitch_um=1.25,
        device_margin_um=3.0,
        pixel_size_um=0.05,
        waveguide_width_um=0.45,
        curved_container=lid.CurvedContainerSpec(
            enabled=True,
            corner_radius_um=0.8,
            box_thickness_um=0.5 * 0.55,
            inner_overlap_px=2,
            taper_overlap_px=2,
        ),
        port_taper=lid.PortTaperSpec(
            mouth_width_um=3.1,
            waveguide_width_um=0.45,
            length_um=3.1,
            samples=101,
            initial_profile=profile,
            mouth_mode="fixed",
        ),
    )


def _adaptive_taper_device_spec(*, mouth_gap_um: float = 0.0):
    import lumix.inverse_design as lid

    return lid.DeviceDesignSpec(
        simulation_mode="2d_effective_index",
        wavelength_um=1.55,
        background_eps=1.44**2,
        core_eps=3.48**2,
        core_thickness_um=0.22,
        design_region_size_um=(21.0, 21.0),
        input_pitch_um=1.25,
        output_pitch_um=1.25,
        device_margin_um=3.0,
        pixel_size_um=0.05,
        waveguide_width_um=0.45,
        curved_container=lid.CurvedContainerSpec(
            enabled=True,
            corner_radius_um=0.8,
            box_thickness_um=0.5 * 0.55,
            inner_overlap_px=2,
            taper_overlap_px=2,
        ),
        port_taper=lid.PortTaperSpec(
            mouth_width_um=1.25,
            waveguide_width_um=0.45,
            length_um=3.1,
            samples=101,
            initial_profile="linear",
            mouth_mode="adjacent_touch",
            mouth_gap_um=mouth_gap_um,
        ),
    )


def test_matrix_array_rejects_nonpassive_target_by_default():
    import lumix.inverse_design as lid

    matrix = 2.0 * np.eye(2, dtype=np.complex128)

    with pytest.raises(ValueError, match="non-passive"):
        lid.from_matrix_array(matrix, device=_device_spec())


def test_matrix_array_can_normalize_nonpassive_target():
    import lumix.inverse_design as lid

    matrix = 2.0 * np.eye(2, dtype=np.complex128)
    objective = lid.MatrixObjectiveSpec(passivity=lid.PassivityPolicy(mode="normalize"))

    template = lid.from_matrix_array(matrix, device=_device_spec(), objective=objective)

    singular_values = np.linalg.svd(template.target_matrix, compute_uv=False)
    assert template.port_counts.n_input == 2
    assert template.port_counts.n_output == 2
    assert np.max(singular_values) == pytest.approx(1.0)


def test_lumix_checkpoint_extracts_named_subunitary_matrix(tmp_path):
    import lumix.inverse_design as lid

    class OpticalClassifier(nn.Module):
        @nn.compact
        def __call__(self, values):
            return SubUnitaryLinear(
                width=4,
                out_features=3,
                insertion_loss_db=(0.5, 2.0),
                name="classifier_optical",
            )(values)

    model = OpticalClassifier()
    sample_x = (jnp.ones((2, 4)) + 1j * jnp.ones((2, 4))).astype(jnp.complex64)
    variables = model.init(jax.random.key(0), sample_x)
    checkpoint_path = tmp_path / "params.msgpack"
    checkpoint_path.write_bytes(serialization.to_bytes(variables["params"]))

    layer_params = variables["params"]["classifier_optical"]
    expected = subunitary_matrix(
        combine_complex_parts(layer_params["left_re"], layer_params["left_im"]),
        combine_complex_parts(layer_params["right_re"], layer_params["right_im"]),
        layer_params["singular_raw"],
        *insertion_loss_bounds((0.5, 2.0)),
        3,
        4,
    )

    template = lid.from_lumix_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        sample_x=sample_x,
        layer=lid.LayerSelector(name="classifier_optical"),
        device=_device_spec(),
    )

    np.testing.assert_allclose(template.target_matrix, np.asarray(expected), atol=1e-5)
    assert template.port_counts.n_input == 4
    assert template.port_counts.n_output == 3


def test_lumix_checkpoint_ambiguous_layer_name_fails(tmp_path):
    import lumix.inverse_design as lid

    class OpticalBlock(nn.Module):
        @nn.compact
        def __call__(self, values):
            return SubUnitaryLinear(width=4, out_features=4, insertion_loss_db=1.0, name="optical")(values)

    class TwoBlockModel(nn.Module):
        @nn.compact
        def __call__(self, values):
            left = OpticalBlock(name="left")(values)
            right = OpticalBlock(name="right")(values)
            return left + right

    model = TwoBlockModel()
    sample_x = (jnp.ones((2, 4)) + 1j * jnp.ones((2, 4))).astype(jnp.complex64)
    variables = model.init(jax.random.key(1), sample_x)
    checkpoint_path = tmp_path / "params.msgpack"
    checkpoint_path.write_bytes(serialization.to_bytes(variables["params"]))

    with pytest.raises(ValueError, match="Ambiguous layer selection"):
        lid.from_lumix_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            sample_x=sample_x,
            layer=lid.LayerSelector(name="optical"),
            device=_device_spec(),
        )

    template = lid.from_lumix_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        sample_x=sample_x,
        layer=lid.LayerSelector(path="left/optical"),
        device=_device_spec(),
    )
    assert template.target_matrix.shape == (4, 4)


def test_base_simulation_is_source_free_tidy3d_template():
    td = pytest.importorskip("tidy3d")
    tdi = pytest.importorskip("tidy3d.plugins.invdes")
    import lumix.inverse_design as lid

    topology = lid.TopologyRegionSpec(
        transformations=(tdi.FilterProject(radius=0.2, beta=12.0, eta=0.5),),
        penalties=(tdi.ErosionDilationPenalty(weight=0.05, length_scale=0.15),),
    )
    template = lid.from_matrix_array(np.eye(2, dtype=np.complex128), device=_device_spec(), topology=topology)

    sim = template.base_simulation

    assert isinstance(sim, td.Simulation)
    assert len(sim.sources) == 0
    assert len(sim.monitors) == 0
    assert sim.size[2] == pytest.approx(0.0)
    assert template.design_region.transformations == topology.transformations
    assert template.design_region.penalties == topology.penalties


def test_curved_container_uses_curved_design_region_fixed_ring_and_box_mesh_override():
    td = pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    topology = lid.TopologyRegionSpec(override_structure_dl=0.2)
    template = lid.from_matrix_array(np.eye(2, dtype=np.complex128), device=_curved_device_spec(), topology=topology)

    design_geometry = template.design_region.geometry
    base_sim = template.base_simulation
    preview_sim = template.base_simulation_with_params(template.initial_design_params)
    override = preview_sim.grid_spec.override_structures[0]

    assert not isinstance(design_geometry, td.Box)
    np.testing.assert_allclose(np.asarray(design_geometry.bounds[0])[:2], np.array([-4.0, -3.0]), atol=1e-9)
    np.testing.assert_allclose(np.asarray(design_geometry.bounds[1])[:2], np.array([4.0, 3.0]), atol=1e-9)
    assert len(base_sim.sources) == 0
    assert len(base_sim.monitors) == 0
    assert len(base_sim.structures) == 5
    assert len(preview_sim.structures) == 6
    assert isinstance(override.geometry, td.Box)
    assert override.geometry.bounds == design_geometry.bounding_box.bounds
    assert override.dl == (0.2, 0.2, None)


def test_curved_container_mask_zeroes_pixels_outside_rounded_design_region():
    pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    template = lid.from_matrix_array(np.eye(2, dtype=np.complex128), device=_curved_device_spec())
    params = np.ones(template.design_region.params_shape, dtype=np.float64)

    masked = template.apply_design_mask(params)

    assert masked.shape == params.shape
    assert masked[0, 0, 0] == pytest.approx(0.0)
    assert masked[masked.shape[0] // 2, masked.shape[1] // 2, 0] == pytest.approx(1.0)


def test_port_taper_defaults_match_current_16x16_matrix_scaffold():
    import lumix.inverse_design as lid

    device = _linear_taper_16x16_device_spec()
    taper = lid.PortTaperSpec()

    assert device.lead_length_min_wavelengths == pytest.approx(1.5)
    assert device.monitor_distance_um is None
    assert device.source_distance_um is None
    assert device.pml_gap_um is None
    assert device.run_time_ps == pytest.approx(10.0)
    assert not device.include_input_monitors
    assert taper.mouth_width_um == pytest.approx(1.25)
    assert taper.length_um == pytest.approx(3.1)
    assert taper.samples == 101
    assert taper.initial_profile == "linear"
    assert taper.mouth_mode == "adjacent_touch"
    assert taper.mouth_gap_um == pytest.approx(0.0)


def test_reference_16x16_template_uses_one_source_free_base_model():
    td = pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid
    from lumix.inverse_design.tidy3d_builder import _x_positions

    target = np.zeros((16, 16), dtype=np.complex128)
    device = lid.DeviceDesignSpec(
        simulation_mode="2d_effective_index",
        wavelength_um=1.55,
        background_eps=2.0867588783784097,
        core_eps=8.118899128977837,
        design_eps_bounds=(2.544352165222565**2, 2.849368198211287**2),
        core_thickness_um=0.22,
        design_region_size_um=(21.0, 21.0),
        input_pitch_um=1.25,
        output_pitch_um=1.25,
        waveguide_width_um=0.45,
        mode_window_um=1.2,
        device_margin_um=1.55,
        pixel_size_um=0.05,
        min_steps_per_wvl=13,
        curved_container=lid.CurvedContainerSpec(
            enabled=True,
            corner_radius_um=0.8,
            box_thickness_um=0.5 * 0.55,
            inner_overlap_px=2,
            taper_overlap_px=2,
        ),
        port_taper=lid.PortTaperSpec(
            mouth_width_um=1.25,
            waveguide_width_um=0.45,
            length_um=3.1,
            samples=101,
            initial_profile="linear",
            mouth_mode="adjacent_touch",
            mouth_gap_um=0.0,
        ),
    )
    template = lid.from_matrix_array(target, device=device, topology=lid.TopologyRegionSpec(override_structure_dl=0.1))

    base_sim = template.base_simulation
    input0 = template.excitation_plan.simulation_for_input(0)
    x_positions = _x_positions(device)
    x_input_taper_start = x_positions["x_left"] - device.port_taper.length_um
    x_output_taper_end = x_positions["x_right"] + device.port_taper.length_um

    assert template.port_counts.n_input == 16
    assert template.port_counts.n_output == 16
    assert tuple(template.design_region.eps_bounds) == pytest.approx((2.544352165222565**2, 2.849368198211287**2))
    assert len(base_sim.sources) == 0
    assert len(base_sim.monitors) == 0
    assert len(base_sim.structures) == 16 * 2 + 16 * 2 + 1
    assert len(template.excitation_plan.input_port_names) == 16
    assert len(input0.sources) == 1
    assert len(input0.monitors) == 16
    assert isinstance(base_sim, td.Simulation)
    assert base_sim.run_time == pytest.approx(10e-12)
    assert x_input_taper_start - x_positions["x_source"] == pytest.approx(1.5 * device.wavelength_um)
    assert x_input_taper_start - x_positions["x_in_monitor"] == pytest.approx(1.5 * device.wavelength_um)
    assert x_positions["x_out_monitor"] - x_output_taper_end == pytest.approx(1.5 * device.wavelength_um)
    assert x_positions["x_source"] - x_positions["x_input_start"] == pytest.approx(1.5 * device.wavelength_um)
    assert x_positions["x_output_end"] - x_positions["x_out_monitor"] == pytest.approx(1.5 * device.wavelength_um)
    assert input0.sources[0].name == "src_mode_0"
    assert not any(monitor.name.startswith("in_mode_") for monitor in input0.monitors)
    assert [monitor.name for monitor in input0.monitors if monitor.name.startswith("out_mode_")] == [
        f"out_mode_{index}" for index in range(16)
    ]


def test_excitation_plan_builds_one_source_simulation_per_input():
    td = pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    template = lid.from_matrix_array(np.eye(2, dtype=np.complex128), device=_device_spec())

    input0 = template.excitation_plan.simulation_for_input(0)
    input1 = template.excitation_plan.simulation_for_input(1)

    assert isinstance(input0, td.Simulation)
    assert len(input0.sources) == 1
    assert len(input1.sources) == 1
    assert input0.sources[0].name == "src_mode_0"
    assert input1.sources[0].name == "src_mode_1"
    assert {monitor.name for monitor in input0.monitors} == {
        "out_mode_0",
        "out_mode_1",
    }


def test_excitation_plan_can_include_input_monitors_when_requested():
    td = pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    device = replace(_device_spec(), include_input_monitors=True)
    template = lid.from_matrix_array(np.eye(2, dtype=np.complex128), device=device)

    input0 = template.excitation_plan.simulation_for_input(0)

    assert isinstance(input0, td.Simulation)
    assert {monitor.name for monitor in input0.monitors} == {
        "in_mode_0",
        "in_mode_1",
        "out_mode_0",
        "out_mode_1",
    }


def test_linear_port_taper_spec_builds_current_16x16_polygon_tapers():
    td = pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    target = np.zeros((16, 16), dtype=np.complex128)
    template = lid.from_matrix_array(target, device=_linear_taper_16x16_device_spec())

    params = template.initial_optimization_params(scope="taper")
    sim = template.base_simulation_with_params(params)
    design_geometry = template.design_region.geometry

    assert params.matrix.shape == template.design_region.params_shape
    assert params.input_taper_widths_um.shape == (16, 101)
    assert params.output_taper_widths_um.shape == (16, 101)
    assert not params.optimize_matrix
    assert params.optimize_tapers
    assert not isinstance(design_geometry, td.Box)
    assert len(sim.structures) == 66

    u = np.linspace(0.0, 1.0, 101)
    expected_output = 1.25 + (0.45 - 1.25) * u
    expected_input = 0.45 + (1.25 - 0.45) * u
    np.testing.assert_allclose(params.output_taper_widths_um[0], expected_output)
    np.testing.assert_allclose(params.input_taper_widths_um[0], expected_input)

    taper_structures = [
        structure
        for structure in sim.structures
        if isinstance(structure.geometry, td.PolySlab) and len(structure.geometry.vertices) == 202
    ]
    assert len(taper_structures) == 32

    first_output_taper = taper_structures[16]
    vertices = np.asarray(first_output_taper.geometry.vertices)
    assert np.ptp(vertices[:, 0]) == pytest.approx(3.1)
    assert np.ptp(vertices[:, 1]) == pytest.approx(1.25)

    x_min = sim.center[0] - 0.5 * sim.size[0]
    x_max = sim.center[0] + 0.5 * sim.size[0]
    for structure in sim.structures:
        bounds = structure.geometry.bounds
        assert float(bounds[0][0]) >= x_min - 1e-9
        assert float(bounds[1][0]) <= x_max + 1e-9

    input_source_sim = template.excitation_plan.simulation_for_input(0)
    source_x = input_source_sim.sources[0].center[0]
    first_input_taper = taper_structures[0]
    taper_x_min = float(first_input_taper.geometry.bounds[0][0])
    assert source_x < taper_x_min


def test_device_spec_can_reduce_minimum_lead_length_to_source_monitor_buffer():
    pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid
    from lumix.inverse_design.tidy3d_builder import _x_positions

    wavelength_um = 1.55
    buffer_um = 1.5 * wavelength_um
    source_monitor_offset_um = 2.2
    device = lid.DeviceDesignSpec(
        simulation_mode="2d_effective_index",
        wavelength_um=wavelength_um,
        background_eps=1.44**2,
        core_eps=3.48**2,
        core_thickness_um=0.22,
        design_region_size_um=(21.0, 21.0),
        input_pitch_um=1.25,
        output_pitch_um=1.25,
        waveguide_width_um=0.45,
        device_margin_um=3.0,
        pixel_size_um=0.05,
        monitor_distance_um=source_monitor_offset_um,
        source_distance_um=source_monitor_offset_um,
        pml_gap_um=buffer_um,
        lead_length_min_wavelengths=1.5,
        port_taper=lid.PortTaperSpec(
            mouth_width_um=1.25,
            waveguide_width_um=0.45,
            length_um=3.1,
            samples=101,
            initial_profile="linear",
            mouth_mode="adjacent_touch",
        ),
    )

    positions = _x_positions(device)

    assert positions["x_source"] - positions["x_input_start"] == pytest.approx(buffer_um)
    assert positions["x_output_end"] - positions["x_out_monitor"] == pytest.approx(buffer_um)


def test_initial_optimization_params_switches_between_matrix_taper_and_joint_scopes():
    pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    template = lid.from_matrix_array(np.zeros((3, 4), dtype=np.complex128), device=_linear_taper_16x16_device_spec())

    matrix_params = template.initial_optimization_params(scope="matrix")
    taper_params = template.initial_optimization_params(scope="taper")
    joint_params = template.initial_optimization_params(scope="matrix_and_taper")

    assert matrix_params.optimize_matrix
    assert not matrix_params.optimize_tapers
    assert matrix_params.input_taper_widths_um is None
    assert matrix_params.output_taper_widths_um is None

    assert not taper_params.optimize_matrix
    assert taper_params.optimize_tapers
    assert taper_params.matrix.shape == template.design_region.params_shape
    assert taper_params.input_taper_widths_um.shape == (4, 101)
    assert taper_params.output_taper_widths_um.shape == (3, 101)

    assert joint_params.optimize_matrix
    assert joint_params.optimize_tapers
    assert joint_params.matrix.shape == template.design_region.params_shape
    assert joint_params.input_taper_widths_um.shape == (4, 101)
    assert joint_params.output_taper_widths_um.shape == (3, 101)

    with pytest.raises(ValueError, match="Unsupported optimization scope"):
        template.initial_optimization_params(scope="ports")


@pytest.mark.parametrize(
    "profile",
    ["linear", "quadratic", "raised_cosine", "inverted_quarter_circle", "local_adiabatic"],
)
def test_port_taper_profile_presets_generate_valid_taper_width_arrays(profile):
    pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    template = lid.from_matrix_array(
        np.zeros((3, 4), dtype=np.complex128),
        device=_profile_taper_device_spec(profile),
    )

    params = template.initial_optimization_params(scope="taper")

    assert params.input_taper_widths_um.shape == (4, 101)
    assert params.output_taper_widths_um.shape == (3, 101)
    assert params.input_taper_widths_um[0, 0] == pytest.approx(0.45)
    assert params.input_taper_widths_um[0, -1] == pytest.approx(3.1)
    assert params.output_taper_widths_um[0, 0] == pytest.approx(3.1)
    assert params.output_taper_widths_um[0, -1] == pytest.approx(0.45)
    assert np.all(np.diff(params.input_taper_widths_um[0]) >= -1e-12)
    assert np.all(np.diff(params.output_taper_widths_um[0]) <= 1e-12)


def test_local_adiabatic_port_taper_profile_uses_square_root_width_law():
    pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    template = lid.from_matrix_array(
        np.zeros((3, 4), dtype=np.complex128),
        device=_profile_taper_device_spec("local_adiabatic"),
    )

    params = template.initial_optimization_params(scope="taper")

    u = np.linspace(0.0, 1.0, 101)
    expected_input = np.sqrt(0.45**2 + (3.1**2 - 0.45**2) * u)
    expected_output = np.sqrt(3.1**2 + (0.45**2 - 3.1**2) * u)
    np.testing.assert_allclose(params.input_taper_widths_um[0], expected_input)
    np.testing.assert_allclose(params.output_taper_widths_um[0], expected_output)
    assert params.input_taper_widths_um.shape == (4, 101)
    assert params.output_taper_widths_um.shape == (3, 101)
    assert params.output_taper_widths_um[0, 0] == pytest.approx(3.1)
    assert params.output_taper_widths_um[0, -1] == pytest.approx(0.45)
    assert abs(params.output_taper_widths_um[0, 1] - params.output_taper_widths_um[0, 0]) < abs(
        params.output_taper_widths_um[0, -1] - params.output_taper_widths_um[0, -2]
    )


def test_adjacent_touch_port_taper_mouths_fill_pitch_without_gaps():
    td = pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    template = lid.from_matrix_array(np.zeros((16, 16), dtype=np.complex128), device=_adaptive_taper_device_spec())

    params = template.initial_optimization_params(scope="taper")
    sim = template.base_simulation_with_params(params)
    taper_structures = [
        structure
        for structure in sim.structures
        if isinstance(structure.geometry, td.PolySlab) and len(structure.geometry.vertices) == 202
    ]
    input_mouths = [_mouth_bounds_at_x(structure, x_value=-10.5) for structure in taper_structures[:16]]
    output_mouths = [_mouth_bounds_at_x(structure, x_value=10.5) for structure in taper_structures[16:]]

    np.testing.assert_allclose(params.input_taper_widths_um[:, -1], np.full(16, 1.25))
    np.testing.assert_allclose(params.output_taper_widths_um[:, 0], np.full(16, 1.25))
    assert input_mouths[0] == pytest.approx((-10.0, -8.75))
    assert input_mouths[-1] == pytest.approx((8.75, 10.0))
    assert output_mouths[0] == pytest.approx((-10.0, -8.75))
    assert output_mouths[-1] == pytest.approx((8.75, 10.0))
    for left, right in zip(input_mouths, input_mouths[1:]):
        assert left[1] == pytest.approx(right[0])
    for left, right in zip(output_mouths, output_mouths[1:]):
        assert left[1] == pytest.approx(right[0])


def test_adjacent_touch_port_taper_can_keep_fabrication_gap_between_mouths():
    pytest.importorskip("tidy3d")
    import lumix.inverse_design as lid

    template = lid.from_matrix_array(
        np.zeros((16, 16), dtype=np.complex128),
        device=_adaptive_taper_device_spec(mouth_gap_um=0.05),
    )

    params = template.initial_optimization_params(scope="taper")

    np.testing.assert_allclose(params.input_taper_widths_um[:, -1], np.full(16, 1.20))
    np.testing.assert_allclose(params.output_taper_widths_um[:, 0], np.full(16, 1.20))


def _mouth_bounds_at_x(structure, *, x_value: float) -> tuple[float, float]:
    vertices = np.asarray(structure.geometry.vertices)
    y_values = vertices[np.isclose(vertices[:, 0], float(x_value), atol=1e-9), 1]
    assert y_values.size >= 2
    return float(np.min(y_values)), float(np.max(y_values))
