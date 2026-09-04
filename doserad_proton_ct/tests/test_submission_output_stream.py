"""Contract tests for the streaming 4D dose-stack writer.

The submission adapter used to hold every dose map of a run in host memory and
only serialize at the end. For the representative proton case of 500 beamlets
that is 53 GiB on the median training grid and 83 GiB on the largest, while the
A10G instances offered by Grand Challenge cap out at 31 GiB of usable DRAM.
These tests pin the replacement writer to two guarantees: the file it produces
is byte-identical to the previous ``sitk.JoinSeries`` output, and peak memory
stays proportional to a single frame rather than to the stack.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WORKSPACE / "doserad_photon_ct" / "src"))
sys.path.insert(0, str(ROOT / "submission"))

import inference as submission  # noqa: E402
from doserad_photon_ct.conditioning import SpatialGeometry  # noqa: E402
from doserad_proton.conditioning import ProtonCondition, build_proton_channels  # noqa: E402
from doserad_proton.data import _mri_bounds  # noqa: E402


def _reference_image(size_xyz=(5, 4, 3)) -> sitk.Image:
    """A 3D grid with deliberately awkward, non-round geometry."""
    image = sitk.Image(list(size_xyz), sitk.sitkFloat32)
    image.SetSpacing((0.9765625, 1.3333333333333333, 2.5))
    image.SetOrigin((-249.51171875, 3.25, -70.7))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    return image


def _frames(reference: sitk.Image, count: int) -> list[np.ndarray]:
    size_x, size_y, size_z = reference.GetSize()
    rng = np.random.default_rng(20260816)
    return [
        rng.random((size_z, size_y, size_x), dtype=np.float32) for _ in range(count)
    ]


def _join_series_bytes(reference: sitk.Image, frames, path: Path) -> bytes:
    """Serialize the same stack the old code path would have produced."""
    stack = sitk.JoinSeries([submission._to_sitk(frame, reference) for frame in frames])
    sitk.WriteImage(stack, str(path), useCompression=False)
    return path.read_bytes()


def test_streamed_stack_is_byte_identical_to_join_series(tmp_path):
    reference = _reference_image()
    frames = _frames(reference, 4)

    expected = _join_series_bytes(reference, frames, tmp_path / "join.mha")

    streamed_path = tmp_path / "streamed" / "output.mha"
    writer = submission.StackWriter(streamed_path, reference, len(frames))
    for index, frame in enumerate(frames):
        writer.write(index, frame)
    writer.close()

    assert streamed_path.read_bytes() == expected


def test_out_of_order_writes_land_in_the_declared_frame(tmp_path):
    reference = _reference_image()
    frames = _frames(reference, 5)

    expected = _join_series_bytes(reference, frames, tmp_path / "join.mha")

    streamed_path = tmp_path / "streamed" / "output.mha"
    writer = submission.StackWriter(streamed_path, reference, len(frames))
    for index in (3, 0, 4, 2, 1):
        writer.write(index, frames[index])
    writer.close()

    assert streamed_path.read_bytes() == expected


def test_streamed_stack_round_trips_through_simpleitk(tmp_path):
    reference = _reference_image()
    frames = _frames(reference, 3)
    streamed_path = tmp_path / "output.mha"

    writer = submission.StackWriter(streamed_path, reference, len(frames))
    for index, frame in enumerate(frames):
        writer.write(index, frame)
    writer.close()

    restored = sitk.ReadImage(str(streamed_path))
    joined = sitk.JoinSeries([submission._to_sitk(frame, reference) for frame in frames])
    assert restored.GetSize() == joined.GetSize()
    assert restored.GetSpacing() == pytest.approx(joined.GetSpacing())
    assert restored.GetOrigin() == pytest.approx(joined.GetOrigin())
    assert restored.GetDirection() == pytest.approx(joined.GetDirection())
    np.testing.assert_array_equal(
        sitk.GetArrayFromImage(restored), np.stack(frames, axis=0)
    )


def test_peak_memory_tracks_one_frame_not_the_stack(tmp_path):
    reference = _reference_image((40, 40, 40))
    frame_bytes = 40 * 40 * 40 * 4
    frame_count = 48
    stack_bytes = frame_bytes * frame_count
    rng = np.random.default_rng(7)
    writer = submission.StackWriter(tmp_path / "output.mha", reference, frame_count)

    tracemalloc.start()
    try:
        for index in range(frame_count):
            writer.write(index, rng.random((40, 40, 40), dtype=np.float32))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    writer.close()

    # Generous bound: a few frames of slack, but far below the whole stack.
    assert peak < 4 * frame_bytes
    assert peak < stack_bytes // 4
    assert (tmp_path / "output.mha").stat().st_size == len(
        submission._metaimage_header(reference, frame_count)
    ) + stack_bytes


def test_close_rejects_an_incomplete_stack(tmp_path):
    reference = _reference_image()
    writer = submission.StackWriter(tmp_path / "output.mha", reference, 3)
    writer.write(0, _frames(reference, 1)[0])
    with pytest.raises(ValueError, match=r"missing frames \[1, 2\]"):
        writer.close()


def test_writer_rejects_duplicate_and_mismatched_frames(tmp_path):
    reference = _reference_image()
    frame = _frames(reference, 1)[0]
    writer = submission.StackWriter(tmp_path / "output.mha", reference, 2)
    writer.write(0, frame)
    with pytest.raises(ValueError, match="already written"):
        writer.write(0, frame)
    with pytest.raises(ValueError, match="outside 0..1"):
        writer.write(5, frame)
    with pytest.raises(ValueError, match="does not match the input grid"):
        writer.write(1, np.zeros((2, 2, 2), dtype=np.float32))
    writer.discard()


def _metadata(entries):
    """Build proton metadata from (image_idx, [(out_idx, idx_in_output)]) pairs."""
    return [
        {
            "image_file_idx": image_idx,
            "beams": [
                {
                    "gantry_angle": 0.0,
                    "rays": [
                        {
                            "ray_source": [0.0, -1000.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            "beamlets": [
                                {
                                    "energy": 100.0,
                                    "output_info": {
                                        "output_file_idx": out_idx,
                                        "idx_in_output": stack_idx,
                                        "minimum_cutoff": 0.02,
                                    },
                                }
                                for out_idx, stack_idx in slots
                            ],
                        }
                    ],
                }
            ],
        }
        for image_idx, slots in entries
    ]


def test_plan_outputs_counts_frames_per_slot():
    metadata = _metadata(
        [
            (0, [(0, 0), (0, 2), (0, 1), (1, 0)]),
            (3, [(2, 1), (2, 0)]),
        ]
    )
    plans = submission.plan_outputs(metadata)
    assert len(plans) == submission.NUM_OUTPUT_FILES
    assert plans[0] == submission.OutputPlan(image_file_idx=0, frame_count=3)
    assert plans[1] == submission.OutputPlan(image_file_idx=0, frame_count=1)
    assert plans[2] == submission.OutputPlan(image_file_idx=3, frame_count=2)
    assert all(plan is None for plan in plans[3:])


def test_plan_outputs_rejects_gaps():
    with pytest.raises(ValueError, match="has gaps"):
        submission.plan_outputs(_metadata([(0, [(0, 0), (0, 2)])]))


def test_plan_outputs_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate output position"):
        submission.plan_outputs(_metadata([(0, [(0, 0)]), (0, [(0, 0)])]))


def test_plan_outputs_rejects_a_slot_fed_by_two_images():
    with pytest.raises(ValueError, match="mixes input images"):
        submission.plan_outputs(_metadata([(0, [(0, 0)]), (1, [(0, 1)])]))


def test_plan_outputs_rejects_an_out_of_range_slot():
    with pytest.raises(ValueError, match="invalid output_file_idx"):
        submission.plan_outputs(_metadata([(0, [(10, 0)])]))


@pytest.mark.parametrize("task", ["proton-ct", "proton-mri"])
def test_smoke_fixture_survives_the_body_mask(tmp_path, task):
    """A degenerate fixture makes the container smoke test vacuous.

    For MRI the body mask is ``image <= bounds[0]`` where ``bounds[0]`` is the
    0.5th percentile of positive intensities, so a constant-intensity phantom
    masks the whole volume and every slot comes back zero regardless of the
    model. Real validation MRI loses only 0.14-0.34% beyond background.
    """
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "create_submission_smoke_fixture.py"),
            "--output",
            str(tmp_path),
            "--task",
            task,
        ],
        check=True,
        capture_output=True,
    )
    modality = "ct" if task == "proton-ct" else "mri"
    path = next(
        (
            tmp_path
            / "images"
            / f"radiation-dose-calculation-source-{modality}-image-1"
        ).glob("*.mha")
    )
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    # Uncompressed data: a bad compressed header hands back uninitialised memory
    # instead of raising, which is how the fixture used to feed in 1e36 values.
    assert np.isfinite(array).all()
    assert np.abs(array).max() < 2000.0

    if modality == "ct":
        surviving = array > -1000.0
    else:
        low, _ = _mri_bounds(array)
        surviving = array > low
    assert surviving.sum() > array.size // 10, "body mask removes almost everything"


class _FluenceModel(torch.nn.Module):
    """Returns the fluence channel so predictions are exactly reproducible."""

    def forward(self, inputs):
        return inputs[:, 2:3]


def _write_input_image(path: Path, array: np.ndarray, spacing, origin) -> None:
    image = sitk.GetImageFromArray(np.ascontiguousarray(array, dtype=np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), useCompression=False)


_ENERGIES = (99.7909, 150.3508)
_CUTOFF = 0.02


def _end_to_end_metadata():
    """Two images, two slots, indices deliberately out of metadata order."""

    def beamlet(beamlet_idx, energy, out_idx, stack_idx):
        return {
            "beamlet_idx": beamlet_idx,
            "beamlet_uuid": f"beamlet-{out_idx}-{stack_idx}",
            "energy": energy,
            "output_info": {
                "output_file_idx": out_idx,
                "idx_in_output": stack_idx,
                "minimum_cutoff": _CUTOFF,
            },
        }

    return [
        {
            "image_file_idx": 0,
            "anatomical_region": "abdominal",
            "iso_center": [0.0, 0.0, 0.0],
            "beams": [
                {
                    "beam_idx": 0,
                    "gantry_angle": 0.0,
                    "rays": [
                        {
                            "ray_idx": 0,
                            "ray_source": [-500.0, 0.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            # Reverse order on purpose: placement must follow
                            # idx_in_output, never arrival order.
                            "beamlets": [
                                beamlet(0, _ENERGIES[0], 0, 2),
                                beamlet(1, _ENERGIES[1], 0, 0),
                                beamlet(2, _ENERGIES[0], 0, 1),
                            ],
                        },
                        {
                            "ray_idx": 1,
                            "ray_source": [-500.0, 6.0, 0.0],
                            "ray_target": [0.0, 6.0, 0.0],
                            "beamlets": [beamlet(3, _ENERGIES[1], 0, 3)],
                        },
                    ],
                }
            ],
        },
        {
            "image_file_idx": 2,
            "anatomical_region": "thoracic",
            "iso_center": [0.0, 0.0, 0.0],
            "beams": [
                {
                    "beam_idx": 0,
                    "gantry_angle": 90.0,
                    "rays": [
                        {
                            "ray_idx": 0,
                            "ray_source": [0.0, -500.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            "beamlets": [
                                beamlet(0, _ENERGIES[0], 4, 1),
                                beamlet(1, _ENERGIES[0], 4, 0),
                            ],
                        }
                    ],
                }
            ],
        },
    ]


@pytest.fixture()
def end_to_end_run(tmp_path, monkeypatch):
    """Provision /input, run the adapter on CPU and return the artefacts."""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    spacing = (2.0, 2.0, 2.0)
    origin = (-16.0, -16.0, -16.0)
    rng = np.random.default_rng(11)
    images = {
        0: rng.random((16, 16, 16), dtype=np.float32) * 100.0 - 500.0,
        2: rng.random((16, 16, 16), dtype=np.float32) * 100.0 - 500.0,
    }
    for image_idx, array in images.items():
        _write_input_image(
            input_root
            / "images"
            / f"radiation-dose-calculation-source-ct-image-{image_idx + 1}"
            / "image.mha",
            array,
            spacing,
            origin,
        )
    metadata = _end_to_end_metadata()
    (input_root / submission.METADATA_NAME).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(submission, "INPUT_PATH", input_root)
    monkeypatch.setattr(submission, "OUTPUT_PATH", output_root)
    # A chunk size below the per-image beamlet count forces the streaming path
    # to cross chunk boundaries, which is where buffering used to hide bugs.
    monkeypatch.setenv("BEAMLET_CHUNK_SIZE", "2")

    bundle = submission.ModelBundle(
        model=_FluenceModel(),
        device=torch.device("cpu"),
        patch_size_zyx=(16, 16, 16),
        dose_scale=2.0,
        overlap=0.0,
        condition_batch_size=2,
        amp=False,
        ray_gate_threshold=0.0,
        relative_cutoff=0.0,
        compiled=False,
        energy_table=submission._load_energy_table(),
    )
    submission.run(bundle)
    return output_root, metadata, images, spacing, origin


def _expected_frame(bundle_scale, image, spacing, origin, gantry, source, target, energy):
    sigma_energy, sigma_spot = _energy_parameters_for(energy)
    geometry = SpatialGeometry(spacing, origin, (1, 0, 0, 0, 1, 0, 0, 0, 1))
    condition = ProtonCondition(
        gantry_angle_deg=gantry,
        ray_source_xyz=tuple(float(v) for v in source),
        ray_target_xyz=tuple(float(v) for v in target),
        energy_mev=energy,
        sigma_energy_mev=sigma_energy,
        sigma_spot_mm=sigma_spot,
    )
    fluence = build_proton_channels(
        image, (0, 0, 0), geometry, condition, modality="ct"
    )[2]
    frame = (bundle_scale * fluence).astype(np.float32)
    frame[frame <= _CUTOFF] = 0.0
    return frame


def _energy_parameters_for(energy: float):
    table = submission._load_energy_table()
    row = min(table, key=lambda item: abs(item["energy_mev"] - energy))
    return row["sigma_energy_mev"], row["sigma_spot_mm"]


def test_end_to_end_writes_all_ten_slots(end_to_end_run):
    output_root, _, _, _, _ = end_to_end_run
    for slot in range(1, submission.NUM_OUTPUT_FILES + 1):
        directory = output_root / "images" / f"stacked-radiation-dose-map-{slot}"
        files = sorted(directory.glob("*.mha"))
        assert len(files) == 1, f"slot {slot} must hold exactly one .mha"


def test_end_to_end_stacks_are_scalar_4d_with_planned_frame_counts(end_to_end_run):
    output_root, _, _, spacing, origin = end_to_end_run
    expected_frames = {1: 4, 5: 2}
    for slot in range(1, submission.NUM_OUTPUT_FILES + 1):
        path = next(
            (output_root / "images" / f"stacked-radiation-dose-map-{slot}").glob("*.mha")
        )
        image = sitk.ReadImage(str(path))
        assert image.GetDimension() == 4
        assert image.GetNumberOfComponentsPerPixel() == 1
        if slot in expected_frames:
            assert image.GetSize() == (16, 16, 16, expected_frames[slot])
            assert image.GetSpacing() == pytest.approx(spacing + (1.0,))
            assert image.GetOrigin() == pytest.approx(origin + (0.0,))
        else:
            assert image.GetSize() == (1, 1, 1, 1)


def test_end_to_end_places_every_frame_at_its_declared_index(end_to_end_run):
    output_root, metadata, images, spacing, origin = end_to_end_run
    stacks = {}
    for slot in (1, 5):
        path = next(
            (output_root / "images" / f"stacked-radiation-dose-map-{slot}").glob("*.mha")
        )
        stacks[slot - 1] = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

    checked = 0
    for image_entry in metadata:
        image = images[int(image_entry["image_file_idx"])]
        for beam in image_entry["beams"]:
            for ray in beam["rays"]:
                for beamlet in ray["beamlets"]:
                    info = beamlet["output_info"]
                    expected = _expected_frame(
                        2.0,
                        image,
                        spacing,
                        origin,
                        float(beam["gantry_angle"]),
                        ray["ray_source"],
                        ray["ray_target"],
                        float(beamlet["energy"]),
                    )
                    actual = stacks[info["output_file_idx"]][info["idx_in_output"]]
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
                    checked += 1
    assert checked == 6


def test_end_to_end_respects_the_minimum_cutoff(end_to_end_run):
    output_root, _, _, _, _ = end_to_end_run
    for slot in range(1, submission.NUM_OUTPUT_FILES + 1):
        path = next(
            (output_root / "images" / f"stacked-radiation-dose-map-{slot}").glob("*.mha")
        )
        values = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        assert np.isfinite(values).all()
        assert (values >= 0.0).all()
        nonzero = values[values != 0.0]
        if nonzero.size:
            assert nonzero.min() > _CUTOFF

