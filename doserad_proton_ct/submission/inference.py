from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch

from doserad_photon_ct.conditioning import SpatialGeometry
from doserad_photon_ct.inference import warmup_model
from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D
from doserad_proton.conditioning import ProtonCondition
from doserad_proton.inference import predict_conditioned_arrays


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
RESOURCE_PATH = Path(__file__).with_name("beam_parameters.json")
METADATA_NAME = "stacked-proton-beam-level-metadata.json"
OUTPUT_DIRECTORY_BASE = "stacked-radiation-dose-map"
NUM_OUTPUT_FILES = 10

TASK = os.environ.get("TASK", "proton-ct")
TASK_CONFIG = {
    "proton-ct": ("ct", "radiation-dose-calculation-source-ct-image"),
    "proton-mri": ("mri", "radiation-dose-calculation-source-mri-image"),
}
if TASK not in TASK_CONFIG:
    raise ValueError(f"Unsupported TASK {TASK!r}")
MODALITY, INPUT_DIRECTORY_BASE = TASK_CONFIG[TASK]


@dataclass
class ModelBundle:
    model: torch.nn.Module
    device: torch.device
    patch_size_zyx: tuple[int, int, int]
    dose_scale: float
    overlap: float
    condition_batch_size: int
    amp: bool
    ray_gate_threshold: float
    relative_cutoff: float
    compiled: bool
    energy_table: list[dict[str, float]]
    # Defaults to the ten-channel contract so older checkpoints and existing
    # callers keep working; load_model() derives it from the weights.
    range_channels: bool = False


@dataclass(frozen=True)
class BeamletJob:
    beam_idx: int
    ray_idx: int
    beamlet_idx: int
    output_index: int
    stack_index: int
    minimum_cutoff: float
    condition: ProtonCondition


@dataclass(frozen=True)
class OutputPlan:
    """How many frames an output slot holds and which input grid defines it."""

    image_file_idx: int
    frame_count: int


_DIMSIZE_PROBE = b"DimSize = 1 1 1 1\n"
_DATA_MARKER = b"ElementDataFile = LOCAL\n"


def _metaimage_header(reference: sitk.Image, frame_count: int) -> bytes:
    """Return a 4D MetaImage header matching a ``sitk.JoinSeries`` stack.

    SimpleITK writes the probe header itself, so origin, spacing, direction and
    every float formatting decision are identical to what ``JoinSeries`` plus
    ``WriteImage`` would emit. Only ``DimSize`` is patched, because the probe is
    deliberately a single 1x1x1 voxel and therefore costs no real memory.
    """
    tiny = sitk.Image([1, 1, 1], sitk.sitkFloat32)
    tiny.SetSpacing(reference.GetSpacing())
    tiny.SetOrigin(reference.GetOrigin())
    tiny.SetDirection(reference.GetDirection())
    probe = sitk.JoinSeries([tiny])
    with tempfile.TemporaryDirectory() as scratch:
        probe_path = Path(scratch) / "probe.mha"
        sitk.WriteImage(probe, str(probe_path), useCompression=False)
        raw = probe_path.read_bytes()
    marker_at = raw.find(_DATA_MARKER)
    if marker_at < 0:
        raise ValueError("probe MetaImage is not an uncompressed LOCAL-data header")
    header = raw[: marker_at + len(_DATA_MARKER)]
    if _DIMSIZE_PROBE not in header:
        raise ValueError(f"unexpected probe MetaImage header: {header!r}")
    size_x, size_y, size_z = (int(value) for value in reference.GetSize())
    patched = f"DimSize = {size_x} {size_y} {size_z} {frame_count}\n".encode("ascii")
    return header.replace(_DIMSIZE_PROBE, patched, 1)


class StackWriter:
    """Stream frames of a 4D MetaImage to disk, holding one frame in memory.

    Buffering every dose map until the end of the run costs host memory
    proportional to the total number of maps: at 500 proton beamlets and the
    largest training grid that is above 80 GiB, while an A10G instance offers at
    most 31 GiB of usable DRAM. Frames are therefore written straight into their
    final byte offset, which also tolerates out-of-order arrival.
    """

    def __init__(self, path: Path, reference: sitk.Image, frame_count: int) -> None:
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        size_x, size_y, size_z = (int(value) for value in reference.GetSize())
        header = _metaimage_header(reference, frame_count)
        self.path = path
        self.frame_count = frame_count
        self._frame_shape_zyx = (size_z, size_y, size_x)
        self._frame_bytes = size_x * size_y * size_z * 4
        self._header_bytes = len(header)
        self._pending = set(range(frame_count))
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("wb+")
        self._handle.write(header)
        self._handle.truncate(self._header_bytes + frame_count * self._frame_bytes)

    def write(self, index: int, array: np.ndarray) -> None:
        if self._closed:
            raise ValueError(f"{self.path} is already closed")
        if index not in self._pending:
            if 0 <= index < self.frame_count:
                raise ValueError(f"frame {index} was already written to {self.path}")
            raise ValueError(f"frame {index} outside 0..{self.frame_count - 1}")
        # "<f4" pins little-endian, matching BinaryDataByteOrderMSB = False.
        frame = np.ascontiguousarray(array, dtype="<f4")
        if frame.shape != self._frame_shape_zyx:
            raise ValueError(
                f"frame shape {frame.shape} does not match the input grid "
                f"{self._frame_shape_zyx}"
            )
        self._handle.seek(self._header_bytes + index * self._frame_bytes)
        # Writing the buffer directly avoids a second copy of the frame.
        self._handle.write(frame.data)
        self._pending.discard(index)

    def close(self) -> None:
        """Flush and validate that no frame of the stack is missing."""
        if self._closed:
            return
        try:
            if self._pending:
                raise ValueError(
                    f"{self.path} is missing frames {sorted(self._pending)}"
                )
            # No fsync: the platform reads the file through the page cache,
            # which is coherent the moment write() returns. Syncing a stack
            # of hundreds of raw frames to disk blocked close() for ~30s per
            # 225-map job on the hidden set, all of it billed as runtime.
            self._handle.flush()
        finally:
            self._closed = True
            self._handle.close()

    def discard(self) -> None:
        """Release the file handle without validating, for failure paths."""
        if self._closed:
            return
        self._closed = True
        self._handle.close()


def _find_checkpoint() -> Path:
    explicit = os.environ.get("MODEL_CHECKPOINT")
    if explicit:
        path = MODEL_PATH / explicit
        if not path.is_file():
            raise FileNotFoundError(f"MODEL_CHECKPOINT does not exist: {path}")
        return path
    candidates = sorted(MODEL_PATH.rglob("best.pt")) or sorted(MODEL_PATH.rglob("*.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one .pt checkpoint under {MODEL_PATH}, found {len(candidates)}"
        )
    return candidates[0]


def _load_energy_table() -> list[dict[str, float]]:
    payload = json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))
    table = payload["proton"]["energy_table"]
    if not table:
        raise ValueError("Proton energy table is empty")
    return [
        {
            "energy_mev": float(row["energy_mev"]),
            "sigma_energy_mev": float(row["sigma_energy_mev"]),
            "sigma_spot_mm": float(row["sigma_spot_mm"]),
        }
        for row in table
    ]


def load_model() -> ModelBundle:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = _find_checkpoint()
    print(f"Loading {TASK} checkpoint {checkpoint_path} on {device}", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    eager_model = PhotonDoseUNet3D(ModelConfig(**checkpoint["model_config"]))
    eager_model.load_state_dict(checkpoint["model"])
    eager_model.to(device).eval()
    model: torch.nn.Module = eager_model
    compile_enabled = device.type == "cuda" and os.environ.get("TORCH_COMPILE", "1") != "0"
    if compile_enabled:
        model = torch.compile(eager_model, mode="reduce-overhead", fullgraph=True)
    training_config = checkpoint.get("training_config", {})
    default_patch = training_config.get("inference_patch_size") or (128, 128, 128)
    patch_size = tuple(
        int(value)
        for value in os.environ.get(
            "INFERENCE_PATCH_SIZE", ",".join(str(value) for value in default_patch)
        ).split(",")
    )
    if len(patch_size) != 3:
        raise ValueError("INFERENCE_PATCH_SIZE must contain Z,Y,X")
    training_gate = float(training_config.get("ray_gate_threshold", 0.0))
    bundle = ModelBundle(
        model=model,
        device=device,
        patch_size_zyx=patch_size,
        dose_scale=float(training_config["dose_scale"]),
        overlap=float(os.environ.get("INFERENCE_OVERLAP", "0.25")),
        condition_batch_size=int(os.environ.get("INFERENCE_BATCH_SIZE", "8")),
        amp=device.type == "cuda" and os.environ.get("DISABLE_AMP", "0") != "1",
        ray_gate_threshold=float(
            os.environ.get("RAY_GATE_THRESHOLD", str(max(training_gate, 1.0e-6)))
        ),
        relative_cutoff=float(os.environ.get("RELATIVE_CUTOFF", "0")),
        compiled=compile_enabled,
        energy_table=_load_energy_table(),
        # Twelve inputs mean the checkpoint was trained with the water-
        # equivalent depth and Bragg-prior channels, so the flag follows the
        # weights instead of a separate setting that could disagree with them.
        range_channels=int(checkpoint["model_config"]["in_channels"]) >= 12,
    )
    if os.environ.get("WARMUP_MODEL", "1") != "0":
        try:
            warmup_model(
                bundle.model,
                device=device,
                batch_size=bundle.condition_batch_size,
                in_channels=int(checkpoint["model_config"]["in_channels"]),
                patch_size_zyx=bundle.patch_size_zyx,
                amp=bundle.amp,
            )
        except Exception as error:
            if not bundle.compiled or os.environ.get("TORCH_COMPILE_FALLBACK", "1") == "0":
                raise
            print(f"Compiled warm-up failed; using eager model: {error}", flush=True)
            bundle.model = eager_model
            bundle.compiled = False
            if device.type == "cuda":
                torch.cuda.empty_cache()
            warmup_model(
                bundle.model,
                device=device,
                batch_size=bundle.condition_batch_size,
                in_channels=int(checkpoint["model_config"]["in_channels"]),
                patch_size_zyx=bundle.patch_size_zyx,
                amp=bundle.amp,
            )
    print(
        f"Ready modality={MODALITY} range_channels={bundle.range_channels} "
        f"patch={bundle.patch_size_zyx} "
        f"batch={bundle.condition_batch_size} gate={bundle.ray_gate_threshold:g}",
        flush=True,
    )
    return bundle


def _load_metadata() -> list[dict[str, Any]]:
    payload = json.loads((INPUT_PATH / METADATA_NAME).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("stacked proton metadata must be a list")
    return payload


def _find_input_image(image_file_idx: int) -> Path:
    directory = INPUT_PATH / "images" / f"{INPUT_DIRECTORY_BASE}-{image_file_idx + 1}"
    candidates = sorted(directory.glob("*.mha"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one MHA in {directory}, found {len(candidates)}")
    return candidates[0]


def _geometry_from_sitk(image: sitk.Image) -> SpatialGeometry:
    return SpatialGeometry(
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        origin_xyz=tuple(float(value) for value in image.GetOrigin()),
        direction=tuple(float(value) for value in image.GetDirection()),
    )


def _to_sitk(array_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(array_zyx, dtype=np.float32))
    image.CopyInformation(reference)
    return image


def _empty_output_stack() -> sitk.Image:
    frame = sitk.Image(1, 1, 1, sitk.sitkFloat32)
    return sitk.JoinSeries([frame])


def _energy_parameters(bundle: ModelBundle, energy_mev: float) -> tuple[float, float]:
    row = min(bundle.energy_table, key=lambda item: abs(item["energy_mev"] - energy_mev))
    if abs(row["energy_mev"] - energy_mev) > 0.02:
        raise ValueError(f"energy {energy_mev} MeV is absent from the machine table")
    return row["sigma_energy_mev"], row["sigma_spot_mm"]


def _collect_jobs(bundle: ModelBundle, image_entry: dict[str, Any]) -> list[BeamletJob]:
    jobs: list[BeamletJob] = []
    seen: set[tuple[int, int]] = set()
    for beam in image_entry["beams"]:
        beam_idx = int(beam.get("beam_idx", 0))
        gantry_angle = float(beam["gantry_angle"])
        for ray in beam["rays"]:
            ray_idx = int(ray.get("ray_idx", 0))
            source = tuple(float(value) for value in ray["ray_source"])
            target = tuple(float(value) for value in ray["ray_target"])
            for beamlet in ray["beamlets"]:
                output_info = beamlet["output_info"]
                output_index = int(output_info["output_file_idx"])
                stack_index = int(output_info["idx_in_output"])
                if not 0 <= output_index < NUM_OUTPUT_FILES:
                    raise ValueError(f"invalid output_file_idx {output_index}")
                position = (output_index, stack_index)
                if position in seen:
                    raise ValueError(f"duplicate output position {position}")
                seen.add(position)
                energy = float(beamlet["energy"])
                sigma_energy, sigma_spot = _energy_parameters(bundle, energy)
                jobs.append(
                    BeamletJob(
                        beam_idx=beam_idx,
                        ray_idx=ray_idx,
                        beamlet_idx=int(beamlet.get("beamlet_idx", 0)),
                        output_index=output_index,
                        stack_index=stack_index,
                        minimum_cutoff=float(output_info.get("minimum_cutoff", 0.0)),
                        condition=ProtonCondition(
                            gantry_angle_deg=gantry_angle,
                            ray_source_xyz=source,
                            ray_target_xyz=target,
                            energy_mev=energy,
                            sigma_energy_mev=sigma_energy,
                            sigma_spot_mm=sigma_spot,
                        ),
                    )
                )
    return jobs


def _output_directory(output_index: int) -> Path:
    return OUTPUT_PATH / "images" / f"{OUTPUT_DIRECTORY_BASE}-{output_index + 1}"


def plan_outputs(metadata: list[dict[str, Any]]) -> list[OutputPlan | None]:
    """Resolve every output slot from the metadata before any inference runs.

    Running this first means a malformed metadata file fails in milliseconds
    instead of after several minutes of GPU work, and it gives each stack its
    frame count up front, which the streaming writer needs.
    """
    owners: dict[int, int] = {}
    indices: dict[int, set[int]] = {}
    for image_entry in metadata:
        image_index = int(image_entry["image_file_idx"])
        for beam in image_entry["beams"]:
            for ray in beam["rays"]:
                for beamlet in ray["beamlets"]:
                    output_info = beamlet["output_info"]
                    output_index = int(output_info["output_file_idx"])
                    stack_index = int(output_info["idx_in_output"])
                    if not 0 <= output_index < NUM_OUTPUT_FILES:
                        raise ValueError(f"invalid output_file_idx {output_index}")
                    owner = owners.setdefault(output_index, image_index)
                    if owner != image_index:
                        raise ValueError(
                            f"output slot {output_index} mixes input images {owner} "
                            f"and {image_index}; one stack must share one dose grid"
                        )
                    slot = indices.setdefault(output_index, set())
                    if stack_index in slot:
                        raise ValueError(
                            f"duplicate output position ({output_index}, {stack_index})"
                        )
                    slot.add(stack_index)
    plans: list[OutputPlan | None] = [None] * NUM_OUTPUT_FILES
    for output_index, slot in indices.items():
        expected = set(range(len(slot)))
        if slot != expected:
            raise ValueError(
                f"output stack {output_index} has gaps: got {sorted(slot)}, "
                f"expected {sorted(expected)}"
            )
        plans[output_index] = OutputPlan(
            image_file_idx=owners[output_index], frame_count=len(slot)
        )
    return plans


def run(bundle: ModelBundle) -> None:
    started = time.perf_counter()
    metadata = _load_metadata()
    plans = plan_outputs(metadata)
    total_maps = 0
    chunk_size = int(os.environ.get("BEAMLET_CHUNK_SIZE", "32"))
    if chunk_size < 1:
        raise ValueError("BEAMLET_CHUNK_SIZE must be positive")
    writers: dict[int, StackWriter] = {}
    emptied_by_cutoff = 0
    peak_dose = 0.0

    try:
        for image_entry in metadata:
            image_started = time.perf_counter()
            image_index = int(image_entry["image_file_idx"])
            image_path = _find_input_image(image_index)
            reference = sitk.ReadImage(str(image_path))
            image = sitk.GetArrayFromImage(reference).astype(np.float32, copy=False)
            geometry = _geometry_from_sitk(reference)
            jobs = _collect_jobs(bundle, image_entry)
            window_cache: dict = {}
            wepl_cache: dict = {}
            for offset in range(0, len(jobs), chunk_size):
                chunk = jobs[offset : offset + chunk_size]
                predictions = predict_conditioned_arrays(
                    bundle.model,
                    image=image,
                    geometry=geometry,
                    conditions=[job.condition for job in chunk],
                    modality=MODALITY,
                    device=bundle.device,
                    patch_size_zyx=bundle.patch_size_zyx,
                    dose_scale=bundle.dose_scale,
                    overlap=bundle.overlap,
                    condition_batch_size=bundle.condition_batch_size,
                    amp=bundle.amp,
                    skip_empty_ray=os.environ.get("SKIP_EMPTY_RAY", "1") != "0",
                    mask_outside_body=os.environ.get("MASK_OUTSIDE_BODY", "1") != "0",
                    relative_cutoff=bundle.relative_cutoff,
                    ray_gate_threshold=bundle.ray_gate_threshold,
                    pad_to_batch_size=os.environ.get("PAD_INFERENCE_BATCH", "1") != "0",
                    roi_mode=os.environ.get("ROI_MODE", "off"),
                    roi_distal_margin_mm=float(
                        os.environ.get("ROI_DISTAL_MARGIN_MM", "80")
                    ),
                    window_cache=window_cache,
                    range_channels=bundle.range_channels,
                    wepl_cache=wepl_cache,
                )
                for job, prediction in zip(chunk, predictions):
                    # The pre-cutoff maximum is exact evidence of whether the
                    # cutoff can empty the map, and it costs one pass instead of
                    # a second scan after masking.
                    raw_max = float(prediction.max())
                    peak_dose = max(peak_dose, raw_max)
                    prediction[prediction <= job.minimum_cutoff] = 0.0
                    if raw_max <= job.minimum_cutoff:
                        emptied_by_cutoff += 1
                        if emptied_by_cutoff <= 3:
                            print(
                                f"WARNING: minimum_cutoff {job.minimum_cutoff:g} "
                                f"exceeds the whole map for output "
                                f"({job.output_index}, {job.stack_index}); "
                                f"pre-cutoff max was {raw_max:.6g}",
                                flush=True,
                            )
                    writer = writers.get(job.output_index)
                    if writer is None:
                        plan = plans[job.output_index]
                        if plan is None:
                            raise ValueError(
                                f"output slot {job.output_index} is absent from the plan"
                            )
                        writer = StackWriter(
                            _output_directory(job.output_index) / "output.mha",
                            reference,
                            plan.frame_count,
                        )
                        writers[job.output_index] = writer
                    writer.write(job.stack_index, prediction)
                # Predictions of this chunk are on disk; drop the host copies.
                del predictions
            total_maps += len(jobs)
            elapsed = time.perf_counter() - image_started
            print(
                f"Image {image_index}: maps={len(jobs)} seconds={elapsed:.3f}",
                flush=True,
            )

        for output_index in range(NUM_OUTPUT_FILES):
            directory = _output_directory(output_index)
            directory.mkdir(parents=True, exist_ok=True)
            writer = writers.get(output_index)
            if writer is None:
                if plans[output_index] is not None:
                    raise ValueError(
                        f"output slot {output_index} was planned but never written"
                    )
                sitk.WriteImage(_empty_output_stack(), str(directory / "output.mha"))
                continue
            writer.close()
            print(
                f"Wrote output {output_index + 1}: maps={writer.frame_count}",
                flush=True,
            )
    finally:
        for writer in writers.values():
            writer.discard()
    if emptied_by_cutoff:
        print(
            f"WARNING: {emptied_by_cutoff}/{total_maps} maps were fully zeroed by "
            f"their minimum_cutoff; highest pre-cutoff dose in this run was "
            f"{peak_dose:.6g}. Check that the declared cutoff and the predicted "
            f"dose share the same units.",
            flush=True,
        )
    print(
        f"Invoke complete: images={len(metadata)} maps={total_maps} "
        f"peak_dose={peak_dose:.6g} "
        f"seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )
