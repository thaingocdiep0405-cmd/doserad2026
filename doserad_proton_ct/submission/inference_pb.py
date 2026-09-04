"""Pencil-beam dose backend for the proton submissions.

On validation beamlets the pyRadPlan pencil-beam engine is far closer to the
Monte-Carlo reference than the trained network — masked MAE 0.028 against
0.052 and IDD 0.0096 against 0.130 — and it runs about five times faster,
which matters because runtime carries double weight in the ranking.

The engine needs mass density. Proton-CT submissions hand it the CT directly.
Proton-MRI submissions have no density at all: bone and lung both read dark on
MRI while their densities sit at 1.8 and 0.3, so no intensity threshold can
separate them (measured: water-only bodies give MAE 0.013-0.033 on abdominal
patients but 0.18-0.29 on thoracic ones). A synthetic-CT network supplies the
density instead, and falls back to a water body if no such checkpoint ships
with the algorithm.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch

from doserad_proton.pencilbeam import BeamletSpec, PencilBeamEngine

import inference as base

HU_MIN, HU_MAX = -1024.0, 2000.0
SCT_PATCH = (96, 96, 96)


@dataclass
class PencilBeamBundle:
    engine: PencilBeamEngine
    synthetic_ct: torch.nn.Module | None
    device: torch.device
    mri_percentile: float
    # The trainer writes the window it encoded HU into; reading it back beats
    # trusting that the constant here still matches the one it used.
    hu_range: tuple[float, float] = (HU_MIN, HU_MAX)


def _find_synthetic_ct() -> Path | None:
    explicit = os.environ.get("SYNTHETIC_CT_CHECKPOINT")
    if explicit:
        path = base.MODEL_PATH / explicit
        return path if path.is_file() else None
    candidates = sorted(base.MODEL_PATH.rglob("sct*.pt"))
    return candidates[0] if candidates else None


def load_model() -> PencilBeamBundle:
    from doserad_photon_ct.model import ModelConfig, PhotonDoseUNet3D

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = PencilBeamEngine(base.RESOURCE_PATH.with_name("beam_parameters.json"))
    model = None
    hu_range = (HU_MIN, HU_MAX)
    checkpoint_path = _find_synthetic_ct()
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stored = checkpoint.get("hu_range")
        if stored is not None:
            hu_range = (float(stored[0]), float(stored[1]))
        model = PhotonDoseUNet3D(ModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        print(f"Loaded synthetic-CT checkpoint {checkpoint_path.name}", flush=True)
    elif base.MODALITY == "mri" and os.environ.get("ALLOW_WATER_BODY", "0") != "1":
        # Falling back to water on MRI is not a degraded run, it is a worse run
        # than the network this backend replaces: measured over 45 validation
        # beamlets, water gives IDD 0.1417 against 0.0895 with a synthetic CT.
        # Failing here costs a container start; staying quiet costs a
        # submission, and this phase allows only a handful of them.
        raise RuntimeError(
            f"pencil-beam on MRI needs a synthetic-CT checkpoint (sct*.pt under "
            f"{base.MODEL_PATH}) and none was found. Set ALLOW_WATER_BODY=1 to "
            f"run against a water body on purpose."
        )
    else:
        print("No synthetic-CT checkpoint found; using a water body", flush=True)
    print(
        f"Ready engine=pencil-beam modality={base.MODALITY} "
        f"synthetic_ct={model is not None} device={device}",
        flush=True,
    )
    return PencilBeamBundle(
        engine=engine,
        synthetic_ct=model,
        device=device,
        mri_percentile=float(os.environ.get("MRI_SCALE_PERCENTILE", "99")),
        hu_range=hu_range,
    )


@torch.inference_mode()
def _synthesize_ct(
    bundle: PencilBeamBundle, image: np.ndarray
) -> np.ndarray:
    """Run the synthetic-CT network over the volume in overlapping windows."""
    from doserad_photon_ct.inference import _accumulate_patch, gaussian_blend_weight, sliding_window_starts
    from doserad_photon_ct.dataset import crop_with_padding

    positive = image[image > 0]
    scale = float(np.percentile(positive, bundle.mri_percentile)) if positive.size else 1.0
    normalized = image / max(scale, 1e-6)
    output = np.zeros(image.shape, dtype=np.float32)
    weights = np.zeros(image.shape, dtype=np.float32)
    blend = gaussian_blend_weight(SCT_PATCH)
    starts = [
        (z, y, x)
        for z in sliding_window_starts(image.shape[0], SCT_PATCH[0], 0.25)
        for y in sliding_window_starts(image.shape[1], SCT_PATCH[1], 0.25)
        for x in sliding_window_starts(image.shape[2], SCT_PATCH[2], 0.25)
    ]
    batch_size = int(os.environ.get("SCT_BATCH_SIZE", "4"))
    for offset in range(0, len(starts), batch_size):
        chunk = starts[offset : offset + batch_size]
        patches = np.stack(
            [crop_with_padding(normalized, start, SCT_PATCH, pad_value=0.0) for start in chunk]
        )[:, None]
        tensor = torch.from_numpy(patches).to(bundle.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=bundle.device.type == "cuda"):
            predicted = bundle.synthetic_ct(tensor)
            if isinstance(predicted, list):
                predicted = predicted[0]
        values = predicted.float().cpu().numpy()[:, 0]
        for value, start in zip(values, chunk):
            _accumulate_patch(output, weights, value, blend, start)
    normalized_output = output / np.maximum(weights, 1.0e-8)
    # Undo the [0, 1] encoding the synthetic-CT trainer uses, which exists
    # because the shared network ends in a softplus.
    low, high = bundle.hu_range
    return (normalized_output * (high - low) + low).astype(np.float32)


def _density_volume(bundle: PencilBeamBundle, image: np.ndarray) -> np.ndarray:
    if base.MODALITY == "ct":
        return np.asarray(image, dtype=np.float32)
    if bundle.synthetic_ct is not None:
        return _synthesize_ct(bundle, image)
    from doserad_proton.data import _mri_bounds

    return bundle.engine.pseudo_ct_from_mri(image, _mri_bounds(image)[0])


# pyRadPlan is single-threaded CPU work and each beamlet is independent of every
# other, so beamlets are farmed out to worker processes. The workers are forked
# after the density volume exists, which lets them read it copy-on-write instead
# of shipping a hundred-odd megabytes to each one, and they hand back only the
# z-slab the dose lives in rather than a full map.
_WORKER_ENGINE: PencilBeamEngine | None = None
_WORKER_HU: np.ndarray | None = None
_WORKER_GEOM: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None = None


def _worker_init(engine: PencilBeamEngine, hu: np.ndarray, geom) -> None:
    global _WORKER_ENGINE, _WORKER_HU, _WORKER_GEOM
    _WORKER_ENGINE, _WORKER_HU, _WORKER_GEOM = engine, hu, geom
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass


def _crop_slab(z0: int, slab: np.ndarray):
    """Crop a slab to the bounding box of its nonzero dose.

    Chunked workers hand back a dozen slabs per task; shipping the full
    transverse plane for each would push hundreds of megabytes through the
    result pipe while the dose itself lives inside the lateral cutoff.
    """
    nonzero = np.nonzero(slab)
    if nonzero[0].size == 0:
        return z0, 0, 0, np.zeros((0, 0, 0), dtype=np.float32), slab.shape
    lows = [int(axis.min()) for axis in nonzero]
    highs = [int(axis.max()) + 1 for axis in nonzero]
    cropped = np.ascontiguousarray(
        slab[lows[0] : highs[0], lows[1] : highs[1], lows[2] : highs[2]]
    )
    return z0 + lows[0], lows[1], lows[2], cropped, slab.shape


def _worker_dose_chunk(item):
    indices, specs = item
    origin, spacing, direction = _WORKER_GEOM
    slabs = _WORKER_ENGINE.dose_slab_batch(_WORKER_HU, origin, spacing, direction, specs)
    return [
        (index, _crop_slab(z0, slab)) for index, (z0, slab) in zip(indices, slabs)
    ]


def _worker_count() -> int:
    override = os.environ.get("PB_WORKERS")
    if override:
        return max(1, int(override))
    return max(1, min(12, os.cpu_count() or 1))


def _chunks(specs, workers):
    """Split beamlets into per-slab chunks sized for the worker pool.

    Beamlets sharing an isocenter slice share their slab CT, body
    segmentation and engine setup, so they batch into one influence call.
    Groups are split further so every worker stays busy: a 225-beamlet plan
    typically holds ~5 distinct isocenter slices, and 8 workers chewing 5
    monolithic groups would leave three of them idle.
    """
    groups: dict[int, list[int]] = {}
    for index, spec in enumerate(specs):
        groups.setdefault(int(round(spec.ray_target_xyz[2])), []).append(index)
    target = max(4, min(16, (len(specs) + 2 * workers - 1) // (2 * workers)))
    chunks = []
    for indices in groups.values():
        for start in range(0, len(indices), target):
            part = indices[start : start + target]
            chunks.append((part, [specs[i] for i in part]))
    return chunks


def _dose_all(bundle, hu, geom, specs, workers):
    """Yield (index, cropped-slab payload) for every beamlet, unordered."""
    if workers <= 1 or len(specs) <= 1:
        for index, spec in enumerate(specs):
            z0, slab = bundle.engine.dose_slab(hu, *geom, spec)
            yield index, _crop_slab(z0, slab)
        return
    context = mp.get_context("fork")
    with context.Pool(
        processes=min(workers, len(specs)),
        initializer=_worker_init,
        initargs=(bundle.engine, hu, geom),
    ) as pool:
        for results in pool.imap_unordered(
            _worker_dose_chunk, _chunks(specs, workers), chunksize=1
        ):
            yield from results


def run(bundle: PencilBeamBundle) -> None:
    started = time.perf_counter()
    workers = _worker_count()
    print(f"Pencil-beam workers: {workers} cpu_count={os.cpu_count()}", flush=True)
    metadata = base._load_metadata()
    plans = base.plan_outputs(metadata)
    writers: dict[int, base.StackWriter] = {}
    total_maps = 0
    emptied_by_cutoff = 0
    peak_dose = 0.0

    try:
        for image_entry in metadata:
            image_started = time.perf_counter()
            image_index = int(image_entry["image_file_idx"])
            image_path = base._find_input_image(image_index)
            reference = sitk.ReadImage(str(image_path))
            image = sitk.GetArrayFromImage(reference).astype(np.float32, copy=False)
            hu = _density_volume(bundle, image)
            jobs = base._collect_jobs(bundle_stub(bundle), image_entry)
            specs = [
                BeamletSpec(
                    gantry_angle_deg=job.condition.gantry_angle_deg,
                    ray_source_xyz=job.condition.ray_source_xyz,
                    ray_target_xyz=job.condition.ray_target_xyz,
                    energy_mev=job.condition.energy_mev,
                )
                for job in jobs
            ]
            geom = (reference.GetOrigin(), reference.GetSpacing(), reference.GetDirection())
            shape = tuple(int(v) for v in hu.shape)
            for index, payload in _dose_all(bundle, hu, geom, specs, workers):
                job = jobs[index]
                z0, y0, x0, cropped, _ = payload
                prediction = np.zeros(shape, dtype=np.float32)
                prediction[
                    z0 : z0 + cropped.shape[0],
                    y0 : y0 + cropped.shape[1],
                    x0 : x0 + cropped.shape[2],
                ] = cropped
                raw_max = float(prediction.max())
                peak_dose = max(peak_dose, raw_max)
                prediction[prediction <= job.minimum_cutoff] = 0.0
                if raw_max <= job.minimum_cutoff:
                    emptied_by_cutoff += 1
                writer = writers.get(job.output_index)
                if writer is None:
                    plan = plans[job.output_index]
                    if plan is None:
                        raise ValueError(f"output slot {job.output_index} is absent from the plan")
                    writer = base.StackWriter(
                        base._output_directory(job.output_index) / "output.mha",
                        reference,
                        plan.frame_count,
                    )
                    writers[job.output_index] = writer
                writer.write(job.stack_index, prediction)
            total_maps += len(jobs)
            print(
                f"Image {image_index}: maps={len(jobs)} "
                f"seconds={time.perf_counter() - image_started:.3f}",
                flush=True,
            )

        for output_index in range(base.NUM_OUTPUT_FILES):
            directory = base._output_directory(output_index)
            directory.mkdir(parents=True, exist_ok=True)
            writer = writers.get(output_index)
            if writer is None:
                if plans[output_index] is not None:
                    raise ValueError(f"output slot {output_index} was planned but never written")
                sitk.WriteImage(base._empty_output_stack(), str(directory / "output.mha"))
                continue
            writer.close()
            print(f"Wrote output {output_index + 1}: maps={writer.frame_count}", flush=True)
    finally:
        for writer in writers.values():
            writer.discard()
    if emptied_by_cutoff:
        print(
            f"WARNING: {emptied_by_cutoff}/{total_maps} maps were fully zeroed by their "
            f"minimum_cutoff; highest pre-cutoff dose was {peak_dose:.6g}",
            flush=True,
        )
    print(
        f"Invoke complete: images={len(metadata)} maps={total_maps} "
        f"peak_dose={peak_dose:.6g} seconds={time.perf_counter() - started:.3f}",
        flush=True,
    )


class _Stub:
    """Minimal stand-in exposing the energy table _collect_jobs reads."""

    def __init__(self, energy_table: list[dict[str, float]]) -> None:
        self.energy_table = energy_table


def bundle_stub(bundle: PencilBeamBundle) -> Any:
    return _Stub(base._load_energy_table())
