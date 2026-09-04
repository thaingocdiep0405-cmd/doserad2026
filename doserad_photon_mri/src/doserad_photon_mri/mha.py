from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .dataset_index import read_mha_header


ELEMENT_DTYPES = {
    "MET_CHAR": "i1",
    "MET_UCHAR": "u1",
    "MET_SHORT": "<i2",
    "MET_USHORT": "<u2",
    "MET_INT": "<i4",
    "MET_UINT": "<u4",
    "MET_FLOAT": "<f4",
    "MET_DOUBLE": "<f8",
}


def load_mha_array(path: Path) -> NDArray[np.generic]:
    """Load a LOCAL MetaImage volume without requiring SimpleITK.

    This small reader is only for dataset inspection and statistics. Training
    and submission code should use SimpleITK so spatial metadata is preserved.
    """
    path = path.resolve()
    header = read_mha_header(path)
    if header["ElementDataFile"].upper() != "LOCAL":
        raise ValueError(f"only LOCAL MHA payloads are supported: {path}")
    element_type = header["ElementType"]
    if element_type not in ELEMENT_DTYPES:
        raise ValueError(f"unsupported ElementType {element_type}: {path}")

    raw = path.read_bytes()
    marker = b"ElementDataFile"
    marker_position = raw.find(marker)
    line_end = raw.find(b"\n", marker_position)
    if marker_position < 0 or line_end < 0:
        raise ValueError(f"cannot locate MHA payload: {path}")
    payload = raw[line_end + 1 :]
    if header.get("CompressedData", "False").lower() == "true":
        payload = zlib.decompress(payload)

    dimensions_xyz = tuple(int(value) for value in header["DimSize"].split())
    expected_values = int(np.prod(dimensions_xyz))
    array = np.frombuffer(payload, dtype=np.dtype(ELEMENT_DTYPES[element_type]))
    if array.size != expected_values:
        raise ValueError(
            f"voxel count mismatch for {path}: expected {expected_values}, got {array.size}"
        )
    return array.reshape(tuple(reversed(dimensions_xyz)))


def write_mha_array_like(
    output_path: Path,
    array_zyx: NDArray[np.floating],
    reference_path: Path,
    *,
    compress: bool = False,
) -> None:
    """Write a float32 MHA while preserving reference spatial metadata."""
    reference_header = read_mha_header(reference_path)
    array = np.asarray(array_zyx, dtype="<f4", order="C")
    expected_shape = tuple(
        reversed(tuple(int(value) for value in reference_header["DimSize"].split()))
    )
    if array.shape != expected_shape:
        raise ValueError(f"output shape {array.shape} does not match reference {expected_shape}")

    payload = array.tobytes(order="C")
    if compress:
        payload = zlib.compress(payload)
    lines = [
        "ObjectType = Image",
        "NDims = 3",
        "BinaryData = True",
        "BinaryDataByteOrderMSB = False",
        f"CompressedData = {'True' if compress else 'False'}",
    ]
    if compress:
        lines.append(f"CompressedDataSize = {len(payload)}")
    for key in ("TransformMatrix", "Offset", "CenterOfRotation", "AnatomicalOrientation"):
        if key in reference_header:
            lines.append(f"{key} = {reference_header[key]}")
    lines.extend(
        [
            f"ElementSpacing = {reference_header['ElementSpacing']}",
            f"DimSize = {reference_header['DimSize']}",
            "ElementType = MET_FLOAT",
            "ElementDataFile = LOCAL",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        handle.write(payload)
