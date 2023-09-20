import numpy as np
import re
from dataclasses import dataclass
from typing import Tuple


@dataclass
class lut_1d_properties:
    size: int = 0
    contents: np.ndarray = np.zeros((size, 3))
    title: str = ""
    domain_min: np.ndarray = np.zeros(3)
    domain_max: np.ndarray = np.ones(3)


def lookup_1d_lut(x: np.ndarray, lut: lut_1d_properties) -> Tuple[float, float, float]:
    """
    Uses linear interpolation (if needed) and outputs the appropriate 3-channel value.
    """
    assert all(x >= lut.domain_min), f"out of range: {x}"
    assert all(x <= lut.domain_max), f"out of range: {x}"
    idx = (x - lut.domain_min) * (lut.domain_max - lut.domain_min) * (lut.size - 1)
    idx_floor = np.floor(idx).astype(int)
    idx_ceil = np.ceil(idx).astype(int)
    y_floor = lut.contents[idx_floor, [0, 1, 2]]
    y_ceil = lut.contents[idx_ceil, [0, 1, 2]]
    y_interp = (idx - idx_floor) * (y_ceil - y_floor) + y_floor
    return (y_interp[0], y_interp[1], y_interp[2])


def scalar_lookup_1d_lut(x: float, lut: lut_1d_properties, channel: int = 0) -> float:
    assert x >= lut.domain_min[channel], f"out of range: {x}"
    assert x <= lut.domain_max[channel], f"out of range: {x}"
    idx: float = (
        (x - lut.domain_min[channel])
        * (lut.domain_max[channel] - lut.domain_min[channel])
        * (lut.size - 1)
    )
    idx_floor = np.floor(idx).astype(int)
    idx_ceil = np.ceil(idx).astype(int)
    y_floor = lut.contents[idx_floor, channel]
    y_ceil = lut.contents[idx_ceil, channel]
    y_interp = (idx - idx_floor) * (y_ceil - y_floor) + y_floor
    return float(y_interp)


def read_1d_lut(fname: str) -> lut_1d_properties:
    title_pat = r'TITLE "(.+)"'
    lut_3d_size_pat = r"LUT_3D_SIZE (\d+)"
    lut_1d_size_pat = r"LUT_1D_SIZE (\d+)"
    domain_min_pat = r"DOMAIN_MIN ([\d\.]+ [\d\.]+ [\d\.]+)"
    domain_max_pat = r"DOMAIN_MAX ([\d\.]+ [\d\.]+ [\d\.]+)"
    line_pat = r"^([\d\.]+ [\d\.]+ [\d\.]+)"

    with open(fname, "r") as f:
        contents = []
        properties = lut_1d_properties()

        line = f.readline()
        while line != "":
            if match := re.match(line_pat, line):
                entries = match.group(1).split(" ")
                entries = [float(x) for x in entries]
                contents.append(entries)
            elif match := re.match(title_pat, line):
                properties.title = match.group(1)
            elif match := re.match(lut_1d_size_pat, line):
                properties.size = int(match.group(1))
            elif match := re.match(lut_3d_size_pat, line):
                assert False, "3D LUTs not supported!"
            elif match := re.match(domain_min_pat, line):
                mins = match.group(1).split(" ")
                mins = [float(x) for x in mins]
                properties.domain_min = np.array(mins)
            elif match := re.match(domain_max_pat, line):
                maxs = match.group(1).split(" ")
                maxs = [float(x) for x in maxs]
                properties.domain_max = np.array(maxs)
            line = f.readline()
        assert properties.size == len(
            contents
        ), "LUT size disagrees with content length!"
        properties.contents = np.array(contents)
        assert properties.size > 0, "LUT has zero size!"
    return properties


if __name__ == "__main__":
    fname = "zlog2_to_linear_4096.cube"
    lut = read_1d_lut(fname)
    print(lookup_1d_lut(np.array([0.3, 0.5, 0.3]), lut))
