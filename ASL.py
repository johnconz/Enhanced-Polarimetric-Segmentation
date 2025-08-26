#!/usr/bin/env python

# ----------------------------------------------------------------------->
# Authors: Connor Prikkel, Shaik Nordin Abouzahara, Bradley M. Ratliff
# Applied Sensing Lab & Vision Lab, University of Dayton
# 8/8/2025
#
# An object-oriented Python parser for ASL (.asl.hdr) files and associated raw data.
# ----------------------------------------------------------------------->

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class FrameMetadata:
    """Holds parsed metadata parameters for a single frame."""

    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASLHeader:
    """Represents the full ASL header, including required, global, and per-frame metadata."""

    asl_filename: str
    asl_header_filename: str
    required: Dict[str, Any] = field(default_factory=dict)
    global_metadata: Dict[str, Any] = field(default_factory=dict)
    frame_metadata: List[FrameMetadata] = field(default_factory=list)


class ASL:
    """Main class to read ASL headers and image data."""

    def __init__(self, path_to_asl_file: Path) -> None:
        """
        Initialize with the path to the ASL header file (Path object).
        The header is lazy-loaded; data is loaded on get_data().
        """
        self.path = path_to_asl_file
        self.header: Optional[ASLHeader] = None
        self.data = None

        assert isinstance(self.path, Path), "provided path is not a pathlib.Path object"

    def get_header(self) -> ASLHeader:
        """
        Read and parse the ASL header if not already done.
        Returns a fully parsed ASLHeader.
        """
        if self.header is None:
            raw_hdr = self._read_asl_header(self.path)
            self.header = self._parse_asl_header_values(raw_hdr)
        return self.header

    def get_data(
        self,
        channels: Optional[List[int]] = None,
        instances: Optional[List[int]] = None,
        frames: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, ASLHeader, ASLHeader]:
        """
        Read image data slices for specified channels, instances, and frames.
        Defaults to all available indices if None is provided.
        Returns:
          - data: NumPy array of shape (height, width, total_slices)
          - partial_header: ASLHeader trimmed to match the data read
          - full_header: The complete ASLHeader
        """
        hdr = self.get_header()
        # Default to all channels/instances/frames
        ch = channels or list(range(1, hdr.required["channels"] + 1))
        inst = instances or list(range(1, hdr.required["instances"] + 1))
        frs = frames or list(range(1, hdr.required["frames"] + 1))

        # Validate ranges
        self._validate_range("channels", ch, hdr.required["channels"])
        self._validate_range("instances", inst, hdr.required["instances"])
        self._validate_range("frames", frs, hdr.required["frames"])

        # Compute sizes
        width, height = hdr.required["width"], hdr.required["height"]
        dtype = np.dtype(hdr.required["data type"])
        byte_order = "<" if hdr.required["byte order"] == "Little Endian" else ">"
        dtype = dtype.newbyteorder(byte_order)

        nbpc = width * height * dtype.itemsize
        nbpi = nbpc * hdr.required["channels"]
        nbpf = nbpi * hdr.required["instances"]

        # Prepare output array: (height, width, total_slices)
        total = len(frs) * len(inst) * len(ch)
        data = np.empty((height, width, total), dtype=dtype)

        # Read raw file
        filename = Path(hdr.asl_filename)
        with filename.open("rb") as f:
            idx = 0
            for frame in frs:
                for instance in inst:
                    for channel in ch:
                        offset = (
                            (frame - 1) * nbpf
                            + (instance - 1) * nbpi
                            + (channel - 1) * nbpc
                        )
                        f.seek(offset, 0)
                        block = np.fromfile(f, count=width * height, dtype=dtype)
                        if block.size != width * height:
                            raise IOError("Unexpected EOF while reading data.")
                        data[:, :, idx] = block.reshape((height, width))
                        idx += 1

        # Build partial header
        partial = ASLHeader(
            asl_filename=hdr.asl_filename,
            asl_header_filename=hdr.asl_header_filename,
            required={**hdr.required},
            global_metadata={**hdr.global_metadata},
            frame_metadata=[hdr.frame_metadata[f - 1] for f in frs],
        )
        partial.required.update(channels=len(ch), instances=len(inst), frames=len(frs))

        return data, partial, hdr
    
    def rotate_aop(self, s1: np.ndarray, s2: np.ndarray, k: int) -> np.ndarray:
        """
        Rotate the AoP based on the Stokes parameters s1 and s2.
        hdr: header information containing rotation angles.
        k: index of the current frame.
        NOTE: This function is a conversion of the MATLAB function of the same name.
        """

        hdr = self.get_header()

        # Convert degrees to radians
        solaz = np.deg2rad(hdr.frame_metadata[k].solar_azimuth)
        solel = np.deg2rad(hdr.frame_metadata[k].solar_elevation)
        senaz = np.deg2rad(hdr.frame_metadata[0].sensor_azimuth)   # index 1 in MATLAB → 0 in Python
        senel = np.deg2rad(hdr.frame_metadata[0].sensor_elevation)

        # Solar pointing vector
        Usol = np.array([
            np.cos(solel) * np.cos(solaz),
            np.cos(solel) * np.sin(solaz),
            np.sin(solel)
        ])

        # Sensor pointing vector
        Usen = np.array([
            np.cos(senel) * np.cos(senaz),
            np.cos(senel) * np.sin(senaz),
            np.sin(senel)
        ])

        # Scattering plane normal
        cross_prod = np.cross(Usen, Usol)
        n = cross_prod / np.linalg.norm(cross_prod)

        # Angle between sensor and scattering plane normal
        alpha = np.arccos(np.dot(Usen, n))

        # Rotate s1 and s2, then compute AoP
        s1r = np.cos(2 * alpha) * s1 + np.sin(2 * alpha) * s2
        s2r = -np.sin(2 * alpha) * s1 + np.cos(2 * alpha) * s2
        eaop = 0.5 * np.arctan2(s2r, s1r)

        return eaop
    
    def relabel_mask_by_class_name_map(self, class_map, zero_out_unmatched=True):
        """
        Relabel mask with unified class IDs based on classMap.

        Parameters
        ----------
        classMap : dict
            Keys = substrings (case-insensitive), values = new class IDs
        zero_out_umatched: : bool, default=True
            Whether to set unmatched pixels to 0

        Returns
        -------
        relabeled_mask : np.ndarray
            Relabeled mask with unified class IDs
        valid_pixels : np.ndarray (bool)
            Boolean mask of pixels belonging to new classes (excluding background)
        """
        # Read header + data in
        hdr = self.get_header()
        masks, _, _ = self.get_data()

        # Initialize output arrays
        relabeled_masks = np.zeros_like(masks, dtype=np.uint8)
        valid_pixels = np.zeros_like(masks, dtype=bool)

        num_frames = masks.shape[2]

        # Loop through each turntable azimuth position's mask
        #scene_ids = hdr.frame_metadata.params.get("scene id")
        for turntable_pos in range(num_frames):
            masks_frame = masks[:, :, turntable_pos]
            relabeled_mask = np.zeros_like(masks_frame, dtype=np.uint8)
            matched_mask = np.zeros_like(masks_frame, dtype=bool)

            # To avoid type errors
            object_names = hdr.frame_metadata[turntable_pos].params.get("object class name")
            object_ids = hdr.frame_metadata[turntable_pos].params.get("object id number")

            # Loop through class names
            for name, class_id in zip(object_names, object_ids):
                name = str(name).lower()
                #print(f"Processing class name: {name}, ID: {class_id}")

                # Skip null or background classes
                if "null" in name or "background" in name:
                    continue

                # Check substrings in class map
                for key, new_id in class_map.items():
                    if key.lower() in name:

                        # Find indices where mask equals class_id
                        indices = np.where(masks_frame == class_id)
                        relabeled_mask[indices] = new_id
                        matched_mask[indices] = True
                        break  # first match wins

            if not zero_out_unmatched:
                # Preserve unmatched labels
                unmatched_indices = ~matched_mask
                relabeled_mask[unmatched_indices] = masks_frame[unmatched_indices]


            relabeled_masks[:, :, turntable_pos] = relabeled_mask
            valid_pixels[:, :, turntable_pos] = relabeled_mask != 0

        return relabeled_masks, valid_pixels
    
    # NOTE: Kinda obsolete, but kept for compatibility
    def get_frames_for_turntable_pos(self, turntable_pos: Union[int, str]) -> List[int]:
        """
        Return a list of 1-based frame indices where the 'turntable device azimuth' metadata
        matches the specified value (as int or string).
        """
        hdr = self.get_header()
        matching_frames = []

        for idx, frame in enumerate(hdr.frame_metadata):
            value = frame.params.get("turntable device azimuth")

            if value is None:
                continue

            valid_values = [-180, -135, -90, -45, 0, 45, 90, 135]
            if value not in valid_values:
                print(f"WARNING: Frame {idx + 1} has unexpected turntable azimuth value: {value}")

            # Match int/float/string values with loose equality
            if str(value) == str(turntable_pos):
                matching_frames.append(idx + 1)  # Convert 0-based index to 1-based

        return matching_frames
    
    def get_frames_by_azimuth(self):
        """
        Returns a dictionary mapping azimuth positions to a list of frame indices
        by reading per-frame metadata from the ASL file.
        Example: { -180: [0, 8, 16], -135: [1, 9, 17], ... }
        """
        hdr = self.get_header()

        valid_positions = [-180, -135, -90, -45, 0, 45, 90, 135]
        azimuth_frames = {pos: [] for pos in valid_positions}

        for idx, frame in enumerate(hdr.frame_metadata):
            azimuth = frame.params.get("turntable device azimuth")

            if azimuth is None:
                raise ValueError(f"Frame {idx + 1} missing 'turntable device azimuth' metadata")

            if azimuth not in valid_positions:
                print(f"WARNING: Frame {idx + 1} has unexpected turntable azimuth value: {azimuth}")

            if azimuth in azimuth_frames:
                azimuth_frames[azimuth].append(idx+1)

        return azimuth_frames

    def _validate_range(self, name: str, sel: List[int], maximum: int) -> None:
        """Ensure selected indices are within [1, maximum]."""
        if min(sel) < 1 or max(sel) > maximum:
            raise ValueError(f"{name.capitalize()} selection out of range")

    def _parse_asl_header_values(self, hdr: ASLHeader) -> ASLHeader:
        """
        Post-process raw header strings:
            - Check version compatibility
            - Normalize description
            - Validate & convert required fields
            - Parse global and frame metadata values
        """
        self._check_version(hdr)
        self._normalize_file_description(hdr)
        self._validate_and_parse_required(hdr)
        self._parse_section(hdr.global_metadata)
        for frame in hdr.frame_metadata:
            self._parse_section(frame.params)
        return hdr

    def _check_version(self, hdr: ASLHeader) -> None:
        """Warn if header version ≠ parser version 1.0.0."""
        version = hdr.required.get("asl_file_version")
        if version:
            major, minor, patch = map(int, version.split("."))
            if (major, minor, patch) != (1, 0, 0):
                print(
                    f"WARNING: Parser v1.0.0 older than file v{major}.{minor}.{patch}"
                )

    def _normalize_file_description(self, hdr: ASLHeader) -> None:
        """Strip braces from a single file_description string."""
        desc = hdr.required.get("file_description")
        if isinstance(desc, str):
            # strip leading/trailing quotes/braces
            hdr.required["file_description"] = desc.strip("{}")

    def _validate_and_parse_required(self, hdr: ASLHeader) -> None:
        """
        Ensure data_format == 'Image', map MATLAB types to NumPy dtypes,
        validate byte order, and convert numeric strings to ints/floats.
        """
        req = hdr.required
        fmt = req.get("data format")
        if fmt != "Image":
            raise ValueError(f"Unsupported data format: {fmt}")
        # Map allowed data type strings → numpy dtype names
        dtype_map = {
            "Unsigned char": "uint8",
            "Signed Short": "int16",
            "Int": "int32",
            "Float": "float32",
            "Double": "float64",
            "Unsigned Short": "uint16",
            "Unsigned Int": "uint32",
            "Signed Int64": "int64",
            "Unsigned Int64": "uint64",
        }
        dt = req.get("data type")
        if dt not in dtype_map:
            raise ValueError("Invalid data type")
        req["data type"] = dtype_map[dt]

        # Byte order check
        bo = req.get("byte order")
        if bo not in ("Little Endian", "Big Endian"):
            raise ValueError("Invalid byte order")

        # Parse numeric required fields
        for numeric_field in ("width", "height", "channels", "instances", "frames"):
            val = req.get(numeric_field)
            if val is None:
                raise ValueError(f"Missing required field {numeric_field}")
            num = self._to_number(val)
            if num is None:
                raise ValueError(f"Invalid {numeric_field} value")
            req[numeric_field] = num

    def _parse_section(self, section: Dict[str, Any]) -> None:
        """Apply _parse_value_field to every entry in a metadata dict."""
        for key, val in list(section.items()):
            section[key] = self._parse_value_field(val)

    def _parse_value_field(self, val: str) -> Union[float, int, str, List[Any]]:
        """
        Convert a raw header value string into:
          - NaN for empty
          - List for `{…}`
          - Numeric or mixed list for `[…]`
          - Scalar int/float or fallback string
        """
        val = val.strip()
        if not val:
            return float("nan")
        # String array in braces {…}
        if val.startswith("{") and val.endswith("}"):
            inner = val[1:-1]
            return [item.strip().strip("'\"") for item in inner.split(",")]
        # Numeric or mixed array in brackets […]
        if val.startswith("[") and val.endswith("]"):
            return self._parse_bracket_array(val[1:-1].strip())
        # Scalar number?
        num = self._to_number(val)
        return num if num is not None else val

    def _old_parse_bracket_array(self, text: str) -> List[Any]:
        """
        Detect 2D arrays via nested brackets or parse a flat list of values.
        """
        if text.startswith("[") and text.endswith("]"):
            # rows = re.findall(r"$$([^$$]+)\]", text)
            # rows = re.findall(r"$$([^]]+)$$", text)
            rows = re.findall(r"\[([^\[\]]+)\]", text)
            return [self._parse_numeric_list(row) for row in rows]
        else:
            return self._parse_numeric_list(text)

    def _parse_bracket_array(self, text: str) -> List[Any]:
        """
        Detect 2D arrays via nested brackets or parse a flat list of values.
        Uses ast.literal_eval for safe and fast parsing of bracketed content.
        """
        if text.startswith("[") and text.endswith("]"):
            try:
                data = ast.literal_eval(text)
                if isinstance(data, list):
                    if all(isinstance(row, list) for row in data):
                        # 2D array
                        return [self._parse_numeric_list(row) for row in data]
                    else:
                        # 1D array
                        return self._parse_numeric_list(data)
                else:
                    raise ValueError("Input is not a list.")
            except Exception as e:
                raise ValueError(f"Invalid bracketed array input: {e}")
        else:
            return self._parse_numeric_list(text)

    def _parse_numeric_list(self, text: Union[str, List[Any]]) -> List[Any]:
        """
        Convert comma-separated string or list to int/float/str elements.
        """
        if isinstance(text, str):
            items = [item.strip() for item in text.split(",")]
        else:
            items = text  # assume already list-like

        parsed = []
        for item in items:
            if isinstance(item, (int, float)):
                parsed.append(item)
            else:
                num = self._to_number(str(item))
                parsed.append(num if num is not None else str(item).strip("'\""))
        return parsed

    def _to_number(self, s: str) -> Optional[Union[int, float]]:
        """
        Attempt to convert a string to int or float. Returns None on failure.
        Recognizes decimal points and scientific notation.
        """
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except ValueError:
            return None

    def _read_asl_header(self, hdrfile: Path) -> ASLHeader:
        """
        Open and read the header file:
          - Ensure .hdr extension
          - Confirm existence and first line "ASLFILE"
          - Partition into required, global, and frame params
          - Return a raw ASLHeader with string values
        """
        hdr_path = self._ensure_extension(hdrfile)
        self._assert_exists(hdr_path)

        raw_lines = hdr_path.read_text().splitlines()
        self._assert_first_line(raw_lines)

        section_entries = self._partition_sections(raw_lines[1:])
        required, global_md, frame_md = self._collect_params(section_entries)

        return ASLHeader(
            asl_filename=str(hdr_path.with_suffix("")),
            asl_header_filename=str(hdr_path),
            required=required,
            global_metadata=global_md,
            frame_metadata=frame_md,
        )

    # Helpers

    def _ensure_extension(self, path: Path) -> Path:
        """Append .hdr if missing, else return path unchanged."""
        return path.with_suffix(".asl.hdr") if path.suffix.lower() != ".hdr" else path

    def _assert_exists(self, path: Path) -> None:
        """Raise FileNotFoundError if header path does not exist."""
        if not path.exists():
            raise FileNotFoundError(f"Header file not found: {path}")

    def _assert_first_line(self, lines: List[str]) -> None:
        """Ensure the first non-empty line is 'ASLFILE'."""
        if not lines or lines[0].strip() != "ASLFILE":
            raise ValueError("Invalid ASL header: missing ASLFILE")

    def _partition_sections(self, lines: List[str]) -> List[Tuple[str, int, str]]:
        section_lines = []
        current_section = None
        frame_index = -1

        for line in lines:
            text = line.strip()
            if not text:
                continue
            if text == "-->>Required File Parameters<<--":
                current_section = "required"
                continue
            if text == "-->>Global Parameters<<--":
                current_section = "global_metadata"
                continue
            if text == "-->>Frame Parameters<<--":
                current_section = "frame_metadata"
                frame_index = -1
                continue
            if text.startswith("<<Frame"):
                frame_index = int(text.split()[1].strip(">>"))
                continue
            section_lines.append((current_section, frame_index, text))

        return section_lines

    def _collect_params(
        self, section_entries: List[Tuple[str, int, str]]
    ) -> Tuple[Dict[str, str], Dict[str, str], List[FrameMetadata]]:
        required: Dict[str, str] = {}
        global_md: Dict[str, str] = {}
        frames: List[FrameMetadata] = []

        for section, frame_idx, text in section_entries:
            if "=" not in text:
                continue
            key, val = [p.strip() for p in text.split("=", 1)]
            val = val.replace("'", '"')

            if section == "required":
                required[key] = val
            elif section == "global_metadata":
                global_md[key] = val
            elif section == "frame_metadata":
                while len(frames) <= frame_idx:
                    frames.append(FrameMetadata())
                frames[frame_idx].params[key] = val

        return required, global_md, frames
