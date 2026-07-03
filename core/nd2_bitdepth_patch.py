"""
core/nd2_bitdepth_patch.py

``nd2reader==3.2.1`` (pinned in requirements.txt) hardcodes 16-bit
("H" / unsigned-short) decoding of raw pixel bytes in
``Parser._get_raw_image_data``. For an .nd2 file whose pixel container is
actually 8-bit (``uiBpcInMemory == 8`` in the file's own SLxImageAttributes
chunk — common for some single-molecule/SPT acquisitions saved at reduced
bit depth), this silently unpacks half as many pixels as exist, which then
makes ``number_of_true_channels`` compute to 0 and crashes with
``ValueError: slice step cannot be zero`` the moment any frame is read
(via ``pims.open(...)[i]`` or ``ND2Reader`` directly, as used by both
``core/settings.py`` and the bundled ``fastQuot/quot/read.py``).

This module monkeypatches that one method to pick the correct array
typecode from the file's own metadata, falling back to the original
16-bit behavior whenever that metadata is unavailable (so ordinary
16-bit .nd2 files are completely unaffected).
"""

from __future__ import annotations

import array
import struct
import warnings

import numpy as np

_PATCHED = False

# uiBpcInMemory -> (array typecode, bytes per pixel)
_TYPECODE_BY_BPC = {8: ("B", 1), 16: ("H", 2), 32: ("I", 4)}


def apply() -> None:
    """Monkeypatch nd2reader's Parser to decode pixels at their real bit depth.

    Safe to call multiple times (no-ops after the first call) and safe to
    call when nd2reader isn't installed (silently does nothing).
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from nd2reader.parser import Parser
        from nd2reader.common import read_chunk
        from pims.base_frames import Frame
    except ImportError:
        return

    def _get_raw_image_data(self, image_group_number, channel_offset, height, width):
        chunk = self._label_map.get_image_data_location(image_group_number)
        data = read_chunk(self._fh, chunk)

        timestamp = struct.unpack("d", data[:8])[0]

        typecode, itemsize = "H", 2
        try:
            attrs = self._raw_metadata.image_attributes[b"SLxImageAttributes"]
            bpc_in_memory = attrs.get(b"uiBpcInMemory", 16)
            typecode, itemsize = _TYPECODE_BY_BPC.get(bpc_in_memory, ("H", 2))
        except (KeyError, AttributeError, TypeError):
            pass  # metadata missing/unexpected shape — fall back to original "H" behavior

        image_group_data = array.array(typecode, data)

        # The original code skips a fixed 4 "H" (2-byte) units to get past
        # the 8-byte timestamp header. That skip is 8 bytes regardless of
        # pixel bit depth, so express it in units of the *chosen* typecode
        # rather than hardcoding 4 (which is only correct for 2-byte units).
        header_skip = 8 // itemsize
        image_data_start = header_skip + channel_offset

        number_of_true_channels = int(len(image_group_data[header_skip:]) / (height * width))
        try:
            image_data = np.reshape(
                image_group_data[image_data_start::number_of_true_channels],
                (height, width),
            )
        except ValueError:
            image_data = np.reshape(
                image_group_data[image_data_start::number_of_true_channels],
                (
                    height,
                    int(
                        round(
                            len(image_group_data[image_data_start::number_of_true_channels])
                            / height
                        )
                    ),
                ),
            )

        # Skip images that are all zeros! NIS Elements creates blank "gap"
        # images if you don't have the same number of images each cycle.
        if np.any(image_data):
            return timestamp, Frame(image_data, metadata=self._get_frame_metadata())
        else:
            empty_frame = np.full((height, width), np.nan)
            warnings.warn(
                "ND2 file contains gap frames which are represented by "
                "np.nan-filled arrays; to convert to zeros use e.g. np.nan_to_num(array)"
            )
            return timestamp, Frame(empty_frame, metadata=self._get_frame_metadata())

    Parser._get_raw_image_data = _get_raw_image_data
    _PATCHED = True
