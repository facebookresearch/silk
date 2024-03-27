# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import math
import os
import pickle as pkl
import re
import shutil
import sqlite3
import struct
import zlib
from glob import glob
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
from diskcache import Cache
from pytorchvideo.data.utils import optional_threaded_foreach, thwc_to_cthw
from pytorchvideo.data.video import Video

from silk.logger import LOG


class _CacheBase:
    def __contains__(self, key: bytes) -> bool:
        raise NotImplementedError

    def __getitem__(self, key: bytes) -> bytes:
        raise NotImplementedError

    def __setitem__(self, key: bytes, value: bytes) -> None:
        raise NotImplementedError


class _AlwaysEmptyCache(_CacheBase):
    """Fake cache that is always empty and never actually caches anything."""

    def __init__(self) -> None:
        super().__init__()

    def __contains__(self, key: bytes) -> bool:
        return False

    def __getitem__(self, key: bytes) -> bytes:
        raise AssertionError(
            "`__getitem__` should not have been called since the cache is always empty"
        )

    def __setitem__(self, key: bytes, value: bytes) -> None:
        pass


class _DiskCache(_CacheBase):
    """Cache wrapper to handle potention errors."""

    FAIL_RETRIES = 3
    VERSION = 1
    VERSION_BYTES = int.to_bytes(1, 2, "little")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._args = args
        self._kwargs = kwargs
        self._cache = Cache(*args, **kwargs)

        # check version
        if ("version" not in self) or (self["version"] != _DiskCache.VERSION_BYTES):
            self._reopen_cache(delete_cache=True)
            self["version"] = _DiskCache.VERSION_BYTES

    def _reopen_cache(self, delete_cache=False):
        self._cache.close()
        if delete_cache:
            LOG.warning(f'cache "{self._cache.directory}" has been removed')
            shutil.rmtree(self._cache.directory)
        self._cache = Cache(*self._args, **self._kwargs)

    def _try(
        self,
        fn,
        ignore_exception=False,
        default_return=None,
    ):
        result = default_return
        for attempt in range(_DiskCache.FAIL_RETRIES + 1):
            try:
                result = fn()
                break
            except sqlite3.DatabaseError:
                if attempt < _DiskCache.FAIL_RETRIES:
                    LOG.warning(
                        f"disk cache error occured (attempt #{attempt}/{_DiskCache.FAIL_RETRIES}), attempting to re-open cache ..."
                    )

                    # try re-opening the cache
                    self._reopen_cache()
                else:
                    LOG.opt(exception=True).error(
                        f"disk cache error occured (attempt #{attempt}/{_DiskCache.FAIL_RETRIES})"
                    )
                    raise
            except BaseException:
                LOG.opt(exception=True).error("unhandled disk cache error occured")
                if ignore_exception:
                    break
                else:
                    raise
        return result

    def __contains__(self, key: bytes) -> bool:
        def contains():
            return key in self._cache

        return self._try(contains, ignore_exception=True, default_return=False)

    def __getitem__(self, key: bytes) -> bytes:
        def getitem():
            return self._cache[key]

        return self._try(getitem, ignore_exception=False, default_return=None)

    def __setitem__(self, key: bytes, value: bytes) -> None:
        def setitem():
            self._cache[key] = value

        return self._try(setitem, ignore_exception=True, default_return=None)


CacheType = Union[Cache, _AlwaysEmptyCache]


class Space:
    """A Space is a collection of Scans"""

    def __init__(self, id: int) -> None:
        self._id = id
        self._scans = []

    @property
    def id(self):
        return self._id

    def _add(self, scan: Scan):
        scan._parent = self
        self._scans.append(scan)

    def _remove(self, scan: Scan):
        scan._parent = None
        self._scans.remove(scan)

    def __getitem__(self, idx: int):
        return self._scans[idx]

    def __len__(self):
        return len(self._scans)


class Scan:
    """Scan contains information related to a particular scan (frames, mesh, metadata, ...)."""

    # regex used to extract ids
    SPACE_ID_SCAN_ID_RE = re.compile(
        r"scene(?P<spaceId>[0-9]{4})_(?P<sceneId>[0-9]{2})"
    )

    @staticmethod
    def space_id_scan_id_from_path(path: str) -> Tuple[int, int]:
        """Return space id and scan id from directory path of scan."""
        basename = os.path.basename(path)
        m = Scan.SPACE_ID_SCAN_ID_RE.match(basename)

        # no match
        if m is None:
            raise RuntimeError(
                f'cannot extract space id and scan id from basename "{basename}"'
            )

        spaceId = int(m.group("spaceId"))
        sceneId = int(m.group("sceneId"))

        return spaceId, sceneId

    @staticmethod
    def scan_file_path(path: str, extension: str, mesh=None) -> str:
        """Get full path of file present in scan directory by providing its extension."""
        if not mesh:
            relative_path = os.path.basename(path)
        else:
            relative_path = f"{os.path.basename(path)}_vh_{mesh}"

        return os.path.join(path, f"{relative_path}.{extension}")

    @staticmethod
    def load_from_directory(
        path: str,
        cache: CacheType,
    ) -> Union[Scan, None]:
        """Load a Scan object by providing its directory path."""
        # get ids
        try:
            _, scan_id = Scan.space_id_scan_id_from_path(path)
        except RuntimeError:
            LOG.opt(exception=True).warning(f"could not extract ids from path : {path}")
            return None

        # load sens file
        sens_path = Scan.scan_file_path(path, "sens")
        frames = Frames.load_from_file(sens_path)

        return Scan(path, scan_id, frames=frames, cache=cache)

    def __init__(self, path: str, id: int, frames: Frames, cache: CacheType) -> None:
        self._parent: Space = None
        self._path = path
        self._id = id
        self._frames = frames
        self._frames._parent = self
        self._txt = None
        self._mesh = None
        self._cache = cache

    @property
    def parent(self) -> Space:
        return self._parent

    @property
    def id(self) -> int:
        return self._id

    @property
    def uid(self) -> Tuple[int, int]:
        return (self.parent.id, self.id)

    @property
    def frames(self) -> Frames:
        return self._frames

    @property
    def cache(self) -> CacheType:
        return self._cache

    @property
    def txt(self) -> Txt:
        if self._txt is None:
            txt_path = Scan.scan_file_path(self._path, "txt")
            self._txt = Txt(txt_path)
        return self._txt

    def mesh(self, mode="clean") -> Mesh:
        assert mode in {"clean", "clean_2", "clean_2.labels"}

        mesh_path = Scan.scan_file_path(self._path, "ply", mode)
        self._mesh = Mesh(mesh_path)

        return self._mesh

    def __repr__(self) -> str:
        return f"Scan(path={self._path}, id={self._id}, frames={self._frames}, txt={self._txt})"

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx]


class Txt(dict):
    """Loads information contained in the ".txt" file."""

    SCENE_TYPE_LIST = [
        "Apartment",
        "Bathroom",
        "Bedroom / Hotel",
        "Bookstore / Library",
        "Classroom",
        "Closet",
        "ComputerCluster",
        "Conference Room",
        "Copy/Mail Room",
        "Dining Room",
        "Game room",
        "Gym",
        "Hallway",
        "Kitchen",
        "Laundry Room",
        "Living room / Lounge",
        "Lobby",
        "Misc.",
        "Office",
        "Stairs",
        "Storage/Basement/Garage",
    ]

    TEST_SET_SCENE_TYPE_FIX = {
        "scene0707_00": "Kitchen",
        "scene0708_00": "Living room / Lounge",
        "scene0709_00": "Kitchen",
        "scene0710_00": "Office",
        "scene0711_00": "Office",
        "scene0712_00": "Bedroom / Hotel",
        "scene0713_00": "Bathroom",
        "scene0714_00": "Living room / Lounge",
        "scene0715_00": "Hallway",
        "scene0716_00": "Laundry Room",
        "scene0717_00": "Bedroom / Hotel",
        "scene0718_00": "Hallway",
        "scene0719_00": "Bedroom / Hotel",
        "scene0720_00": "Bedroom / Hotel",
        "scene0721_00": "Bedroom / Hotel",
        "scene0722_00": "Bedroom / Hotel",
        "scene0723_00": "Bedroom / Hotel",
        "scene0724_00": "Bedroom / Hotel",
        "scene0725_00": "Bedroom / Hotel",
        "scene0726_00": "Bathroom",
        "scene0727_00": "Bathroom",
        "scene0728_00": "Bathroom",
        "scene0729_00": "Bathroom",
        "scene0730_00": "Bedroom / Hotel",
        "scene0731_00": "Bedroom / Hotel",
        "scene0732_00": "Kitchen",
        "scene0733_00": "Living room / Lounge",
        "scene0734_00": "Bedroom / Hotel",
        "scene0735_00": "Bathroom",
        "scene0736_00": "Office",
        "scene0737_00": "Office",
        "scene0738_00": "Bedroom / Hotel",
        "scene0739_00": "Office",
        "scene0740_00": "Bedroom / Hotel",
        "scene0741_00": "Bedroom / Hotel",
        "scene0742_00": "Bathroom",
        "scene0743_00": "Kitchen",
        "scene0744_00": "Living room / Lounge",
        "scene0745_00": "Office",
        "scene0746_00": "Kitchen",
        "scene0747_00": "Living room / Lounge",
        "scene0748_00": "Living room / Lounge",
        "scene0749_00": "Kitchen",
        "scene0750_00": "Bathroom",
        "scene0751_00": "Bathroom",
        "scene0752_00": "Bedroom / Hotel",
        "scene0753_00": "Office",
        "scene0754_00": "Office",
        "scene0755_00": "Office",
        "scene0756_00": "Office",
        "scene0757_00": "Apartment",
        "scene0758_00": "Office",
        "scene0759_00": "Office",
        "scene0760_00": "Office",
        "scene0761_00": "Apartment",
        "scene0762_00": "Office",
        "scene0763_00": "Closet",
        "scene0764_00": "Bathroom",
        "scene0765_00": "Bathroom",
        "scene0766_00": "Bedroom / Hotel",
        "scene0767_00": "Bathroom",
        "scene0768_00": "Bedroom / Hotel",
        "scene0769_00": "Office",
        "scene0770_00": "Living room / Lounge",
        "scene0771_00": "Conference Room",
        "scene0772_00": "Bathroom",
        "scene0773_00": "Kitchen",
        "scene0774_00": "Conference Room",
        "scene0775_00": "Bathroom",
        "scene0776_00": "Game room",
        "scene0777_00": "Game room",
        "scene0778_00": "Bathroom",
        "scene0779_00": "Bathroom",
        "scene0780_00": "Laundry Room",
        "scene0781_00": "Living room / Lounge",
        "scene0782_00": "Living room / Lounge",
        "scene0783_00": "Living room / Lounge",
        "scene0784_00": "Apartment",
        "scene0785_00": "Apartment",
        "scene0786_00": "Living room / Lounge",
        "scene0787_00": "Storage/Basement/Garage",
        "scene0788_00": "Gym",
        "scene0789_00": "Hallway",
        "scene0790_00": "Copy/Mail Room",
        "scene0791_00": "Classroom",
        "scene0792_00": "Stairs",
        "scene0793_00": "Classroom",
        "scene0794_00": "Dining Room",
        "scene0795_00": "Dining Room",
        "scene0796_00": "Dining Room",
        "scene0797_00": "Dining Room",
        "scene0798_00": "Dining Room",
        "scene0799_00": "Bookstore / Library",
        "scene0800_00": "Bookstore / Library",
        "scene0801_00": "Dining Room",
        "scene0802_00": "Lobby",
        "scene0803_00": "Storage/Basement/Garage",
        "scene0804_00": "Copy/Mail Room",
        "scene0805_00": "Living room / Lounge",
        "scene0806_00": "Game room",
    }

    @staticmethod
    def _remove_new_line(value: str):
        if len(value) > 0 and value[-1] == "\n":
            value = value[:-1]
        return value

    def __init__(self, path) -> None:
        super().__init__()
        self._path = path

        # TEMPORARY(Pierre): Fix missing label classes in test set
        # TODO(Pierre): Integrate in original files directly
        scan_name = os.path.basename(path)
        if scan_name.endswith(".txt"):
            scan_name = scan_name[: -len(".txt")]

        if scan_name in Txt.TEST_SET_SCENE_TYPE_FIX:
            # TODO(Pierre): Add warning here
            self["sceneType"] = Txt.TEST_SET_SCENE_TYPE_FIX[scan_name]

        with open(path, "r") as f:
            for line in f.readlines():
                key, value = line.split(" = ")
                self[key] = Txt._remove_new_line(value)


class Mesh:
    def __init__(
        self,
        path,
    ):
        self._path = path
        self.mesh_name = os.path.basename(path).removesuffix(".ply")

    # TODO(Pierre) : Where does IO come from ?
    # def get_mesh(self, device="cpu"):
    #     return IO().load_mesh(self._path, device=device)


class Frames(Video):
    """The Frames class handled the loading / reading of the ".sens" files containing RGB and depth frames."""

    SENS_VERSION = 4
    FRAME_RATE = 30
    SENS_VALID_COLOR_COMPRESSION_TYPE = "jpeg"
    SENS_VALID_DEPTH_COMPRESSION_TYPE = "zlib_ushort"
    COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
    COMPRESSION_TYPE_DEPTH = {
        -1: "unknown",
        0: "raw_ushort",
        1: "zlib_ushort",
        2: "occi_ushort",
    }

    @staticmethod
    def load_from_file(path: str) -> Frames:
        """Load Frames object by providing path to ".sens" file."""
        with open(path, "rb") as f:
            header = Frames._read_sens_header(f)
        return Frames(path, header)

    @staticmethod
    def _read_sens_header(reader):
        """Read header of ".sens" file."""
        header = {}

        # version
        header["version"] = struct.unpack("I", reader.read(4))[0]
        assert header["version"] == Frames.SENS_VERSION

        # sensor name
        strlen = struct.unpack("Q", reader.read(8))[0]
        header["sensor_name"] = reader.read(strlen).decode("utf-8")

        # camera poses
        header["intrinsic_color"] = np.asarray(
            struct.unpack("f" * 16, reader.read(16 * 4)),
            dtype=np.float32,
        ).reshape(4, 4)
        header["extrinsic_color"] = np.asarray(
            struct.unpack("f" * 16, reader.read(16 * 4)),
            dtype=np.float32,
        ).reshape(4, 4)
        header["intrinsic_depth"] = np.asarray(
            struct.unpack("f" * 16, reader.read(16 * 4)),
            dtype=np.float32,
        ).reshape(4, 4)
        header["extrinsic_depth"] = np.asarray(
            struct.unpack("f" * 16, reader.read(16 * 4)),
            dtype=np.float32,
        ).reshape(4, 4)

        header["color_compression_type"] = Frames.COMPRESSION_TYPE_COLOR[
            struct.unpack("i", reader.read(4))[0]
        ]
        assert (
            header["color_compression_type"] == Frames.SENS_VALID_COLOR_COMPRESSION_TYPE
        )

        header["depth_compression_type"] = Frames.COMPRESSION_TYPE_DEPTH[
            struct.unpack("i", reader.read(4))[0]
        ]
        assert (
            header["depth_compression_type"] == Frames.SENS_VALID_DEPTH_COMPRESSION_TYPE
        )

        header["color_width"] = struct.unpack("I", reader.read(4))[0]
        header["color_height"] = struct.unpack("I", reader.read(4))[0]
        header["depth_width"] = struct.unpack("I", reader.read(4))[0]
        header["depth_height"] = struct.unpack("I", reader.read(4))[0]
        header["depth_shift"] = struct.unpack("f", reader.read(4))[0]
        header["num_frames"] = struct.unpack("Q", reader.read(8))[0]

        # useful to skip header next time we read the file
        header["header_size"] = reader.tell()

        return header

    def __init__(self, path: str, header: Dict[str, Any]) -> None:
        self._parent: Scan = None
        self._path = path
        self._header = header
        self._frames = None
        self._duration = self._header["num_frames"] / self.FRAME_RATE

    @property
    def header(self):
        return self._header

    def _load_frames(self) -> None:
        # check if memory cached
        if self._frames is not None:
            return

        # check if disk cached
        cache = self.parent.cache
        if self._path in cache:
            self._frames = pkl.loads(cache[self._path])
            for frame in self._frames:
                frame._parent = self
            return

        with open(self._path, "rb") as f:
            # skip header
            f.seek(self._header["header_size"])
            self._frames = [Frame.load_from_reader(f) for _ in range(len(self))]
            for frame in self._frames:
                frame._parent = self

        # add frames to disk cache
        cache[self._path] = pkl.dumps(self._frames)

    def __getitem__(self, idx: int) -> Frame:
        # load frame offsets / sizes only during first call to `__getitem__`
        self._load_frames()
        return self._frames[idx]

    def __len__(self) -> int:
        return self._header["num_frames"]

    def __repr__(self) -> str:
        return f"Frames(path={self._path}, header={self._header})"

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def parent(self) -> Scan:
        return self._parent

    def get_clip(
        self,
        start_sec: float,
        end_sec: float,
        frame_filter: Optional[Callable[[List[int]], List[int]]] = None,
        frame_transform: Optional[
            Callable[[np.ndarray], Union[torch.Tensor, np.ndarray]]
        ] = None,
        multithreaded: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if start_sec < 0 or end_sec > self.duration:
            raise RuntimeError(
                f"out-of-bound `start_sec`, should be in range [0, {self._duration}]"
            )

        if end_sec > self.duration:
            LOG.warning(
                f"provided `end_sec` has been clamped since it has been found to be above video duration ({end_sec} > {self._duration})"
            )
            end_sec = self._duration

        idx_start = math.ceil(start_sec * Frames.FRAME_RATE)
        idx_end = math.ceil(end_sec * Frames.FRAME_RATE)
        idx_end = min(idx_end, len(self))

        idxs = list(range(idx_start, idx_end))
        if frame_filter:
            idxs = frame_filter(idxs)

        frames = [self[i] for i in idxs]

        # extract color frames
        def set_color_frame(i, frame):
            frames[i] = frame.color

        optional_threaded_foreach(set_color_frame, enumerate(frames), multithreaded)

        # transform frames
        if frame_transform:

            def transform_frame(i, frame):
                frames[i] = frame_transform(frame)

            optional_threaded_foreach(transform_frame, enumerate(frames), multithreaded)

        # check output type
        frames = [
            torch.from_numpy(frame) if isinstance(frame, np.ndarray) else frame
            for frame in frames
        ]

        frames = torch.stack(frames, dim=0)
        frames = thwc_to_cthw(frames)
        frames = frames.to(torch.float32)

        return {"video": frames, "frame_indices": idxs, "audio": None}


class Frame:
    """Contains all frame related information."""

    STATE_ATTRS = {
        "_header_offset",
        "_color_offset",
        "_color_size",
        "_depth_offset",
        "_depth_size",
    }

    @staticmethod
    def _read_header(reader):
        header = {}
        header["camera_to_world"] = np.asarray(
            struct.unpack("f" * 16, reader.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        header["timestamp_color"] = struct.unpack("Q", reader.read(8))[0]
        header["timestamp_depth"] = struct.unpack("Q", reader.read(8))[0]

        color_size = struct.unpack("Q", reader.read(8))[0]
        depth_size = struct.unpack("Q", reader.read(8))[0]
        return header, color_size, depth_size

    @staticmethod
    def load_from_reader(reader):
        """Load frame from read by only gathering lightweight information (ignoring color and depth frames)."""

        header_offset = reader.tell()
        _, color_size, depth_size = Frame._read_header(reader)

        # useful for quick random access loading later (reduces memory uusage)
        color_offset = reader.tell()
        reader.seek(color_size, os.SEEK_CUR)
        depth_offset = reader.tell()
        reader.seek(depth_size, os.SEEK_CUR)

        return Frame(
            header_offset,
            color_offset,
            color_size,
            depth_offset,
            depth_size,
        )

    def __init__(
        self,
        header_offset,
        color_offset,
        color_size,
        depth_offset,
        depth_size,
    ) -> None:
        self._parent: Frames = None
        self._header_offset = header_offset
        self._color_offset = color_offset
        self._color_size = color_size
        self._depth_offset = depth_offset
        self._depth_size = depth_size

    def _read_bytes(self, offset, size):
        """Read bytes at specific offset / size."""
        with open(self._parent._path, "rb") as f:
            f.seek(offset)
            return f.read(size)

    @property
    def header(self):
        with open(self._parent._path, "rb") as f:
            f.seek(self._header_offset)
            header, _, _ = Frame._read_header(f)
        return header

    @property
    def color(self):
        """Get color frame as a HxWx3 numpy array."""
        bytes = self._read_bytes(self._color_offset, self._color_size)
        array = imageio.imread(bytes)
        return np.asarray(array)

    @property
    def depth(self):
        """Get depth frame as a HxW numpy array."""
        bytes = self._read_bytes(self._depth_offset, self._depth_size)
        bytes = zlib.decompress(bytes)
        depth = np.frombuffer(bytes, dtype=np.uint16)
        header = self._parent._header
        depth_shift = self._parent.header["depth_shift"]
        depth = depth / depth_shift
        return depth.reshape(header["depth_height"], header["depth_width"])

    def __repr__(self) -> str:
        return f"Frame(header={self.header}, color_offset={self._color_offset}, color_size={self._color_size}, depth_offset={self._depth_offset}, depth_size={self._depth_size})"

    @property
    def parent(self) -> Frames:
        return self._parent

    @property
    def scan(self) -> Scan:
        return self.parent.parent

    def __getstate__(self):
        return {attr: getattr(self, attr) for attr in Frame.STATE_ATTRS}

    def __setstate__(self, newstate):
        keys = set(newstate.keys())
        if keys != Frame.STATE_ATTRS:
            raise RuntimeError(
                f"invalid keys found : {keys}, should be : {Frame.STATE_ATTRS}"
            )
        self.__dict__.update(newstate)


class ScanIDFilter:
    def __init__(self, scan_uids: Union[str, Iterable[Tuple[int, int]]]) -> None:
        if isinstance(scan_uids, str):
            with open(scan_uids, "r") as f:
                scan_uids = json.load(f)

        self._uids = {tuple(uid) for uid in scan_uids}

    def __call__(self, scan: Scan) -> bool:
        return scan.uid in self._uids


class ScanNet:
    """Helper class to load ScanNet data in an efficient way (speed and memory-wise).
    We do only load the necessary data as they are required, in order to void huge memory footprint and costly initialization.
    IMPORTANT : This class is NOT a PyTorch dataset. Please refer to `ScansDataset` and `FramesDataset` for that.
    """

    def __init__(
        self,
        path: str,
        cache_path: Union[str, None] = None,
        scan_filter: Callable[[Scan], bool] = None,
    ) -> None:
        self._path = path
        self._scans = []
        self._spaces = []
        self._space_id_to_space = {}
        self._frames = []
        self._cache = (
            _AlwaysEmptyCache() if cache_path is None else _DiskCache(cache_path)
        )

        self._load_initial_data(scan_filter)

    @property
    def n_spaces(self):
        """Return number of spaces found."""
        return len(self._spaces)

    @property
    def n_scans(self):
        """Return number of scans found."""
        return len(self._scans)

    @property
    def n_frames(self):
        """Return number of frames found."""
        return self._n_frames

    def space(self, idx):
        """Access to ith Space."""
        return self._spaces[idx]

    def scan(self, idx):
        """Access to ith Scan."""
        return self._scans[idx]

    def frame(self, idx):
        """Access to ith Frame."""
        frame = self._frames[idx]
        # lazy initialization of requested frame
        if isinstance(frame, tuple):
            self._frames[idx] = frame = self._scans[frame[0]][frame[1]]
        return frame

    def _load_initial_data(self, scan_filter):
        LOG.info(f"loading ScanNet dataset from : {self._path}")
        self._n_frames = 0
        glob_path = os.path.join(self._path, "scene*")
        for scan_path in sorted(glob(glob_path, recursive=False)):
            # try to load scan
            scan = Scan.load_from_directory(scan_path, self._cache)

            if scan is None:
                scan_basename = os.path.basename(scan_path)
                LOG.warning(f'"{scan_basename}" is not a scene, skipping ...')
                continue

            spaceId, _ = Scan.space_id_scan_id_from_path(scan_path)

            # check if same space has already been loaded
            if not (spaceId in self._space_id_to_space):
                space = Space(spaceId)
                self._space_id_to_space[spaceId] = space
                self._spaces.append(space)
            else:
                space = self._space_id_to_space[spaceId]

            space._add(scan)

            if (scan_filter is not None) and (not scan_filter(scan)):
                space._remove(scan)
                continue

            self._frames.extend((len(self._scans), i) for i in range(len(scan)))
            self._n_frames += len(scan)
            self._scans.append(scan)

        LOG.info(f"{self.n_spaces} spaces found")
        LOG.info(f"{self.n_scans} scans found")
        LOG.info(f"{self.n_frames} frames found")
