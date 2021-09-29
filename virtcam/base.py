import contextlib
from abc import abstractmethod
from collections import namedtuple

# from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from typing import List

import numpy as np
from pipey import Pipeable

import virtcam.debug as debug
from virtcam.color import RGB_SIZE, WHITE, IntegerRGB, BLACK

PROBE_FRAME = -1

StreamConfig = namedtuple("StreamConfig", ("width", "height", "fps"))
Frame = namedtuple("Frame", ("config", "image", "mask"))


def immutable(data: np.ndarray) -> np.ndarray:
    data.flags.writeable = False
    return data


@contextlib.contextmanager
def mutating(array: np.ndarray):
    array.flags.writeable = True
    yield array
    array.flags.writeable = False


def Image(width: int, height: int, dflt: IntegerRGB = None) -> np.ndarray:
    data = np.full((height, width, RGB_SIZE), dflt or WHITE, dtype=np.uint8)
    data.flags.writeable = False
    return data


def Mask(width: int, height: int, dflt: float = None) -> np.ndarray:
    data = np.full((height, width), dflt or 1.0, dtype=np.float32)
    data.flags.writeable = False
    return data


DEFAULT_FPS = 30
DEFAULT_CONFIG = StreamConfig(640, 480, DEFAULT_FPS)
DEFAULT_FRAME = Image(640, 480)
DEFAULT_MASK = Mask(640, 480)


class FrameProcessingBase:
    def __init__(self, config: StreamConfig):
        self._config = config

    @property
    def config(self) -> StreamConfig:
        return self._config

    def setconfig(self, cfg: StreamConfig):
        self._config = cfg

    def reconfig(self):
        pass


class FrameProcessor(FrameProcessingBase):
    def __init__(self, config: StreamConfig):
        super().__init__(config)

    @property
    def fullmask(self) -> np.ndarray:
        return DEFAULT_MASK

    @abstractmethod
    def next(self, frame_id: int) -> Frame:
        pass


class FrameFilter(FrameProcessor):
    def __init__(self, source: FrameProcessor):
        super().__init__(source.config)
        self._source = source

    @property
    def source(self) -> FrameProcessor:
        return self._source

    @property
    def fullmask(self) -> np.ndarray:
        return self.source.fullmask


class FrameSource(FrameProcessor):
    cameras: List["FrameSource"] = list()

    def __init__(self, config: StreamConfig = None):
        super().__init__(config or DEFAULT_CONFIG)
        self._fullmask = Mask(config.width, config.height) if config else DEFAULT_MASK
        FrameSource.cameras.append(self)

    @property
    def fullmask(self) -> np.ndarray:
        return self._fullmask

    def _init_config(self, config: StreamConfig, mask=None):
        self._config = config
        self._fullmask = Mask(config.width, config.height) if mask is None else mask

    @abstractmethod
    def grab(self) -> bool:
        return True

    @staticmethod
    def capture_frames():
        working = FrameSource.cameras.copy()
        while working:
            cam = working.pop(0)
            if not cam.grab():
                working.append(cam)


class FrameSink(FrameProcessingBase):
    def __init__(self, source: FrameProcessor):
        super().__init__(source.config)
        self._source = source

    @property
    def source(self) -> FrameProcessor:
        return self._source
