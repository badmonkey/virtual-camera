import cv2
import numpy as np
from pipey import Pipeable
import virtcam.debug as debug
from virtcam.base import (
    Frame,
    FrameFilter,
    FrameProcessor,
    Image,
    StreamConfig,
    PROBE_FRAME,
    Mask,
    immutable,
)


class Embed(FrameFilter):
    def __init__(self, source: FrameProcessor, config: StreamConfig):
        super().__init__(source)

        self.setconfig(config)

        frame = source.next(PROBE_FRAME)
        if frame.config.width > config.width or frame.config.height > config.height:
            raise RuntimeError("Source image too big for container frame")

        self.centerX = (config.width - frame.config.width) / 2
        self.centerY = (config.height - frame.config.height) / 2
        self.dim = (config.width, config.height)

        self._fullmask = Mask(config.width, config.height)

        debug.config("Embed:init:config", config)

    @property
    def fullmask(self) -> np.ndarray:
        return self._fullmask

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)
        image = immutable(cv2.resize(frame.image, self.dim))
        return Frame(frame.config, image, self.fullmask)

    @staticmethod
    @Pipeable
    def p(src, cfg):
        return Embed(src, cfg)
