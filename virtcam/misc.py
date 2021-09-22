import cv2
import numpy as np
from pipey import Pipeable

from virtcam.base import WHITE, Frame, FrameFilter, FrameProcessor, Image, IntegerRGB, Mask


class DisplayMask(FrameFilter):
    def __init__(self, source: FrameProcessor, rgb: IntegerRGB = WHITE):
        super().__init__(source)
        self.red, self.green, self.blue = float(rgb.red), float(rgb.green), float(rgb.blue)

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)

        mask = np.uint8(np.clip(frame.mask * 255, 0, 255))
        return Frame(
            frame.config,
            np.uint8(np.dstack([mask, mask, mask])),
            self.fullmask,
        )

    @staticmethod
    @Pipeable
    def p(src, rgb=WHITE):
        return DisplayMask(src, rgb)


class MaskRunningAverage(FrameFilter):
    def __init__(self, source: FrameProcessor, ratio: float):
        super().__init__(source)
        self.mrar = ratio
        self.previous = source.fullmask

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)

        if self.mrar < 1.0:
            if self.previous is None:
                self.previous = frame.mask

            mask = frame.mask * self.mrar + self.previous * (1.0 - self.mrar)
            self.previous = mask

            return Frame(frame.config, frame.image, mask)

        return frame

    @staticmethod
    @Pipeable
    def p(src, ratio: float):
        return MaskRunningAverage(src, ratio)


class MaskPassFilter(FrameFilter):
    def __init__(self, source: FrameProcessor, low: float, high: float):
        super().__init__(source)
        self.low = low
        self.high = high

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)

        mask = (frame.mask > self.high) * frame.mask

        return Frame(frame.config, frame.image, mask)

    @staticmethod
    @Pipeable
    def p(src, low, high):
        return MaskPassFilter(src, low, high)
