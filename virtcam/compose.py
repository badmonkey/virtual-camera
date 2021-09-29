import cv2
import numpy as np
from pipey import Pipeable

from virtcam.base import Frame, FrameFilter, FrameProcessor, Image, Mask, immutable


class Composite(FrameFilter):
    def __init__(self, foreground: FrameProcessor, background: FrameProcessor):
        super().__init__(foreground)
        self.background = background

    def next(self, frame_id: int) -> Frame:
        # print("NEXT Composite")
        frame = self.source.next(frame_id)
        back = self.background.next(frame_id)

        image = frame.image.copy()
        mask = frame.mask

        for c in range(image.shape[2]):
            image[:, :, c] = image[:, :, c] * mask + back.image[:, :, c] * (1 - mask)

        return Frame(frame.config, immutable(image), back.mask)

    @staticmethod
    @Pipeable
    def p(src, bkgd):
        return Composite(src, bkgd)
