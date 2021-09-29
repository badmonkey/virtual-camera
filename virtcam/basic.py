import cv2
import numpy as np
from pipey import Pipeable

from virtcam.base import Frame, FrameFilter, FrameProcessor, Image, Mask, immutable


class Mirror(FrameFilter):
    def __init__(self, source: FrameProcessor):
        super().__init__(source)

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)
        image = cv2.flip(frame.image, 1)
        mask = cv2.flip(frame.mask, 1)
        return Frame(frame.config, immutable(image), immutable(mask))

    @staticmethod
    @Pipeable
    def p(src):
        return Mirror(src)
