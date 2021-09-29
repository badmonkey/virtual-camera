import cv2
import numpy as np
from pipey import Pipeable

from virtcam.base import Frame, FrameFilter, FrameProcessor, Image, Mask, immutable


class Erode(FrameFilter):
    def __init__(self, source: FrameProcessor, size: int):
        super().__init__(source)
        self.kernel = np.ones((size, size), dtype=np.uint8)

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)
        image = cv2.erode(frame.image, self.kernel, iterations=None)
        return Frame(frame.config, immutable(image), frame.mask)

    @staticmethod
    @Pipeable
    def p(src, size):
        return Erode(src, size)


class Hologram(FrameFilter):
    def __init__(self, source: FrameProcessor):
        super().__init__(source)

    def next(self, frame_id: int) -> Frame:
        frame = self.source.next(frame_id)

        # add a blue tint
        holo = cv2.applyColorMap(frame.image, cv2.COLORMAP_WINTER)
        # add a halftone effect
        bandLength, bandGap = 3, 4
        for y in range(holo.shape[0]):
            if y % (bandLength + bandGap) < bandLength:
                holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)
                # add some ghosting
        holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
        holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)
        # combine with the original color, oversaturated
        image = cv2.addWeighted(frame.image, 0.5, holo_blur, 0.6, 0)

        return Frame(frame.config, immutable(image), frame.mask)

    @staticmethod
    @Pipeable
    def p(src):
        return Hologram(src)


def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0
    return img
