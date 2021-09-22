import signal
import cv2
import numpy as np
from pipey import Pipeable

import virtcam.debug as debug
from virtcam.base import Frame, FrameFilter, FrameProcessor, Image, Mask


class CrossFade(FrameFilter):
    def __init__(self, source: FrameProcessor, other: FrameProcessor, fade: int = 100):
        super().__init__(source)
        # check sizes are the same
        self.primary = source
        self.other = other
        self.fadeMax = fade
        self.fadeCnt = 0
        self.processFade = False
        signal.signal(signal.SIGQUIT, self.handle_switch)

    def handle_switch(self, signal, frame):
        print("handle switch")
        self.switch()

    def switch(self):
        if self.processFade:
            self.fadeCnt = self.fadeMax - self.fadeCnt
            tmp = self.primary
            self.primary = self.other
            self.other = tmp
        else:
            self.processFade = True
            self.fadeCnt = self.fadeMax

    def next(self, frame_id: int) -> Frame:
        frame = self.primary.next(frame_id)

        if self.processFade:
            print(f"CrossFade:next {self.fadeCnt}")
            flip = self.other.next(frame_id)
            if self.fadeCnt > 0:
                frac = self.fadeCnt / self.fadeMax

                for c in range(frame.image.shape[2]):
                    frame.image[:, :, c] = frame.image[:, :, c] * frac + flip.image[:, :, c] * (
                        1 - frac
                    )

                self.fadeCnt -= 1
            else:
                frame = flip
                self.fadeCnt = 0
                tmp = self.primary
                self.primary = self.other
                self.other = tmp
                self.processFade = False

        # debug.frame(f"CrossFade:next[{frame_id}]", frame)
        return frame

    @staticmethod
    @Pipeable
    def p(src, other, fade=100):
        return CrossFade(src, other, fade)
