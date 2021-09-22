import cv2

import virtcam.debug as debug
from virtcam.base import Frame, FrameSource, Image, Mask, StreamConfig


class Webcam(FrameSource):
    def __init__(self):
        super().__init__()
        self.current_id = -1
        self.camera = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

        c1, c2, c3, c4 = "M", "J", "P", "G"
        codec = cv2.VideoWriter_fourcc(c1, c2, c3, c4)
        self.camera.set(cv2.CAP_PROP_FOURCC, codec)

        camConfig = StreamConfig(
            int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.camera.get(cv2.CAP_PROP_FPS)),
        )
        self._init_config(camConfig)
        self.frame = Frame(self.config, Image(self.config.width, self.config.height), self.fullmask)

        debug.config("Webcam:init:config", camConfig)

    def grab(self) -> bool:
        return True

    def next(self, frame_id: int) -> Frame:
        if not self.frame or self.current_id != frame_id:
            grabbed = False
            while not grabbed:
                grabbed, image = self.camera.read()

            self.frame = Frame(self.config, image, self.fullmask)
            self.current_id = frame_id
        # debug.frame(f"Webcam:next[{frame_id}]", self.frame)
        return self.frame
