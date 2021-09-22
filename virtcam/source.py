import cv2
from virtcam.base import Frame, Image, StreamConfig, FrameSource, DEFAULT_FPS
from virtcam.color import BLACK, WHITE, RED, GREEN, BLUE, color_by_name
import virtcam.debug as debug


class MatteFrameSource(FrameSource):
    def __init__(self, config: StreamConfig, color: str):
        super().__init__(config)
        rgbcolor = color_by_name(color)
        image = Image(config.width, config.height, rgbcolor)
        self.frame = Frame(config, image, self.fullmask)

    def next(self, frame_id: int) -> Frame:
        # debug.frame("MatteFrameSource:next", self.frame)
        return self.frame


class ImageSource(FrameSource):
    def __init__(self, imgFile: str, maskFile: str = None):
        super().__init__()

        img = cv2.imread(imgFile)
        sizey, sizex = img.shape[0], img.shape[1]

        if maskFile:
            mask = cv2.imread(maskFile)
            # check compatible sizes
            mask = cv2.normalize(
                mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # inverted_mask = 1 - mask
        else:
            mask = None

        self._init_config(StreamConfig(sizex, sizey, DEFAULT_FPS), mask)
        self.frame = Frame(self.config, img, self.fullmask)

        debug.config("Image:init:config", self.config)

    def next(self, frame_id: int) -> Frame:
        # debug.frame(f"Image:next[{frame_id}]", self.frame)
        return self.frame
