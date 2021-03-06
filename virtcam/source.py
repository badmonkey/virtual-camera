import cv2
from virtcam.base import Frame, Image, StreamConfig, FrameSource, DEFAULT_FPS, immutable
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

        image = cv2.imread(imgFile)
        if image is None:
            raise RuntimeError(f"Failed to open {imgFile}")

        sizey, sizex = image.shape[0], image.shape[1]

        if maskFile:
            mask = cv2.imread(maskFile)
            # check compatible sizes
            mask = cv2.normalize(
                mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            immutable(mask)
            # inverted_mask = 1 - mask
        else:
            mask = None

        self._init_config(StreamConfig(sizex, sizey, DEFAULT_FPS), mask)

        self.frame = Frame(self.config, immutable(image), self.fullmask)

        debug.config("Image:init:config", self.config)

    def next(self, frame_id: int) -> Frame:
        # debug.frame(f"Image:next[{frame_id}]", self.frame)
        return self.frame


class VideoFileSource(FrameSource):
    """
    Video source that we can seek to a frame (the first in particular).
    """

    def __init__(self, imgFile: str, maskFile: str = None):
        super().__init__()
        self.current_id = -1

        self.vid = cv2.VideoCapture(imgFile)
        if not self.vid.isOpened():
            raise RuntimeError(f"Couldn't open video '{imgFile}'")
        fps = self.vid.get(cv2.CAP_PROP_FPS)

        ret, frame = self.vid.read()
        assert ret, f"VideoFile:init: cannot read frame from {imgFile}"

        sizey, sizex = frame.shape[0], frame.shape[1]
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._init_config(StreamConfig(sizex, sizey, fps))

        self.frame = Frame(self.config, frame, self.fullmask)

        debug.config("VideoFile:init:config", self.config)

    def next(self, frame_id: int) -> Frame:
        if not self.frame or self.current_id != frame_id:
            ret, image = self.vid.read()
            if not ret:
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, image = self.vid.read()
                assert ret, "cannot read frame"

            self.frame = Frame(self.config, immutable(image), self.fullmask)
            self.current_id = frame_id

        # debug.frame(f"VideoFile:next[{frame_id}]", self.frame)
        return self.frame


class StreamingVideoSource(FrameSource):
    """
    A Live Video Stream (so no seeking to frames)
    """

    def __init__(self, imgFile: str, maskFile: str = None):
        super().__init__()
        self.current_id = -1

        self.vid = cv2.VideoCapture(imgFile)
        if not self.vid.isOpened():
            raise RuntimeError(f"Couldn't open video '{imgFile}'")
        fps = self.vid.get(cv2.CAP_PROP_FPS)

        ret, frame = self.vid.read()
        assert ret, f"StreamingVideo:init: cannot read frame from {imgFile}"

        sizey, sizex = frame.shape[0], frame.shape[1]

        self._init_config(StreamConfig(sizex, sizey, fps))

        self.frame = Frame(self.config, frame, self.fullmask)

        debug.config("StreamingVideo:init:config", self.config)

    def next(self, frame_id: int) -> Frame:
        if not self.frame or self.current_id != frame_id:
            ret, image = self.vid.read()
            if not ret:
                ret, image = self.vid.read()
                assert ret, "cannot read frame"

            self.frame = Frame(self.config, immutable(image), self.fullmask)
            self.current_id = frame_id

        # debug.frame(f"StreamingVideo:next[{frame_id}]", self.frame)
        return self.frame
