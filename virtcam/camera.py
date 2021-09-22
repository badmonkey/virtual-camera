import signal
import sys
import time

import cv2
import numpy as np
import pyfakewebcam
from inotify_simple import INotify, flags
from pipey import Pipeable

import virtcam.debug as debug
from virtcam.base import FrameProcessor, FrameSink, FrameSource, Image


class Camera(FrameSink):
    def __init__(self, source: FrameProcessor):
        super().__init__(source)
        self.loopback_path = "/dev/video2"
        self.framecap_path = "framecap.png"
        self.maskcap_path = "maskcap.png"
        self.loopback = pyfakewebcam.FakeWebcam(
            self.loopback_path, self.config.width, self.config.height
        )
        watch_flags = flags.CREATE | flags.OPEN | flags.CLOSE_NOWRITE | flags.CLOSE_WRITE
        self.inotify = INotify(nonblocking=True)
        wd = self.inotify.add_watch(self.loopback_path, watch_flags)
        self.paused = True
        self.consumers = 0
        self.take_frameshot = False
        self.blank_image = Image(self.config.width, self.config.height)
        # signal.signal(signal.SIGQUIT, self.handle_pause)
        signal.signal(signal.SIGINT, self.handle_quit)
        signal.signal(signal.SIGTSTP, self.handle_snapshot)
        print("No consumers, paused")

    def check_consumers(self):
        for event in self.inotify.read(0):
            for flag in flags.from_mask(event.mask):
                if flag == flags.CLOSE_NOWRITE or flag == flags.CLOSE_WRITE:
                    self.consumers -= 1
                if flag == flags.OPEN:
                    self.consumers += 1
            if self.consumers > 0:
                self.paused = False
                # load_images
                print("Consumers:", self.consumers)
            else:
                self.consumers = 0
                self.paused = True
                print("No consumers remaining, paused")

    def toggle(self):
        if not self.paused:
            self.paused = True
            print("\nPaused.")
        elif self.consumers > 0:
            print("\nResuming, reloading background / foreground images...")
            # self.load_images()
            self.paused = False
        else:
            print("\nRemaining paused, reloading background / foreground images...")
            # self.load_images()

    def run(self):
        fid = 0
        while True:
            self.check_consumers()

            if not self.paused:
                # print("CAPTURE sources")
                FrameSource.capture_frames()
                # print("PROCESS frames")
                frame = self.source.next(fid)

                if self.take_frameshot:
                    cv2.imwrite(self.framecap_path, frame.image)
                    mask = np.uint8(np.clip(frame.mask * 255, 0, 255))
                    mask = np.uint8(np.dstack([mask, mask, mask]))
                    cv2.imwrite(self.maskcap_path, mask)
                    self.take_frameshot = False

                # debug.frame("Camera:run", frame)
                self.loopback.schedule_frame(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB))
                fid += 1
            else:
                self.loopback.schedule_frame(self.blank_image)
                time.sleep(1)

    def handle_snapshot(self, signal, frame):
        print("capture frame")
        self.take_frameshot = True

    def handle_pause(self, signal, frame):
        print("handle reload")
        self.toggle()

    def handle_quit(self, signal, frame):
        print("\n\ndone")
        sys.exit(0)

    @staticmethod
    @Pipeable
    def p(src):
        return Camera(src)
