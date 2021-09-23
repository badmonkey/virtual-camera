import click

from virtcam.camera import Camera
from virtcam.classifier import FaceMeshDrawer, HandDrawer, HolisticDrawer, SelfieSegmentator
from virtcam.compose import Composite
from virtcam.fade import CrossFade
from virtcam.effects import Erode, Hologram
from virtcam.misc import DisplayMask, MaskPassFilter, MaskRunningAverage
from virtcam.base import StreamConfig
from virtcam.source import (
    MatteFrameSource,
    BLACK,
    BLUE,
    GREEN,
    RED,
    WHITE,
    ImageSource,
    VideoSource,
)
from virtcam.webcam import Webcam


@click.command()
def main():
    config = StreamConfig(960, 540, 30)

    # webcam = Webcam()
    # image1 = ImageSource("images/test-signal.jpg")
    image2 = ImageSource("images/person2.jpg", "images/person2_mask.png")
    vid1 = VideoSource("images/background-itsfine.gif")
    # green = MatteFrameSource(config, "green")

    # camera = green >> Camera.p
    # camera = webcam >> Camera.p

    # camera = vid1 >> Camera.p

    # camera = image1 >> Camera.p
    # camera = image2 >> Camera.p
    # camera = image2 >> DisplayMask.p(WHITE) >> Camera.p

    # camera = image2 >> Hologram.p >> Composite.p(image2) >> Camera.p
    # camera = image2 >> SelfieSegmentator.p >> Camera.p

    # camera = (
    #     webcam
    #     >> SelfieSegmentator.p
    #     # >> MaskPassFilter.p(0.1, 0.9)
    #     >> MaskRunningAverage.p(0.2)
    #     >> DisplayMask.p(WHITE)
    #     >> Camera.p
    # )

    # camera = webcam >> Hologram.p >> Camera.p
    # camera = webcam >> SelfieSegmentator.p >> Camera.p
    camera = image2 >> Composite.p(vid1) >> Camera.p

    # camera = webcam >> SelfieSegmentator.p >> Composite.p(webcam >> Hologram.p) >> Camera.p
    # camera = webcam >> SelfieSegmentator.p >> Hologram.p >> Composite.p(webcam) >> Camera.p

    # camera = (
    #     webcam
    #     >> SelfieSegmentator.p
    #     >> MaskPassFilter.p(0.1, 0.9)
    #     >> MaskRunningAverage.p(0.2)
    #     >> Composite.p(green)
    #     >> Camera.p
    # )

    # camera = webcam >> FaceMeshDrawer.p >> Camera.p

    # camera = webcam >> Erode.p(11) >> Camera.p

    # camera = webcam >> CrossFade.p(webcam >> Erode.p(11)) >> Camera.p

    camera.run()


if __name__ == "__main__":
    main()
