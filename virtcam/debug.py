from virtcam.base import Frame, StreamConfig


def frame(msg: str, f: Frame):
    print(f"{msg}: image: [{f.image.shape}] * {f.image.dtype}")
    print(f"{msg}: mask: [{f.mask.shape}] * {f.mask.dtype}")


def config(msg: str, cfg: StreamConfig):
    print(f"{msg}: {cfg.width}x{cfg.height}@{cfg.fps}")
