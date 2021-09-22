from typing import Tuple

from webcolors import IntegerRGB, name_to_rgb

BLACK = IntegerRGB(0, 0, 0)
WHITE = IntegerRGB(255, 255, 255)
RED = IntegerRGB(255, 0, 0)
GREEN = IntegerRGB(0, 255, 0)
BLUE = IntegerRGB(0, 0, 255)
RGB_SIZE = 3


def color_by_name(name: str) -> IntegerRGB:
    try:
        col = name_to_rgb(name)
    except:
        col = WHITE
    return col
