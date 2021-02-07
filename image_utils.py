from enum import IntEnum


class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


def filter_out(image, excluded_colors=None):
    copy = image.copy()
    if excluded_colors is not None:
        excluded_color_set = {color for color in excluded_colors}
        for color in excluded_color_set:
            copy[:, :, color] = 0
    return copy
