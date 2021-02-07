from enum import IntEnum
import cv2


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


def scale(image, scale_factor, interpolation=cv2.INTER_AREA):
    height, width = image.shape[0:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation)
