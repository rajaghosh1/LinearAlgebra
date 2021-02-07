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


def scale(image, scale_factor=1, interpolation=cv2.INTER_AREA):
    return stretch_by_factor(image, scale_width=scale_factor, scale_height=scale_factor, interpolation=interpolation)


def stretch_by_factor(image, scale_width=1, scale_height=1, interpolation=cv2.INTER_AREA):
    if ((scale_width is None) or (scale_width == 1)) and ((scale_height is None) or (scale_height == 1)):
        return image.copy()
    height, width = image.shape[0:2]
    new_width = int(width * scale_width)
    new_height = int(height * scale_height)
    return cv2.resize(image, (new_width, new_height), interpolation)

