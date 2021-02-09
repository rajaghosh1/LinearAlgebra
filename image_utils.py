from enum import IntEnum
import cv2
import numpy as np
import math


class Color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2


class Dimension(IntEnum):
    X = 0
    Y = 1
    RED = 2
    GREEN = 3
    BLUE = 4


DEGREE_TO_RADIANS = math.pi / 180


def orthonormal_basis_vector(dimension):
    if (dimension < Dimension.X) or (dimension > Dimension.BLUE):
        raise ValueError("invalid dimension=" + dimension)
    vector = np.zeros([5])
    vector[dimension] = 1
    return vector


def identity_matrix():
    matrix = np.zeros([5, 5])
    for i in range(5):
        matrix[i, i] = 1
    return matrix


def normalize(vectors):
    x_offset = 0.0
    y_offset = 0.0
    row_min = np.amin(vectors, axis=0)
    if row_min[0] < 0:
        x_offset = -row_min[0]
    if row_min[1] < 0:
        y_offset = -row_min[1]

    if (x_offset != 0.0) or (y_offset != 0.0):
        offsets = [x_offset, y_offset, 0, 0, 0]
        vectors = vectors.copy() + offsets

    return vectors


def rotational_basis_vectors(angle, basis_map=None, is_radian=True):
    new_basis_map = dict()
    if basis_map is not None:
        for k in basis_map:
            new_basis_map[k] = basis_map[k]

    angle_in_radians = angle
    if is_radian is not True:
        angle_in_radians = angle * DEGREE_TO_RADIANS

    new_basis_map[Dimension.X] = [math.cos(angle_in_radians), math.sin(angle_in_radians), 0, 0, 0]
    new_basis_map[Dimension.Y] = [-math.sin(angle_in_radians), math.cos(angle_in_radians), 0, 0, 0]

    return new_basis_map


def transform(vectors, basis_map=None):
    transformation_matrix = np.zeros([5, 5])
    for i in range(5):
        basis = orthonormal_basis_vector(i)
        if (basis_map is not None) and (i in basis_map):
            basis = basis_map[i]
        transformation_matrix[i] = basis

    transformation_matrix = transformation_matrix.transpose()

    transposed_vectors = vectors.transpose()

    transformed = np.matmul(transformation_matrix, transposed_vectors)
    return transformed.transpose()


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


def image2vectors(image):
    height, width = image.shape[0:2]
    vector_size = height * width
    vectors = np.zeros([vector_size, 5])
    count = 0
    for h in range(height):
        for w in range(width):
            rgb = image[h, w]
            vectors[count] = np.array([w, h, rgb[0], rgb[1], rgb[2]])
            count = count + 1
    return vectors


def vectors2image(vectors):
    vectors = normalize(vectors)

    row_max = np.amax(vectors, axis=0)
    max_width_double = math.ceil(row_max[0] + 1)
    max_width = int(max_width_double)
    max_height_double = math.ceil(row_max[1] + 1)
    max_height = int(max_height_double)

    image = np.zeros((max_height, max_width, 3), dtype=np.dtype(np.uint8))
    for v in vectors:
        x = int(math.ceil(v[0]))
        y = int(math.ceil(v[1]))
        red = min(int(math.ceil(v[2])), 255)
        blue = min(int(math.ceil(v[3])), 255)
        green = min(int(math.ceil(v[4])), 255)

        image[y, x] = np.array([red, blue, green])

    return image
