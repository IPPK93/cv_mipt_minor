import cv2
import numpy as np


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    # Ваш код
    height, width = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, scale=1)
    abs_cos, abs_sin = abs(rot_matrix[0, 0]), abs(rot_matrix[0, 1])
    new_height = int(height * abs_cos + width * abs_sin)
    new_width = int(height * abs_sin + width * abs_cos)
    rot_matrix[:, 2] += new_width // 2 - width // 2, new_height // 2 - height // 2

    dst = cv2.warpAffine(image, rot_matrix, (new_width, new_height))

    return dst


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image: исходное изображение
    :param points1: точки до преобразования
    :param points2: точки после преобразования
    :return image: преобразованное изображение
    """
    # Ваш код
    transform_matrix = cv2.getAffineTransform(points1, points2)

    # Найдём точки, в которые перешли крайние точки на картинке
    left_down = transform_matrix @ np.array([[0, 0, 1]]).T
    left_up = transform_matrix @ np.array([[0, len(image) - 1, 1]]).T
    right_down = transform_matrix @ np.array([[len(image[0]) - 1, 0, 1]]).T
    right_up = transform_matrix @ np.array([[len(image[0]) - 1, len(image) - 1, 1]]).T

    corners = np.array([left_down, left_up, right_down, right_up])
    sizes = np.array([np.abs(b - a) for b in corners for a in corners])
    size = int(np.max(sizes[:, 0])), int(np.max(sizes[:, 1]))

    offset = int(np.min(corners[:, 0])), int(np.min(corners[:, 1]))
    transform_matrix[:, 2] -= offset

    image = cv2.warpAffine(image, transform_matrix, size)

    return image
