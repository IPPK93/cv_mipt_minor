import numpy as np
import cv2


def get_stripes_range(only_stripes: np.ndarray) -> list:
    """
    Получить полуинтервалы вида [begin, end) по axis = 0, на которых находятся полосы.

    :param only_stripes: массив, в котором полосы отмечены соответствующим цветом
    :return stripes_range: полуинтервалы [begin,end) по axis = 0, на которых находятся полосы
    """
    stripes_range = list()
    j = 0
    while j < len(only_stripes[0]):
        while j < len(only_stripes[0]) and not only_stripes[0, j]:
            j += 1
        stripes_range.append([j, 0])
        while j < len(only_stripes[0]) and only_stripes[0, j]:
            j += 1
        stripes_range[-1][1] = j
    stripes_range.pop()
    return stripes_range


def get_stripe_num(only_stripes: np.ndarray, stripes_range: list) -> int:
    """
    Получить номер пустой полосы

    :param only_stripes: массив, в котором полосы отмечены соответствующим цветом
    :param stripes_range: полуинтервалы [begin,end) по axis = 0, на которых находятся полосы
    :return stripe_num: номер пустой полосы либо -1, если таковых нет
        Заметим, что речь идёт не о полосах с препятствиями. Если машина стоит на полосе без препятствий,
        то ей не надо никуда перестраиваться, поэтому функция вернёт -1.
    """
    for stripe_num, (begin, end) in enumerate(stripes_range):
        for i in range(len(only_stripes)):
            for j in range(begin, end):
                if not only_stripes[i, j]:
                    break
            else:  # На данной строке ничего кроме полосы нет
                continue
            break
        else:  # На данной полосе ничего кроме полосы нет
            return stripe_num
    return -1


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение c цветовой палитрой BGR (!)
    :return road_number: номер дороги, на котором нет препятсвия на дороге
    """
    road_number = None
    # Ваш код тут
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    low_gray = (0, 0, 30)
    high_gray = (180, 30, 255)

    only_stripes = cv2.inRange(image, low_gray, high_gray)

    stripes_range = get_stripes_range(only_stripes)
    road_number = get_stripe_num(only_stripes, stripes_range)
    # Ваш код тут

    return road_number
