import numpy as np
from queue import Queue
import matplotlib.pyplot as plt


def get_cell_size(grid: np.ndarray) -> int:
    """
    Получить линейный размер клетки лабиринта

    :param grid: сетка (лабиринт)
    :return: линейный размер клетки лабиринта
    """
    size = 0
    for i in grid[0]:
        if i:
            size += 1
        elif size:
            break
    return size


def get_start_point(grid: np.ndarray, cell_size: int) -> tuple:
    """
    Получить начальную точку лабиринта (одной из, будем брать левую верхнюю на данной клетке)

    :param grid: сетка (лабиринт)
    :param cell_size: размер клетки лабиринта
    :return: координаты стартовой точки
    """
    for i in range(2, len(grid[0]))[::cell_size + 2]:
        if grid[0, i]:
            return 2, i
    return -1, -1


def get_end_point(grid: np.ndarray, cell_size: int) -> tuple:
    """
    Получить конечную точку лабиринта (одну из, будем брать левую верхнюю на данной клетке)

    :param grid: Сетка (лабиринт)
    :param cell_size: размер клетки лабиринта
    :return: Координаты конечной точки
    """
    for j in range(2, len(grid[-1]))[::cell_size + 2]:
        if grid[-1, j]:
            return len(grid) - 2 - cell_size, j
    return -1, -1


def has_path(grid: np.ndarray, u: tuple, v: tuple, cell_size: int) -> bool:
    """
    Проверить, нет ли стен между точками u и v соседних клеток

    :param grid: сетка (лабиринт)
    :param u: точка, из которой идём
    :param v: точка, в которую идём
    :param cell_size: размер клетки лабиринта
    :return: True, если есть прямой путь между u и v; False иначе
    """
    if u[0] == v[0]:
        if v[1] - u[1] > 0:
            return grid[u[0]][u[1] + cell_size] != 0
        else:
            return grid[u[0]][u[1] - 1] != 0
    else:  # u[1] == v[1]
        if v[0] - u[0] > 0:
            return grid[u[0] + cell_size][u[1]] != 0
        else:
            return grid[u[0] - 1][u[1]] != 0


def bfs_get_parents(grid: np.ndarray, start: tuple, end: tuple, cell_size: int) -> np.ndarray:
    """
    Получить массив "родителей" для вершин, находящихся на пути от start к end

    :param grid: сетка (лабиринт)
    :param start: начальная точка
    :param end: конечная точка
    :param cell_size: длина клетки лабиринта
    :return: parents: массив "родителей"
    """
    parents = np.array([np.array([(-1, -1) for _ in range(len(grid[0]))]) for _ in range(len(grid))])
    u = (-1, -1)
    q = Queue()
    q.put(start)
    while not q.empty():
        u = q.get()

        if u[0] == end[0] and u[1] == end[1]:
            break

        # Направления с учётом размера клетки + размер стенки
        direction = [-cell_size - 2, 0, cell_size + 2]
        
        for k in [(i, j) for i in direction for j in direction if not i & j and i != j]:
            v = tuple(sum(x) for x in zip(k, u))
            if 0 <= v[0] < len(grid) and 0 <= v[1] < len(grid[0]):
                if parents[v][0] == -1 and parents[v][1] == -1 and has_path(grid, u, v, cell_size):
                    parents[v] = u
                    q.put(v)
    
    return parents                


def get_way(parents: np.ndarray, start: tuple, end: tuple, offset: int) -> tuple:
    """
    Получить массив координат x, y, описывающих путь из start в end
    
    :param parents: массив "родителей", содержащий информацию о пути из start в end
    :param start: начальная точка
    :param end: конечная точка
    :param offset: отступ от левой верхней точки данной клетки лабиринта
    :return x, y: координаты точек на всех клетках лабиринта, которые находятся в пути из start в end
    """
    x, y = list(), list()
    
    u = end
    
    # Добавляем конечную точку пути
    x.append(end[1])
    y.append(len(parents) - 1)
    
    # Добавляем промежуточные вершины пути
    while not u[0] == start[0] or not u[1] == start[1]:
        x.append(u[1])
        y.append(u[0])
        u = tuple(parents[u])
    x.append(u[1])
    y.append(u[0])
    
    # Добавляем начальную точку пути
    x.append(start[1])
    y.append(0)
    
    x, y = np.array(x), np.array(y)
    
    # Учитываем отступы от левой верхней точки каждой клетки на пути
    x += offset
    y[1:-1] += offset
    
    return x, y


def find_way_from_maze(image: np.ndarray, offset: int = None) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :param offset: отступ от левой верхней точки данной клетки лабиринта 
    :return coords: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
    coords = None
    
    # Ваш код тут
    
    cell_size = get_cell_size(image[:, :, 0])
    start = get_start_point(image[:, :, 0], cell_size)
    end = get_end_point(image[:, :, 0], cell_size)

    if offset is None:
        offset = cell_size//2
    parents = bfs_get_parents(image[:, :, 0], start, end, cell_size)
    coords = get_way(parents, start, end, offset)

    # Ваш код тут
    
    return coords


# Для построения пути вместо plot_maze_path рекомендуется использовать следующую функцию:
def plot_way(x: np.ndarray, y: np.ndarray) -> None:
    """
    Соединить точки (x_i, y_i) в ломаную

    :param x: массив координат на оси x
    :param y: массив координат на оси y
    :return: None, функция встраивает путь в график
    """
    plt.plot(x, y, c='r')

# Порядок вызова: find_way_from_maze -> plot_one_image -> plot_way(*way_coords)
