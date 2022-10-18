import numpy as np

# ширина окна
SIZE = 700
# размер окна
WINDOW_XY = np.array([SIZE, SIZE])
# стиль текста
FONT = ('Helvetica', int(SIZE * 0.0175), 'bold')
# цвет фона
BACKGROUND_COLOR = '#8baabf'
# разамер сетки
BOARD_ZOOM = np.array(WINDOW_XY * 0.75, dtype=int)
# отступ сетки от края
BOARD_OFFSET = np.array([(SIZE - BOARD_ZOOM[0]) / 2, 0], dtype=int)
# размер шашки
ARMY_SIZE = np.array(BOARD_ZOOM * 0.04, dtype=int)
# отступ шашки от края
ARMY_OFFSET = np.array(ARMY_SIZE // 3, dtype=int)


# конвертация координат доски в пиксели
def transform_coords(coords):
    temp = np.array(coords * BOARD_ZOOM * 0.052, dtype=int)
    res = ARMY_OFFSET + BOARD_OFFSET + temp
    return res
