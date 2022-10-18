import numpy as np
from human import Player

# ГЛОБАЛЬНЫЕ КОНСТАНТЫ ###
# направляющие в точке [0, 0] (вниз, по диагонали вверх, вправо, по диагонали
# вниз)
REPERS = np.array([[1, 0], [-1, 1], [0, 1], [1, 1]])

COMPETITORS = {
    "human": Player,
    # "minimax": minimax_agent_wrapper("minimax"),
    # "alpha_beta": minimax_agent_wrapper("alpha_beta"),
    # "alpha_beta_memory": minimax_agent_wrapper("alpha_beta_memory"),
    # "alpha_beta_basic": minimax_agent_wrapper("alpha_beta_basic"),
    # "mtdf": minimax_agent_wrapper("mtdf"),
}

# ГРАФИКА #############
# ширина окна
# SIZE = 700
# # размер окна
# WINDOW_XY = np.array([SIZE, SIZE])
# # стиль текста
# FONT = ('Helvetica', int(SIZE * 0.0175), 'bold')
# # цвет фона
# BACKGROUND_COLOR = '#8baabf'
# # размер сетки
# BOARD_ZOOM = np.array(WINDOW_XY * 0.75, dtype=int)
# # отступ сетки от края
# BOARD_OFFSET = np.array([(SIZE - BOARD_ZOOM[0]) / 2, 0], dtype=int)
# # размер  камня
# STONE_SIZE = np.array(BOARD_ZOOM * 0.04, dtype=int)
# # отступ камня от края
# STONE_OFFSET = np.array(STONE_SIZE // 3, dtype=int)
#
#
# # конвертация координат доски в пиксели
# def transform_coords(coords):
# 	temp = np.array(coords * BOARD_ZOOM * 0.052, dtype=int)
# 	res = STONE_OFFSET + BOARD_OFFSET + temp
# 	return res
