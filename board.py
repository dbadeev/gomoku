import numpy as np
from typing import Tuple


class Board:
	"""
	Class Board
    board: 2D array
            Board of the game
	:param filename: str (Default: None)
		Load board from file if specified (сделать варианты восстановления
		доски по ходам и по состоянию доски)
	:param size: int (Default: 19)
		Size of the board side
	:param abbr: dict (Default: {0: '.', 1: 'X', 2: 'O'})
		Notation of board intersections ('.' - empty, 'X' - white stone,
		'O' - black stone)
	"""

	def __init__(self, filename: str = None, size: int = 19,
													abbr: dict = None) -> None:
		if abbr is None:
			abbr = {0: '.', 1: 'X', 2: 'O'}
		self.abbr = abbr
		self.board = np.zeros((size, size))
		self.filename = filename
		self.size = size
		# if filename:
		#     self.parse(filename)

	# отрисовка поля - по горизонтали X, по вертикали Y, (1, 1) в левом верхнем
	# углу, добавляются звездочки вместо точек для удобства восприятия в
	# терминальном режиме
	def __str__(self) -> str:

		"""
		Formation of the game board for output to the terminal.

		:return term_movie: str
			Game board as a string
		"""
		term_movie = f"\uff59/\u02E3"
		for i in range(self.size):
			term_movie += f"{(i + 1):2} "
		term_movie += "\n"
		for i in range(self.size):
			term_movie += f"{(i + 1):3} "
			for j in range(self.size):
				c = self.board[i][j]
				if j in [3, 9, 15] and i in [3, 9, 15] and c == 0:
					term_movie += f"{'*':>2} "
				else:
					term_movie += f"{self.abbr[c]:>2} "
			term_movie += "\n"
		return term_movie

	def reset_board(self) -> None:
		"""
		Reset game board to it's initial state
		"""
		self.board = np.zeros((self.size, self.size))
		# if self.filename:
		#     self.parse(self.filename)

	def is_empty(self, x: int, y: int) -> bool:
		"""
		Check for free intersection (x, y) of the board

		:param x: int
		  x coordinate of intersection (x, y) on the board
		:param y: int
		  y coordinate of intersection (x, y) on the board
		:return True if intersection (x, y) is empty, False otherwise
		"""
		return self.board[x][y] == 0

	def empty_two_in_a_row(self, x: int, y: int, dx: int, dy: int) -> bool:
		"""
		Two intersections in a row on the board in direction (dx, dy)
		from (x, y) - check

		:param x: int
			x coordinate of initial intersection (x, y) on the board
		:param y: int
			y coordinate of initial intersection (x, y) on the board
		:param dx: int
			dx coordinate of reper (dx, dy)
		:param dy: int
		  dy coordinate of reper (dx, dy)
		:return True if two intersections in a raw of the board in
		the direction (dx, dy) from (x, y) are empty
		"""
		return self.board[x][y] == 0 and self.board[x + dx][y + dy] == 0

	def in_borders(self, x: int, y: int) -> bool:
		"""
		(x, y) are coordinates of intersection in board - validation

		:param x: int
		  x coordinate of intersection (x, y) on the board
		:param y: int
		  y coordinate of intersection (x, y) on the board
		:return: True if (x, y) in board, else False
		"""
		if x < 0 or x >= self.size or y < 0 or y >= self.size:
			return False
		else:
			return True

	def is_stone(self, x: int, y: int, color: int) -> bool:
		"""
		Stone of Player(color) is places at (x, y) - validation

		:param x: int
			x coordinate of intersection (x, y) on the board
		:param y: int
			y coordinate of intersection (x, y) on the board
		:param color: int
			Player's color (1 - 'X' == white; 2 - 'O' == black)
		:return: True if color stone is at (x, y) in board, else False
		"""
		return self.board[x][y] == color

	def place_stone(self, x: int, y: int, color: int) -> None:
		"""
		Places the Player's color stone at intersection (x, y)

		:param x: int
			x coordinate of intersection (x, y) on the board
		:param y: int
			y coordinate of intersection (x, y) on the board
		:param color: int
			Player's color (1 - 'X' == white; 2 - 'O' == black)
		"""
		self.board[x][y] = color
		return

	def remove_stone(self, x: int, y: int) -> None:
		"""
		Removes (put 0 - '.' instead) the Player's stone from intersection (
		x, y)

		:param x: int
			x coordinate of intersection (x, y) on the board
		:param y: int
			y coordinate of intersection (x, y) on the board
		"""
		self.board[x][y] = 0
		return

	def is_board_empty(self) -> bool:
		"""
		Checks if the board is empty

		:return: True, if no stones at the board, else False
		"""
		return np.sum(self.board) == 0

	# def board_center(self) -> Tuple[int, int]:
	# 	"""
	# 	Returns the coordinates of the center of the board
	# 	"""
	# 	return self.size // 2, self.size // 2

	# def is_board_full(self):
	# 	"""
	# 	Returns True if the entire board is full
	# 	"""
	# 	return not (0 in self.board)

	# парсинг данных из файла - на потом
	# def parse(self, file):
	#     try:
	#         inv_abbr = {v: k for k, v in self.abbr.items()}
	#         inv_abbr['*'] = 0
	#         if file == "custom":
	#             raw = [line.rstrip().split() for line in sys.stdin]
	#         else:
	#             f = open(file, 'r')
	#             raw = [line.strip().split() for line in f]
	#         board = [l[1:] for l in raw[1:]]
	#         for i in range(self.size):
	#             for j in range(self.size):
	#                 self.board[i][j] = inv_abbr[board[i][j]]
	#     except Exception:
	#         print("Error encountered while parsing board, starting with \
    #                                                           empty board.")
	#         self.board = np.zeros((self.size, self.size))
