from typing import List, Tuple

from numpy import ndarray

from globals import *
from board import Board
from human import Player


class Rules:
	"""
	Class Rules
	"""

	def __init__(self):
		return

	@staticmethod
	def coordinates(x: int, y: int, dx: int, dy: int, num: int = 5) -> \
			List[Tuple[int, int]]:
		"""
		Coordinates of num intersections in a row, starting from
         (x, y) with reper (dx, dy)

		:param x: int
			x coordinate of initial intersection (x, y) on the board
		:param y: int
			y coordinate of initial intersection (x, y) on the board
		:param dx: int
			dx coordinate of reper (dx, dy)
		:param dy: int
		  dy coordinate of reper (dx, dy)
		:param num: int
			Number of intersections in a row. By default = 5
		:return List[Tuple[int, int]]
			List of coordinates of num intersections in a row, starting from
         (x, y) with reper (dx, dy)
         """
		return [(x + i * dx, y + i * dy) for i in range(num)]

	@staticmethod
	def capture(board: Board, player: Player, remove=True) -> list:
		"""
		Manage the Capture Rule
		Two one-color stones in a row can be "captured" only if they are
		surrounded by stones of the other color.
		Example: 'X O O X', here 'O O' is captured by two X

		:param board: Board
			The current board
		:param player: Player
			The current player
		:param remove: Bool
			Remove stones from board if True
		:return: captures: list
		  List of coordinates for the captured stones
		"""
		coords: ndarray = np.append(REPERS, REPERS * -1, axis=0)
		(x, y), size = player.last_move, board.size
		captures = []
		for dx, dy in coords:
			bounds = x + 3 * dx, y + 3 * dy
			if not (0 <= bounds[0] < size and 0 <= bounds[1] < size):
				continue
			if board.board[bounds[0]][bounds[1]] != player.color:
				continue
			stone1 = ((x + dx), (y + dy))
			stone2 = ((x + dx * 2), (y + dy * 2))
			if not board.is_stone(*stone1, player.color) \
					and not board.is_stone(*stone2, player.color) \
					and not board.is_empty(*stone1) \
					and not board.is_empty(*stone2):
				captures += [stone1, stone2]
		if remove:
			for stone in captures:
				board.remove_stone(*stone)
				player.captures += 1
		return captures

	@staticmethod
	def can_be_captured(board: Board, x: int, y: int, player: Player) -> bool:
		"""
		Checks if opponent's stones can be captured after Player's current
		move (x, y)

		:param x: int
		  x coordinate of player's move (x, y)
		:param y: int
		  y coordinate of player's move (x, y)
		:param board: Board
			The current board
		:param player: Player
			The current player
		:return: succeed: bool
		  True if number of captured stones is more than 0, else False
		"""
		player.last_move = (x, y)
		Rules.capture(board, player, remove=False)
		return len(Rules.capture(board, player, remove=False)) > 0

	@staticmethod
	def limits(coord: list) -> Tuple:
		"""
		Returns the extreme values (minimum-maximum) of the received coordinates

		:param coord: list
		:return:  Returns the coordinates of the lower left and upper right
		vertices of the rectangle for the board intersection coordinates
		received as input
		"""
		x_0, y_0, x_1, y_1 = coord[0][0], coord[0][1], coord[-1][0], coord[-1][
			1]
		return min(x_0, x_1), max(x_0, x_1), min(y_0, y_1), max(y_0, y_1)

	@staticmethod
	def all_equal(coord: list, board_array: ndarray) -> bool:
		"""
		Checks if all the given coordinates of intersections of the board has
		equal color

		:param coord: list
			coordinates of checked intersections of the board
		:param board_array: array
			board of the game with stones on it
		:return:  True if there are stones with same color at the checking
		intersections of the board, False otherwise
		"""
		x_min, x_max, y_min, y_max = Rules.limits(coord)
		if not (x_min >= 0 and
				x_max <= len(board_array) - 1 and
			  	y_min >= 0 and
				y_max <= len(board_array) - 1):
			return False
		colors = [board_array[p[0]][p[1]] for p in coord]
		colors_set = set(colors)
		return len(colors_set) == 1 and not (0 in colors_set)

	@staticmethod
	def five_in_a_row_win(board: Board, player: Player) -> bool:
		"""
		Checks the five_in_a_row Rule:  The player wins if he aligns 5 stone
		of his color

		:param board: Board
			The current board
		:param player: Player
			The current player
		:return: True if five stones in a row have the same color as
		the player who made the current move, False otherwise
		"""
		x, y = player.last_move
		for dx, dy in REPERS:
			for _x, _y in Rules.coordinates(x, y, -dx, -dy, 5):
				if Rules.all_equal(Rules.coordinates(_x, _y, dx, dy, 5),
								  board.board):
					return True
		return False

	@staticmethod
	def five_in_a_row_coords(board: Board, player: Player) -> List:
		"""
		 Finds the coordinates of five intersections in a row after Player's
		 last move

		:param board: Board
			The current board
		:param player: Player
			The current player
		:return: List of five intersections in a row coordinates
		"""
		x, y = player.last_move
		for dx, dy in REPERS:
			for _x, _y in Rules.coordinates(x, y, -dx, -dy, 5):
				coords = Rules.coordinates(_x, _y, dx, dy, 5)
				if Rules.all_equal(coords, board.board):
					return coords
		return []

	@staticmethod
	def check_double_free_threes(board: Board, threes_dict: dict) -> int:
		"""
		Calculates the number of free-three alignments in all of 4
		directions. Each direction gives no more than one free-three for
		calculating. I.e. . X X . x . X X . contains only one free-three
		alignment (x - last move).
		threes_ - all 4-in-a-row in all directions which consists potential
		free-three: (. x x x) or (x x x .) or (x . x x) or (x x . x)

		:param board: Board
			The current board
		:param threes_dict: dict
			coordinates of free-three threes_ in dictionary of directions
		:return: threes: int
			If threes > 1, than free-three alignments exists
		"""
		# threes - общее число обнаруженных свободных троек
		threes = 0
		for lst in threes_dict.values():
			if threes < 2:
				for i in range(len(lst)):
					cnt = lst[i]
					# vec - текущий репер
					vec = cnt[1] - cnt[0]
					# Если в четверке три камня подряд
					if board.is_empty(cnt[0][0], cnt[0][1]) or \
							board.is_empty(cnt[-1][0], cnt[-1][1]):
						# Если поля под и над четверкой в границах и они пустые,
						# то четыре свободных по краям поставить можно
						if board.in_borders(cnt[0][0] - vec[0],
											cnt[0][1] - vec[1]) and \
								board.in_borders(cnt[-1][0] + vec[0],
												 cnt[-1][1] + vec[1]) and \
								board.is_empty(cnt[0][0] - vec[0],
											   cnt[0][1] - vec[1]) and \
								board.is_empty(cnt[-1][0] + vec[0],
											   cnt[-1][1] + vec[1]):
							threes += 1
							# каждое направление может дать не более одной
							# свободной тройки
							# i = len(lst)
							break
						# Если четверка начинается с пустого поля и два поля
						# после четверки в границах и пустые, то четыре
						# свободных по краям поставить можно
						elif board.is_empty(cnt[0][0], cnt[0][1]) and \
								board.in_borders(cnt[-1][0] + 2 * vec[0],
												 cnt[-1][1] + 2 * vec[1]) and \
								board.empty_two_in_a_row(cnt[-1][0] + vec[0],
														 cnt[-1][1] + vec[1],
														 vec[0], vec[1]):
							threes += 1
							# каждое направление может дать не более одной
							# свободной тройки
							# i = len(lst)
							break
						# Если четверка кончается пустым полем и два поля до
						# четверки в границах и пустые, то четыре свободных по
						# краям поставить можно
						elif board.is_empty(cnt[-1][0], cnt[-1][1]) and \
								board.in_borders(cnt[0][0] - 2 * vec[0],
												 cnt[0][1] - 2 * vec[1]) and \
								board.empty_two_in_a_row(cnt[0][0] + vec[0],
														 cnt[0][1] + vec[1],
														 -vec[0], -vec[1]):
							threes += 1
							# каждое направление может дать не более одной
							# свободной тройки
							# i = len(lst)
							break
					# Если в четверке пустое поле не с края
					else:
						# Если поля под и над четверкой в границах и они пустые,
						# то четыре свободных по краям поставить можно
						if board.in_borders(cnt[0][0] - vec[0],
											cnt[0][1] - vec[1]) and \
								board.in_borders(cnt[-1][0] + vec[0],
												 cnt[-1][1] + vec[1]) and \
								board.is_empty(cnt[0][0] - vec[0],
											   cnt[0][1] - vec[1]) and \
								board.is_empty(cnt[-1][0] + vec[0],
											   cnt[-1][1] + vec[1]):
							threes += 1
							# каждое направление может дать не более одной
							# свободной тройки
							break
		return threes

	@staticmethod
	def no_double_free_threes(board: Board, player: Player) -> bool:
		"""
		Manage the No Double Free Threes Rule: It is forbidden to play a move
		that introduces two free-three alignments.
		threes_ - all 4-in-a-row in all repers which consists potential
		free-three: (. x x x) or (x x x .) or (x . x x) or (x x . x)

		:param board: Board
			The current board
		:param player: Player
			The current player
	  	:return: True if Player's last move introduces more than one free-three
	  	alignments
		"""
		(x, y), threes_, size = player.last_move, {}, board.size
		# просмотр всевозможных направлений по каждому реперу
		for idx, (dx, dy) in enumerate(REPERS):
			threes_[idx] = []
			# перемещение шаблона длины 4 вдоль новой точки, куда ставится
			# камень в поиске свободных троек.
			for j in range(-3, 1):
				x_, y_ = (x + j * dx), (y + j * dy)
				if x_ < 0 or x_ >= size or y_ < 0 or y_ >= size or \
						x_ + 3 * dx < 0 or x_ + 3 * dx >= size or \
						y_ + 3 * dy < 0 or y_ + 3 * dy >= size:
					continue
				free = same = 0
				coords_ = Rules.coordinates(x_, y_, dx, dy, 4)
				for k in range(4):
					v, w = coords_[k]
					same += board.is_stone(v, w, player.color)
					free += board.is_empty(v, w)
				if free == 1 and same == 3:
					threes_[idx].append(np.array(coords_))
		return Rules.check_double_free_threes(board, threes_) <= 1

	@staticmethod
	def check_captures(board: Board, player: Player) -> bool:
		"""
		Checks the Player's ability to capture opponent's stones on
		the next move (all possible cases).

		:param board: Board
			The current board
		:param player: Player
			The current player
		  :return: True if Player has possibility of capturing opponent's
		  stones by next move, False otherwise.
		"""
		for x in range(board.size):
			for y in range(board.size):
				if Rules.can_be_captured(board, x, y, player):
					return True
		return False

	@staticmethod
	def can_break_five(board: Board, player: Player, opponent: Player) -> bool:
		"""
		Assuming that there are five in a row, returns True if player can break
    	all the five alignments from his opponent.
    	Checks (all possible cases) the Player's ability to break all the
    	five alignments from his opponent by capturing a pair on the next move.

		:param board: Board
			The current board
		:param player: Player
			The current player
		:param opponent: Player
			The opponent player
	  	:return: True if Player has possibility of capturing opponent's
		  stones by next move, False otherwise.
    	"""
		five_in_a_row = Rules.five_in_a_row_coords(board, opponent)
		for x in range(board.size):
			for y in range(board.size):
				if board.is_empty(x, y):
					player.last_move = (x, y)
					captures = Rules.capture(board, player, remove=False)
					if len(set(captures) & set(five_in_a_row)):
						return True
		return False

	@staticmethod
	def can_reach_ten(board: Board, player: Player) -> bool:
		"""
		Manages the Game-ending capture Rule: A player that manages to align
		five stones only wins if the opponent can not break this alignment by
		capturing a pair, or if he has already lost four pairs and the
		opponent can capture one more, therefore winning by capture.
		Checks that possibility for opponent for winning by capture.

		:param board: Board
			The current board
		:param player: Player
			The opponent player
		  :return: True if opponent Player's last move introduces winning by
		  capture
		"""
		return player.captures >= 8 and Rules.check_captures(board, player)

	@staticmethod
	def check_winner(board: Board, players: list) -> Player or None:
		"""
		Checks if there is a winner in the current position on the board

		:param board: Board
			The current board
		:param players: list
			List of Players
		  :return: True if there is a winner in the current position on the
		  board, False otherwise
		"""
		for player in players:
			if Rules.five_in_a_row_win(board, player) and \
													player.five_in_a_row_prev:
				return player
		for (player, opponent) in [players, players[::-1]]:
			if player.captures >= 10:
				return player

			if Rules.five_in_a_row_win(board, player):
				if (player.five_in_a_row_prev or
						not (Rules.can_reach_ten(board, opponent) or
							 Rules.can_break_five(board, opponent, player))):
					return player
				player.five_in_a_row_prev = True
			else:
				player.five_in_a_row_prev = False
		return None

	# @staticmethod
	# def check_winner_basic(board, players):
	# 	for player in players:
	# 		if Rules.five_in_a_row_win(board, player):
	# 			return player
	# 	return None
