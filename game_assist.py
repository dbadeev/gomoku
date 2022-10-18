from typing import List, Any

from globals import *
from messages import Messages as Say
from rules import Rules


class GameAssist:
	"""
	Class GameAssist

	:param board: Board
		The board
	:param players: list
		List of Players - Human OR AI
	:param notation: Notation (default: None)
		moves notation of the game
	:param size: int (default: 19)
		Size of the board
	# :param time_limit: float (default: inf)
	# 	Ограничение по времени для расчета хода (для  AI)

	Attributes
	 ----------
	cnt_player: int
		Current player to make a move
	capture_history: list
		List of coordinates for captured stones on each move
	move_history: list
		List of coordinates for done moves
	state_history: list
		List of move, captures, five_in_a_row_prev for each move
	turn: int
		Current In-game turn to play
	winner: Player
		Player who won
	# help_AI: AI
		# AI's hint for the next move
	# start_time_AI: float
		# Time when AI timer switches ON
	"""

	def __init__(self, board, players, notation=None, size=19, time_limit=np.Inf):
		self.board = board
		self.players = players
		self.notation = notation
		self.size = size
		# self.time_limit = time_limit

		self.cnt_player = 0
		self.capture_history = []
		self.move_history = []
		self.state_history = []
		# полуход
		self.turn = 1
		self.winner = None

	def reset_all(self) -> None:
		"""
		Reset all attributes to their initial states
		"""
		self.board.reset_board()
		for player in self.players:
			player.captures = 0
			player.last_move = (-2, -2)
			player.five_in_a_row_prev = False
			if hasattr(player, 'undo_scores'):
				player.reset()
		# if self.notation:
		#     self.notation.reset_board()

		self.cnt_player = 0
		self.capture_history = []
		self.move_history = []
		self.state_history = []
		self.turn = 1
		self.winner = None
		# self.start_time_AI = -1
		# self.child_list = []
		return

	def start_terminal(self) -> None:
		"""
		Main Function for Terminal mode, run the game
		"""
		print(self)
		while 1:
			player = self.players[self.cnt_player]

			# # self.start_time_AI = time.time()
			# if isinstance(player, AI):
			#     move = player.find_move(self)
			# elif not self.notation or not self.notation.running():
			#     move = player.get_move()
			# else:
			#     move = self.notation.get_move()
			#     print(f"Player {player.color}: {move[0] + 1, move[1] + 1}")

			move = player.get_move()
			if len(move) == 0:
				return None
			self.place_on_board(move)
			if self.winner:
				print(self)
				print(f"P{self.winner.color} won.")
				return None

			# if self.notation and self.notation.running():
			# 	time.sleep(0.25)
		return None

	def place_on_board(self, move: tuple) -> bool:
		"""
		Try to place stone with move coordinates on board

		:param move: tuple, 2d-array
		  The intersection (x, y) on board where we want to put stone

		:return: succeed: bool
		  True if possible to place stone at the intersection (x, y), else False
		"""

		if not self.can_place_on_board(*move):
			return False
		self.make_move(move)

		self.turn += 1

		if self.winner is None:
			self.winner = Rules.check_winner(self.board, self.players)

		print(self)
		return True

	# def move_hint(self):
	# 	"""Return the best move (from AI point of view) for the player"""
	# 	return self.help_AI.find_move(self)[::-1]

	def can_place_on_board(self, x: int, y: int) -> bool:
		"""
		Check if Current Player can place stone at the intersection (x, y)

		:param x: int
		  x coordinate of intersection (x, y) on the board
		:param y: int
		  y coordinate of intersection (x, y) on the board
		:return: succeed: bool
		  True if we can place the stone else False
		"""
		player = self.players[self.cnt_player]
		if (x < 0 or x >= self.board.size or y < 0 or
				y >= self.board.size or not self.board.is_empty(x, y)):
			Say("Move Intersection must be empty (. or *)").error_()
			return False

		player.last_move = (x, y)
		self.board.place_stone(x, y, player.color)

		if Rules.can_be_captured(self.board, x, y, player):
			self.board.remove_stone(x, y)
			return True

		if not Rules.no_double_free_threes(self.board, player):
			self.board.remove_stone(x, y)
			Say("No double free-threes allowed").error_()
			return False
		self.board.remove_stone(x, y)
		return True

	def make_move(self, move: tuple) -> bool:
		"""
		Place stone with move coordinates on board

		:param move: tuple, 2d-array
		  The intersection (x, y) on board where we want to put stone

		:return: succeed: bool
		  True
		"""
		player = self.players[self.cnt_player]
		self.state_history.append([player.last_move,
								   player.captures,
								   player.five_in_a_row_prev])
		self.board.place_stone(*move, player.color)
		player.last_move = tuple(move)
		self.move_history.append(move)

		# updating captures
		captures = Rules.capture(self.board, player)
		self.capture_history.append(captures)

		self.cnt_player = 1 - self.cnt_player
		return True

	def basic_move(self, move: tuple) -> None:
		"""
		Place stone with move coordinates on board + update game parameter

		:param move: tuple, 2d-array
			The intersection (x, y) on board where we want to put stone
		"""
		player = self.players[self.cnt_player]
		self.move_history.append(move)
		self.state_history.append([player.last_move,
								   player.captures,
								   player.five_in_a_row_prev])
		self.board.place(*move, player.color)
		player.last_move = tuple(move)

	def undo_move(self) -> None:
		"""
		Update board and game parameter after undoing move
		"""
		x, y = self.move_history.pop()
		previous_dead = self.capture_history.pop()
		last_move, captures, five_in_a_row_prev = self.state_history.pop()
		stone = self.board.board[x][y]
		player = self.players[0 if stone == 1 else 1]
		opponent = self.players[1 if stone == 1 else 0]
		self.board.remove_stone(x, y)

		for x, y in previous_dead:
			self.board.place_stone(x, y, opponent.color)

		player.last_move = last_move
		player.captures = captures
		player.five_in_a_row_prev = five_in_a_row_prev
		self.cnt_player = 1 - self.cnt_player

	def last_move(self) -> tuple:
		"""
		Last move of the game

		:return: last move if exists or (-1, -1)
		"""
		return (-1, -1) if not self.move_history else self.move_history[-1]

	def get_player_captures(self) -> np.array:
		"""
		Captures for each player

		:return: Array of captures for each player
		"""
		return np.array([player.captures for player in self.players])

	def __str__(self) -> str:
		"""
		Formation of the game board for output to the terminal

		:return: term: str
			Game board as a string
		"""
		# player, opponent = (self.players[self.cnt_player],
		#                     self.players[1 - self.cnt_player])

		term_movie = f"\033[2J\033[H{self.board}"
		term_movie += f"Turn: {self.turn}\n"
		term_movie += f"X: {self.players[0].captures} stone captured\n"
		term_movie += f"O: {self.players[1].captures} stone captured\n"
		return term_movie
