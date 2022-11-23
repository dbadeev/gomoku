from __future__ import annotations

from globals import *
from copy import copy, deepcopy
import numpy as np

# from rules import Rules
from messages import Messages as Say
from typing import List, Tuple, Optional, Iterable
from numpy import ndarray

from player.base_player import Player
from utils import move_point, n_directions

REPERS = np.array([[1, 0], [-1, 1], [0, 1], [1, 1]])


class Field:
    """
    Class Field
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
    empty_color = 0

    def __init__(self, players, filename: str = None, size: int = 19,
                 abbr: dict = None, notation=None, time_limit=np.Inf,
                 board: Optional[np.ndarray] = None) -> None:
        if abbr is None:
            abbr = {0: '.', 1: 'X', 2: 'O'}
        self.abbr = abbr
        if board is None:
            board = np.zeros((size, size))
        self.board = board
        self.filename = filename
        self.size = size
        self.players = players
        self.notation = notation
        # self.time_limit = time_limit
        # if filename:
        #     self.parse(filename)

        self.cnt_player = 0
        self.capture_history = []
        self.move_history = []
        self.state_history = []
        # полуход
        self.turn = 1
        self.winner = None

        self.hash = None
        self.board_half = self.size // 2

    # отрисовка поля - по горизонтали X, по вертикали Y, (1, 1) в левом верхнем
    # углу, добавляются звездочки вместо точек для удобства восприятия в
    # терминальном режиме
    def __str__(self) -> str:

        """
        Formation of the game board for output to the terminal.

        :return term_movie: str
            Game board as a string
        """
        term_movie = f"\033[2J\033[Hᵪ\ʸ"
        for i in range(self.size):
            term_movie += f"{(i + 1):2} "
        term_movie += "\n"
        for i in range(self.size):
            term_movie += f"{(i + 1):3} "
            for j in range(self.size):
                c = self.board[i, j]
                if j in [3, 9, 15] and i in [3, 9, 15] and c == 0:
                    term_movie += f"{'*':>2} "
                else:
                    term_movie += f"{self.abbr[c]:>2} "
            term_movie += "\n"

        term_movie += f"Turn: {self.turn}\n"
        term_movie += f"X: {self.players[0].captures} stone captured\n"
        term_movie += f"O: {self.players[1].captures} stone captured\n"
        return term_movie

    def make_board_readonly(self):
        self.board.flags.writeable = False

    def make_board_writable(self):
        self.board.flags.writeable = True

    def generate_hash(self):
        return hash(self.board.tobytes() + ''.join(map(lambda x: str(x.captures), self.players)).encode('utf-8'))

    def __hash__(self):
        if self.hash is None:
            self.hash = self.generate_hash()
        return self.hash

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False
        return other.__hash__() == self.__hash__()

    def copy(self) -> Field:
        # TODO: реализовать свою deepcopy, чтобы быстро было
        return deepcopy(self)

    def inplace_or_copy_decorator(func):
        def wrapped(self, *args, inplace=True, **kwargs) -> None or Field:
            if inplace is False:
                self_copy = self.copy()
                self_copy.make_board_writable()
                getattr(self_copy, func.__name__)(*args, **kwargs)
                self_copy.make_board_readonly()
                return self_copy
            else:
                func(self, *args, **kwargs)
                self.hash = None

        return wrapped

    @inplace_or_copy_decorator
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
        return self.board[x, y] == 0

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
        return self.board[x, y] == 0 and self.board[x + dx, y + dy] == 0

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
        return self.board[x, y] == color

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
        self.board[x, y] = color
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
        self.board[x, y] = 0
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

    def reset_all(self) -> None:
        """
        Reset all attributes to their initial states
        """
        self.reset_board()
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
                y >= self.board.size or not self.is_empty(x, y)):
            Say("Move Intersection must be empty (. or *)").error_()
            return False

        player.last_move = (x, y)
        self.place_stone(x, y, player.color)

        if Rules.can_be_captured(self, x, y, player):
            self.remove_stone(x, y)
            return True

        if not Rules.no_double_free_threes(self, player):
            self.remove_stone(x, y)
            Say("No double free-threes allowed").error_()
            return False
        self.remove_stone(x, y)
        return True

    @inplace_or_copy_decorator
    def make_move(self, move: tuple):
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
        self.place_stone(*move, player.color)
        player.last_move = tuple(move)
        self.move_history.append(move)

        # updating captures
        captures = Rules.capture(self, player)
        self.capture_history.append(captures)

        self.cnt_player = 1 - self.cnt_player

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
            self.winner = Rules.check_winner(self, self.players)

        # print(self)
        return True

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

            move = player.get_move(self)
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

    @inplace_or_copy_decorator
    def undo_move(self) -> None:
        """
        Update board and game parameter after undoing move
        """
        x, y = self.move_history.pop()
        previous_dead = self.capture_history.pop()
        last_move, captures, five_in_a_row_prev = self.state_history.pop()
        stone = self.board[x, y]
        player = self.players[0 if stone == 1 else 1]
        opponent = self.players[1 if stone == 1 else 0]
        self.remove_stone(x, y)

        for x, y in previous_dead:
            self.place_stone(x, y, opponent.color)

        player.last_move = last_move
        player.captures = captures
        player.five_in_a_row_prev = five_in_a_row_prev
        self.cnt_player = 1 - self.cnt_player

    # Методы для AI игрока
    def get_unfilled_fields(self) -> List[Tuple[int, int]]:
        return np.argwhere(self.board == self.empty_color)

    def is_point_on_board(self, p: Tuple[int, int]):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    def is_point_emtpy(self, p: Tuple[int, int]) -> bool:
        return self.board[p[0], p[1]] == self.empty_color

    def is_game_finished(self) -> Optional[int]:
        for player_val in [val for val, char in self.abbr.items() if val != self.empty_color]:
            player_stones_indices = np.argwhere(self.board == player_val)

            for stone_idx in player_stones_indices:
                if self.is_line_of_n_more(stone_idx, 4, directions=range(2, 6)):
                    return player_val
        return None

    def is_pair(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        return self.board[p1[0], p1[1]] == self.board[p2[0], p2[1]]

    def has_neighbours(self, p: Tuple[int, int], mitigate_borders=True) -> bool:
        for d in range(n_directions):
            new_p = move_point(p, d)
            if not self.is_point_on_board(new_p):
                if mitigate_borders:
                    continue
                else:
                    return True
            if not self.is_point_emtpy(new_p):
                return True
        return False

    def get_player_stones_positions(self, player_char: int) -> np.ndarray:
        if player_char not in self.abbr:
            return np.empty((0,))
        return np.argwhere(self.board == player_char)

    def get_all_neighbours_at_distance(self, p: Tuple[int, int], dist: int) -> list[Tuple[int, int]]:
        return [n for dire in range(n_directions) if self.is_point_on_board(n := move_point(p, dire, dist))]

    def is_line_of_n_more(self, p: Tuple[int, int], n_more: int, direction: int = None,
                          directions: Iterable[int] = None):
        if direction is not None:
            neig = move_point(p, direction)
            if not self.is_point_on_board(neig):
                return False
            if not self.is_pair(p, neig):
                return False
            elif n_more > 1:
                return self.is_line_of_n_more(neig, n_more - 1, direction=direction)
            else:
                return True
        else:
            if directions is None:
                directions = range(8)
            for d in directions:
                if self.is_line_of_n_more(p, n_more, direction=d):
                    return True
            return False

    def is_line_of_n(self, p: Tuple[int, int], n: int, direction: int = None, unbounded_required=False,
                     empty_5_required=False):
        if self.is_point_emtpy(p):
            return False
        if n == 1:
            return not self.has_neighbours(p, mitigate_borders=False)

        if direction is None:
            for d in range(2, 6):
                if self.is_line_of_n(p, n, direction=d):
                    return True
            return False
        else:
            pre_bound = move_point(p, (direction + n_directions // 2) % n_directions)
            post_bound = move_point(p, direction, n)

            if unbounded_required and not (self.is_point_on_board(pre_bound) and
                                           self.is_point_on_board(post_bound) and
                                           self.is_point_emtpy(pre_bound) and
                                           self.is_point_emtpy(post_bound)):
                return False
            return self.is_line_of_n_more(p, n - 1, direction=direction)


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
    def capture(board: Field, player: Player, remove=True) -> list:
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
    def can_be_captured(board: Field, x: int, y: int, player: Player) -> bool:
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
    def five_in_a_row_win(board: Field, player: Player) -> bool:
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
    def five_in_a_row_coords(board: Field, player: Player) -> List:
        """
         Finds the coordinates of five intersections in a row after Player's
         last move

        :param board: Field
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
    def check_double_free_threes(board: Field, threes_dict: dict) -> int:
        """
        Calculates the number of free-three alignments in all of 4
        directions. Each direction gives no more than one free-three for
        calculating. I.e. . X X . x . X X . contains only one free-three
        alignment (x - last move).
        threes_ - all 4-in-a-row in all directions which consists potential
        free-three: (. x x x) or (x x x .) or (x . x x) or (x x . x)

        :param board: Field
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
    def no_double_free_threes(board: Field, player: Player) -> bool:
        """
        Manage the No Double Free Threes Rule: It is forbidden to play a move
        that introduces two free-three alignments.
        threes_ - all 4-in-a-row in all repers which consists potential
        free-three: (. x x x) or (x x x .) or (x . x x) or (x x . x)

        :param board: Field
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
    def check_captures(board: Field, player: Player) -> bool:
        """
        Checks the Player's ability to capture opponent's stones on
        the next move (all possible cases).

        :param board: Field
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
    def can_break_five(board: Field, player: Player, opponent: Player) -> bool:
        """
        Assuming that there are five in a row, returns True if player can break
        all the five alignments from his opponent.
        Checks (all possible cases) the Player's ability to break all the
        five alignments from his opponent by capturing a pair on the next move.

        :param board: Field
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
    def can_reach_ten(board: Field, player: Player) -> bool:
        """
        Manages the Game-ending capture Rule: A player that manages to align
        five stones only wins if the opponent can not break this alignment by
        capturing a pair, or if he has already lost four pairs and the
        opponent can capture one more, therefore winning by capture.
        Checks that possibility for opponent for winning by capture.

        :param board: Field
            The current board
        :param player: Player
            The opponent player
          :return: True if opponent Player's last move introduces winning by
          capture
        """
        return player.captures >= 8 and Rules.check_captures(board, player)

    @staticmethod
    def check_winner(board: Field, players: list) -> Player or None:
        """
        Checks if there is a winner in the current position on the board

        :param board: Field
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
