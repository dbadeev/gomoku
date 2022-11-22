from abc import ABC
from typing import Tuple


class Player(ABC):
    """
    Class Player

    :param color: int
      Player's color (1 - 'X' == white; 2 - 'O' == black)

    Attributes
    ----------
    captures: int
      Number of stone captured
    last_move: tuple (x, y)
      Last move
     five_in_a_row_prev: bool
        Five in a row stones on previous move
    """

    def __init__(self, color: int, opponent_color: int, board) -> None:
        self.color = color
        self.opponent_color = opponent_color
        self.last_move = (-2, -2)
        # captured stones
        self.captures = 0
        self.five_in_a_row_prev = False

        self.pregame_init(board)

    def get_move(self, position) -> Tuple[int, int] or None:
        pass

    def pregame_init(self, board):
        pass



