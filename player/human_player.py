from board.board import Field
from .base_player import Player
from typing import Tuple
from messages import Messages as say


class HumanPlayer(Player):

    def get_move(self, position: Field) -> Tuple[int, int] or None:
        """
        Get move from Player in Terminal mode

        :return: Tuple[int, int] or None
            Move coordinates for valid input values, None in other cases
        """
        x, y = -2, -2
        while 1:
            move_line = input(f"Player{self.color} next move: $> ")
            move = move_line.strip().split()
            if move == ["exit"]:
                return None
            try:
                x, y = [int(i) for i in move]
            except Exception as e:
                say(e.__doc__).error_()
                say('x, y - integers; < 20; > 0. Example: next move: $> 2 3'
                    '').error_()
                continue
            if not (0 < x < 20 and 0 < y < 20):
                say('x, y - integers; < 20; > 0. Example: next move: $> 2 3'
                    '').error_()
                continue
            return x - 1, y - 1
        return None