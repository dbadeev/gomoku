import threading
from abc import ABC
from time import sleep
from typing import Tuple

import networkx as nx
import numpy as np

from board import Field
from heuristics import SimpleSumHeuristic
from messages import Messages as say


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

    def __init__(self, color: int, opponent_color: int, board: Field) -> None:
        self.color = color
        self.opponent_color = opponent_color
        self.last_move = (-2, -2)
        # captured stones
        self.captures = 0
        self.five_in_a_row_prev = False

        self.pregame_init(board)

    def get_move(self, position: Field) -> Tuple[int, int] or None:
        pass

    def pregame_init(self, board: Field):
        pass


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


class GomokuAIPlayer(Player):
    timed = False
    def pregame_init(self, board: Field):
        self.G = nx.DiGraph()
        self.h = SimpleSumHeuristic(board.size, self.color, self.opponent_color, board.empty_color)

        self.my_move_idx = 0

        self.possible_positions_cache = {}

        if self.timed:
            self.thread_event = threading.Event()

    @staticmethod
    def can_get_even_better_for_me(current_best, is_my_move):
        if is_my_move:
            return current_best != np.inf
        else:
            return current_best != -np.inf

    @staticmethod
    def is_better_for_me(current_best, new_score, is_my_move):
        if is_my_move:
            return new_score > current_best
        else:
            return new_score < current_best

    @staticmethod
    def is_worse_for_him(pred_best, new_score, is_my_move):
        if is_my_move:
            return new_score > pred_best
        else:
            return new_score < pred_best

    @staticmethod
    def sort_possible_moves(moves, position: Field):
        return sorted(moves, key=lambda x: np.abs(np.array(x) - position.board_half).sum())

    def get_possible_moves_from_position(self, position: Field):
        return self.generate_possible_moves_from_position(position)
        # if position not in self.possible_positions_cache:
        #     self.possible_positions_cache[position] = self.generate_possible_moves_from_position(position)
        # else:
        #     print('cached!!!')
        # return self.possible_positions_cache[position]

    def generate_possible_moves_from_position(self, position: Field):
        result = set()

        my_stones_positions = position.get_player_stones_positions(self.color)
        for pos in my_stones_positions:
            for neig_2 in position.get_all_neighbours_at_distance(pos, 2):
                if not position.is_point_emtpy(neig_2):
                    continue
                result.add(neig_2)

        center_square_len = (position.size % 2 + 1) ** 2
        unfilled = position.get_unfilled_fields()
        result.update([tuple(p) for i, p in enumerate(position.get_unfilled_fields()) if
                       (position.has_neighbours(p) or ((self.my_move_idx < 3) and (i < center_square_len)))])

        result = GomokuAIPlayer.sort_possible_moves(result, position)
        if len(result) > 0:
            return result
        else:
            return_n_unfilled = min(center_square_len, len(unfilled))
            return unfilled[:return_n_unfilled]

    def generate_next_moves(self, current_position: Field, is_my_move: bool, go_n_more_layers: int,
                            heuristic_cache=None):
        if (result := current_position.is_game_finished()) is not None:  # TODO: это тоже можно заменить на sliding эврситику с кэшем, быстрее будет
            self.G.nodes[current_position]['score'] = np.inf if result == self.color else -np.inf
            self.G.nodes[current_position]['heuristic_cache'] = None
            self.G.nodes[current_position]['steps_to_end'] = 0
            self.G.nodes[current_position]['game_over'] = True
            return

        move_char = self.color if is_my_move else self.opponent_color

        try:
            my_pred = self.G.nodes[next(self.G.predecessors(current_position))]
        except StopIteration:
            my_pred = None

        if go_n_more_layers == 0:
            if 'score' not in self.G.nodes[current_position] or 'heuristic_cache' not in self.G.nodes[current_position]:
                value, cached_vals = self.h(current_position, prev_calc_values=heuristic_cache)

                self.G.nodes[current_position]['score'] = value
                self.G.nodes[current_position]['heuristic_cache'] = cached_vals

            self.G.nodes[current_position]['steps_to_end'] = 0
            return

        possible_moves = self.get_possible_moves_from_position(current_position)
        if len(possible_moves) == 0:  # значит не осталось пустых полей
            self.G.nodes[current_position]['score'] = 0
            self.G.nodes[current_position]['heuristic_cache'] = None
            self.G.nodes[current_position]['steps_to_end'] = 0
            return

        successor_positions = [current_position.make_move((m[0], m[1]), inplace=False) for m in possible_moves]

        # tabul = ''.join(["    "] * (-go_n_more_layers + 3))
        # print(f'{tabul}if {self.player_char(not is_my_move)} play {current_position.from_move}')
        sorted_by_score_positions = sorted(successor_positions,
                                           key=lambda x: (-1 if is_my_move else 1) * self.G.nodes.get(x, {}).get(
                                               'score', 0))

        for i, next_position in enumerate(sorted_by_score_positions):
            if i > 0 and self.timed and self.thread_event.is_set():
                break
            # TODO: добавить обработку capture (мб есть у Димы)

            if not self.G.has_node(next_position):
                self.G.add_node(next_position, move_char=self.opponent_color if is_my_move else self.color)
            if not self.G.has_edge(current_position, next_position):
                self.G.add_edge(current_position, next_position, move=next_position.move_history[-1])

            if self.G.nodes[next_position].get('game_over', False) is not True:
                self.generate_next_moves(next_position, not is_my_move, go_n_more_layers=go_n_more_layers - 1,
                                         heuristic_cache=heuristic_cache)
                if go_n_more_layers == 1 and heuristic_cache is None:  # после подсчета эвристики на первой ноде запоминаем ее кэш и передаем в следующий generate_next_moves
                    heuristic_cache = self.G.nodes[next_position]['heuristic_cache']
            # else:
            # print(f'{tabul}game_over if play ({current_position.from_move}) => ({next_position.from_move}), other moves: {[str(p.from_move)  for p in successor_positions]}\n')
            # continue

            self.G.edges[current_position, next_position]['score'] = self.G.nodes[next_position]['score']

            if self.G.nodes[current_position].get('best_outcome', None) is None or \
                    self.is_better_for_me(self.G.nodes[current_position]['best_outcome'],
                                          self.G.nodes[next_position]['score'], is_my_move):

                self.G.nodes[current_position]['best_outcome'] = self.G.nodes[next_position]['score']
                self.G.nodes[current_position]['steps_to_end'] = self.G.nodes[next_position]['steps_to_end'] + 1

                if (my_pred is not None) and (my_pred.get('best_outcome', None) is not None) and \
                        self.is_worse_for_him(my_pred['best_outcome'], self.G.nodes[current_position]['best_outcome'],
                                              is_my_move):
                    # print(f'{tabul}cutoff due to parent ({my_pred["move_char"]}) best = {my_pred["best_outcome"]} and my best = {self.G.nodes[current_position]["best_outcome"]}')
                    break

                if not self.can_get_even_better_for_me(self.G.nodes[current_position]['best_outcome'], is_my_move):
                    break

        self.G.nodes[current_position]['score'] = self.G.nodes[current_position]['best_outcome']
        # print(f'{tabul}than best score is {self.G.nodes[current_position]["score"]}\n')

    def set_events_after_n_seconds(self, seconds: float):
        move = self.my_move_idx
        sleep(seconds)
        if move == self.my_move_idx:
            self.thread_event.set()

    def get_move_with_time_limit(self, current_position: Field, time_limit_seconds: float):
        self.my_move_idx += 1

        if self.timed:
            threading.Thread(target=self.set_events_after_n_seconds(time_limit_seconds)).start()

        print(f'my_move_idx={self.my_move_idx}, len={len(self.G.nodes)}')

        current_position_copy = current_position.copy()
        if current_position_copy not in self.G:
            print(f'not in G, adding (len={len(self.G.nodes)})')
            self.G.add_node(current_position_copy, move_char=self.color)

        nx.set_node_attributes(self.G, None, 'best_outcome')

        # self.h.init_pool()
        self.generate_next_moves(current_position_copy, True, 2)
        # self.h.destroy_pool()

        neigs = self.G.successors(current_position_copy)
        next_positions, scores, steps_to_end = list(zip(*([(
            x,
            self.G.nodes[x]['score'],
            self.G.nodes[x]['steps_to_end'],
        ) for x in neigs])))

        scores = np.array(scores)
        steps_to_end = np.array(steps_to_end)

        best_score_next_positions_indices = np.argwhere(scores == np.max(scores)).reshape((-1,))
        min_path_idx = np.argmin(steps_to_end[best_score_next_positions_indices])
        best_next_position_idx = best_score_next_positions_indices[min_path_idx]
        return self.G.edges[current_position_copy, next_positions[best_next_position_idx]]['move']

    def get_move(self, current_position: Field) -> Tuple[int, int]:
        return self.get_move_with_time_limit(current_position, 5)
