import threading
from time import sleep

import networkx as nx
import numpy as np

from board.board import Field
from heuristics.heuristics import SimpleSumHeuristic


class AIEngine:
    def __init__(self, board_size: int, empty_color: int, color: int, opponent_color: int, timed=False):
        self.G = nx.DiGraph()
        self.h = SimpleSumHeuristic(board_size, color, opponent_color, empty_color)

        self.color = color
        self.opponent_color = opponent_color

        self.timed = timed
        if self.timed:
            self.thread_event = threading.Event()

        self.current_move_idx = 0

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

    def get_possible_moves_from_position(self, position: Field, my_move_idx: int):
        return self.generate_possible_moves_from_position(position, my_move_idx)

    def generate_possible_moves_from_position(self, position: Field, my_move_idx: int):
        check_center_first_n_moves = 4

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
                       (position.has_neighbours(p) or (
                               (my_move_idx < check_center_first_n_moves) and (i < center_square_len)))])

        result = self.sort_possible_moves(result, position)
        if len(result) > 0:
            return result
        else:
            return_n_unfilled = min(center_square_len, len(unfilled))
            return unfilled[:return_n_unfilled]

    def generate_next_moves(self, current_position: Field, is_my_move: bool, go_n_more_layers: int,
                            my_move_idx: int, heuristic_cache=None):
        if (
                result := current_position.is_game_finished()) is not None:  # TODO: это тоже можно заменить на sliding эврситику с кэшем, быстрее будет
            self.G.nodes[current_position]['score'] = np.inf if result == self.color else -np.inf
            self.G.nodes[current_position]['heuristic_cache'] = None
            self.G.nodes[current_position]['steps_to_end'] = 0
            self.G.nodes[current_position]['game_over'] = True
            return

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

        possible_moves = self.get_possible_moves_from_position(current_position, my_move_idx)
        if len(possible_moves) == 0:  # значит не осталось пустых полей
            self.G.nodes[current_position]['score'] = 0
            self.G.nodes[current_position]['heuristic_cache'] = None
            self.G.nodes[current_position]['steps_to_end'] = 0
            return

        successor_positions = [current_position.make_move((m[0], m[1]), inplace=False) for m in possible_moves]

        sorted_by_score_positions = sorted(successor_positions,
                                           key=lambda x: (-1 if is_my_move else 1) * self.G.nodes.get(x, {}).get(
                                               'score', 0))

        for i, next_position in enumerate(sorted_by_score_positions):
            if i > 0 and self.timed and self.thread_event.is_set():
                break

            if not self.G.has_node(next_position):
                self.G.add_node(next_position, move_char=self.opponent_color if is_my_move else self.color)
            if not self.G.has_edge(current_position, next_position):
                self.G.add_edge(current_position, next_position, move=next_position.move_history[-1])

            if self.G.nodes[next_position].get('game_over', False) is not True:
                self.generate_next_moves(next_position, not is_my_move, go_n_more_layers=go_n_more_layers - 1,
                                         heuristic_cache=heuristic_cache, my_move_idx=my_move_idx + 1)
                if go_n_more_layers == 1 and heuristic_cache is None:  # после подсчета эвристики на первой ноде запоминаем ее кэш и передаем в следующий generate_next_moves
                    heuristic_cache = self.G.nodes[next_position]['heuristic_cache']

            self.G.edges[current_position, next_position]['score'] = self.G.nodes[next_position]['score']

            if self.G.nodes[current_position].get('best_outcome', None) is None or \
                    self.is_better_for_me(self.G.nodes[current_position]['best_outcome'],
                                          self.G.nodes[next_position]['score'], is_my_move):

                self.G.nodes[current_position]['best_outcome'] = self.G.nodes[next_position]['score']
                self.G.nodes[current_position]['steps_to_end'] = self.G.nodes[next_position]['steps_to_end'] + 1

                if (my_pred is not None) and (my_pred.get('best_outcome', None) is not None) and \
                        self.is_worse_for_him(my_pred['best_outcome'], self.G.nodes[current_position]['best_outcome'],
                                              is_my_move):
                    break

                if not self.can_get_even_better_for_me(self.G.nodes[current_position]['best_outcome'], is_my_move):
                    break

        self.G.nodes[current_position]['score'] = self.G.nodes[current_position]['best_outcome']

    def set_events_after_n_seconds(self, seconds: float):
        move = self.current_move_idx
        sleep(seconds)
        if move == self.current_move_idx:
            self.thread_event.set()


_engine_instance: AIEngine = None


def start(board: Field, my_color: int, opponent_color: int, timed=False):
    global _engine_instance

    if _engine_instance is None:
        _engine_instance = AIEngine(board.size, board.empty_color, my_color, opponent_color, timed=timed)


def set_current_position(position: Field, my_move_idx: int, move_color: int):
    time_limit_seconds = 5

    if position not in _engine_instance.G:
        print(f'not in G, adding (len={len(_engine_instance.G.nodes)})')
        _engine_instance.G.add_node(position, move_char=move_color)

    nx.set_node_attributes(_engine_instance.G, None, 'best_outcome')

    if _engine_instance.timed:
        threading.Thread(target=_engine_instance.set_events_after_n_seconds(time_limit_seconds)).start()

    _engine_instance.current_move_idx += 1
    _engine_instance.generate_next_moves(position, True, 2, my_move_idx)


def get_graph() -> nx.DiGraph:
    return _engine_instance.G