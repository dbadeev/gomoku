import multiprocessing
from datetime import datetime
from multiprocessing import Process
from time import sleep

import networkx as nx
import numpy as np

from board.board import Field
from heuristics.heuristics import HEURISTICS

calculation_depth = 2


class AIEngine:
    def __init__(self, board_size: int, empty_color: int, color: int, opponent_color: int, timed=False,
                 heuristic_name='l1', stop_flag: multiprocessing.Value = None):
        self.G = nx.DiGraph()

        if heuristic_name in HEURISTICS:
            self.h = HEURISTICS[heuristic_name](board_size, color, opponent_color, empty_color)
        else:
            raise ValueError(f'unknown heuristic name: {heuristic_name}')

        self.color = color
        self.opponent_color = opponent_color

        self.timed = timed
        if self.timed:
            self.stop_flag = stop_flag

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
    def is_new_best_worse_for_parent_node(pred_best, new_score, is_my_move):
        if is_my_move:
            return new_score > pred_best
        else:
            return new_score < pred_best

    @staticmethod
    def get_l1_distance_from_board_center(board: Field, p: tuple[int, int]):
        return np.abs(np.array(p) - board.board_half).sum()

    @staticmethod
    def sort_possible_moves(moves, position: Field):
        return sorted(moves, key=lambda x: AIEngine.get_l1_distance_from_board_center(position, x))

    def find_or_calc_position_h(self, position: Field, calc_if_not_found=False, placeholder=0) -> float:
        if self.G.has_node(position) and isinstance(self.G.nodes[position].get('h', None), float):
            return self.G.nodes[position]['h']
        elif calc_if_not_found:
            if not self.G.has_node(position):
                self.G.add_node(position)
            h = self.h(position)
            self.G.nodes[position]['h'] = h
            return h

        else:
            return placeholder

    def get_possible_moves_from_position(self, position: Field):
        return self.generate_possible_moves_from_position(position)

    def generate_possible_moves_from_position(self, position: Field) -> list[tuple[int, int]]:
        check_center_first_n_moves = 2

        possible_moves_bin = set()

        my_stones_positions = position.get_player_stones_positions(self.color)
        for pos in my_stones_positions:
            for neig_2 in position.get_all_neighbours_at_distance(pos, 2):
                if not position.is_point_emtpy(neig_2):
                    continue
                possible_moves_bin.add(neig_2)

        my_move_idx = len(my_stones_positions)

        center_square_radius = 3 if position.size % 2 else 2
        center_square_len = center_square_radius ** 2

        possible_moves_bin.update([tuple(p) for i, p in enumerate(position.get_unfilled_fields()) if
                                   (position.has_neighbours(p) or (
                                           (my_move_idx < check_center_first_n_moves) and
                                           self.get_l1_distance_from_board_center(position,
                                                                                  p) < center_square_radius))])
        unfilled_sorted_from_center = self.sort_possible_moves(possible_moves_bin, position)

        if len(unfilled_sorted_from_center) > 0:
            return unfilled_sorted_from_center
        else:
            return_n_unfilled = min(center_square_len, len(unfilled_sorted_from_center))
            return unfilled_sorted_from_center[:return_n_unfilled]

    def generate_next_moves(self, current_position: Field, is_my_move: bool, go_n_more_layers: int):
        if self.G.has_node(current_position) and \
                self.G.nodes[current_position].get('processed_depth', -1) >= go_n_more_layers:
            return True

        if current_position.winner is not None:
            self.G.nodes[current_position]['score'] = np.inf if current_position.winner.color == self.color else -np.inf
            self.G.nodes[current_position]['steps_to_end'] = 0
            self.G.nodes[current_position]['processed_depth'] = 0
            self.G.nodes[current_position]['game_over'] = True
            return True

        if go_n_more_layers == 0:
            value = self.find_or_calc_position_h(current_position, calc_if_not_found=True)
            self.G.nodes[current_position]['score'] = value
            self.G.nodes[current_position]['h'] = value
            self.G.nodes[current_position]['steps_to_end'] = 0
            self.G.nodes[current_position]['processed_depth'] = 0
            return True

        if self.G.has_node(current_position) and self.G.nodes[current_position].get('processed_depth', -1) > 0 and \
                self.G.nodes[current_position].get('cutoff_performed', False) is False:
            successor_positions = self.G.successors(current_position)
        else:
            possible_moves = self.get_possible_moves_from_position(current_position)
            if len(possible_moves) == 0:  # значит не осталось пустых полей
                print('no more moves!!', flush=True)
                self.G.nodes[current_position]['score'] = 0
                self.G.nodes[current_position]['h'] = 0
                self.G.nodes[current_position]['steps_to_end'] = 0
                self.G.nodes[current_position]['processed_depth'] = 0
                return True

            # Если прошли дальше - вершина не терминальная

            present_successors = self.G.successors(current_position)
            present_successors_moves = set([p.last_move() for p in present_successors])

            successor_positions = []
            for move in possible_moves:
                possible_position = current_position.copy()
                possible_position.make_board_writable()

                if move in present_successors_moves:
                    possible_position.make_move(move)
                    successor_positions.append(possible_position)
                elif possible_position.place_on_board(move):
                    successor_positions.append(possible_position)
                possible_position.make_board_readonly()
                possible_position.hash = None

        sorted_by_score_positions = sorted(
            successor_positions,
            key=lambda x: (-1 if is_my_move else 1) * self.find_or_calc_position_h(
                x,
                calc_if_not_found=go_n_more_layers >= calculation_depth,
                placeholder=(-np.inf if is_my_move else np.inf)
            ))

        if len(sorted_by_score_positions) == 0:
            raise ValueError(f'sorted_by_score_positions len = 0')

        try:
            my_pred = self.G.nodes[next(self.G.predecessors(current_position))]
        except StopIteration:
            my_pred = None

        for i, next_position in enumerate(sorted_by_score_positions):
            if i > 0 and self.stop_flag.value == 1:
                # print(f'exiting {go_n_more_layers}, {self.stop_flag.value}', flush=True)
                return False

            if not self.G.has_node(next_position):
                self.G.add_node(next_position)
            if not self.G.has_edge(current_position, next_position):
                self.G.add_edge(current_position, next_position, move=next_position.last_move())

            if not self.generate_next_moves(next_position, not is_my_move, go_n_more_layers=go_n_more_layers - 1):
                return False

            self.G.edges[current_position, next_position]['score'] = self.G.nodes[next_position]['score']

            if self.G.nodes[current_position].get('best_outcome', None) is None or \
                    self.is_better_for_me(self.G.nodes[current_position]['best_outcome'],
                                          self.G.nodes[next_position]['score'], is_my_move):

                self.G.nodes[current_position]['best_outcome'] = self.G.nodes[next_position]['score']
                self.G.nodes[current_position]['steps_to_end'] = self.G.nodes[next_position]['steps_to_end'] + 1

                if (my_pred is not None) and (my_pred.get('best_outcome', None) is not None) and \
                        self.is_new_best_worse_for_parent_node(my_pred['best_outcome'],
                                                               self.G.nodes[current_position]['best_outcome'],
                                                               is_my_move):
                    self.G.nodes[current_position]['cutoff_performed'] = True
                    break

                if not self.can_get_even_better_for_me(self.G.nodes[current_position]['best_outcome'], is_my_move):
                    self.G.nodes[current_position]['cutoff_performed'] = True
                    break

        self.G.nodes[current_position]['score'] = self.G.nodes[current_position]['best_outcome']
        self.G.nodes[current_position]['processed_depth'] = go_n_more_layers
        return True


class EnginePortal:
    def __init__(self, board: Field, my_color: int, opponent_color: int, heuristic_name='l1', user_opponent_time=False,
                 time_limit_for_move=None):

        self.user_opponent_time = user_opponent_time
        self.time_limit_for_move = time_limit_for_move

        self.out, self.inn = multiprocessing.Pipe(True)

        self.flag = multiprocessing.Value('b', 0)

        self.process = Process(target=self.start_generating_moves,
                               args=(
                                   self.flag, board.size, board.empty_color, my_color, opponent_color, heuristic_name))
        self.process.start()

    def start_generating_moves(self, flag, board_size, empty_color, my_color, opponent_color, heuristic_name):
        engine = AIEngine(board_size, empty_color, my_color, opponent_color, heuristic_name=heuristic_name,
                          stop_flag=flag, timed=True)

        while True:
            flag.value = 0
            position, is_my_move, send_result, search_depth = self.inn.recv()

            nx.set_node_attributes(engine.G, None, 'best_outcome')

            if not engine.G.has_node(position):
                engine.G.add_node(position)

            engine.generate_next_moves(position, is_my_move, search_depth)

            if send_result:
                self.inn.send(nx.DiGraph(engine.G.subgraph([position] + list(engine.G.successors(position)))))

    def get_graph(self, position: Field, is_my_move: bool) -> nx.DiGraph:
        if self.user_opponent_time:
            self.flag.value = 1  # останавливаем прошлый обсчет

        self.out.send((position, is_my_move, True, calculation_depth))  # кладем текущую позицию на обсчет

        if self.time_limit_for_move is not None:
            sleep(self.time_limit_for_move)  # ждем обсчет
            self.flag.value = 1  # останавливаем текущий обсчет

        graph: nx.DiGraph = self.out.recv()  # получаем обсчитанный граф

        return graph

    def set_my_move(self, position: Field):
        if self.user_opponent_time:
            self.out.send(
                (position, False, False, calculation_depth + 1))  # кладем мой ход на обсчет того, что сделает противник


_portals: dict[int, EnginePortal] = {}


def get_portal(portal_idx) -> EnginePortal:
    global _portals
    return _portals[portal_idx]


def start(board: Field, my_color: int, opponent_color: int, time_limit_for_move=None, heuristic_name='l1',
          user_opponent_time=False) -> int:
    global _portals

    new_portal_idx = len(_portals) + 1
    _portals[new_portal_idx] = EnginePortal(board, my_color, opponent_color, time_limit_for_move=time_limit_for_move,
                                            heuristic_name=heuristic_name, user_opponent_time=user_opponent_time)
    return new_portal_idx


def terminate_processes():
    global _portals

    for idx in _portals.keys():
        _portals[idx].process.terminate()

    print('all processes terminated')
