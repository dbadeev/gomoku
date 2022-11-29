from __future__ import annotations

import numpy as np
from itertools import product

from board.board import Field

from multiprocessing import Pool
from typing import Tuple, Union, Optional


def calculate_score_for_line(line: np.ndarray, my_char, opponent_char) -> np.float:
    power_foundation = 19

    if my_char in line and opponent_char in line:
        return 0

    els, c = np.unique(line, return_counts=True)

    counts = dict(zip(els, c))

    if my_char in els:
        return power_foundation ** counts[my_char]
    if opponent_char in els:
        return -(power_foundation ** counts[opponent_char])

    return 0


class SlidingWindowHeuristicBase:  # отсюда надо просто наследовать разные классы, по-разному реализуя calculate_score_for_line и давать играть друг с другом
    line_len_to_analyze = 5
    power_foundation = 19

    def __init__(self, board_size, my_char, opponent_char, empty_char):
        self.board_size = board_size

        self.my_char = my_char
        self.opponent_char = opponent_char
        self.empty_char = empty_char

        self.window_path_len = self.board_size - self.line_len_to_analyze + 1

        self.horizontal_lines_indices = [np.arange(self.line_len_to_analyze) + i for i in range(self.window_path_len)]
        self.vertical_lines_indices = [np.arange(self.line_len_to_analyze) + i for i in range(self.window_path_len)]

        self.straight_line_indices_0 = [np.arange(self.line_len_to_analyze) + i for i in
                                        range(self.window_path_len)] * self.board_size
        self.straight_line_indices_1 = np.concatenate(
            [np.ones((self.window_path_len,)) * i for i in range(board_size)]).astype(int)[:, np.newaxis]

        diags = [np.arange(self.line_len_to_analyze) + i for i in range(self.window_path_len)]
        inv_diags = [np.flip(np.arange(self.line_len_to_analyze)) + i for i in range(self.window_path_len)]

        self.diag_indices_0, self.diag_indices_1 = [list(x) for x in zip(*list(product(diags, diags + inv_diags)))]

        self.pool = None
        self.line_score_cache = {}
        self.position_score_cache = {}

    def init_pool(self):
        self.pool = Pool(processes=4)

    def destroy_pool(self):
        self.pool.close()
        self.pool = None

    def calculate_score_for_line(self, line: np.ndarray) -> np.float:
        raise NotImplementedError()

    def get_score_for_line(self, line: np.ndarray) -> np.float:
        line_tup = tuple(line)
        if line_tup not in self.line_score_cache:
            self.line_score_cache[line_tup] = self.calculate_score_for_line(line)
        return self.line_score_cache[line_tup]

    def lines_to_scores(self, lines: np.ndarray) -> np.ndarray:
        unique_lines, unique_inverse = np.unique(lines, axis=0, return_inverse=True)

        unique_scores = np.apply_along_axis(self.get_score_for_line, 1, unique_lines).astype(np.float)

        scores_in_original = unique_scores[unique_inverse]

        return scores_in_original

    def calculate_score_for_position(self, board: Field) -> Tuple[np.float, dict]:
        lines = np.concatenate([
            board.board[self.straight_line_indices_0, self.straight_line_indices_1],
            board.board[self.straight_line_indices_1, self.straight_line_indices_0],
            board.board[self.diag_indices_0, self.diag_indices_1],
            board.board[self.diag_indices_1, self.diag_indices_0],
        ])

        score_for_lines = self.lines_to_scores(lines)

        result = score_for_lines.sum()
        return result

    def __call__(self, board: Field) -> Tuple[np.float, dict]:
        return self.calculate_score_for_position(board)


class SimpleSumHeuristic(SlidingWindowHeuristicBase):
    def calculate_score_for_line(self, line: np.ndarray) -> np.float:
        if self.my_char in line and self.opponent_char in line:
            return 0

        els, c = np.unique(line, return_counts=True)
        counts = dict(zip(els, c))

        if self.my_char in els:
            count = counts[self.my_char]
            flag = 1
        elif self.opponent_char in els:
            count = counts[self.opponent_char]
            flag = -1
        else:
            return 0

        return flag * (self.power_foundation ** count)


class LittleMoreSophisticatedHeuristic(SlidingWindowHeuristicBase):
    def calculate_score_for_line(self, line: np.ndarray) -> np.float:
        if self.my_char in line and self.opponent_char in line:
            return 0

        els, c = np.unique(line, return_counts=True)
        counts = dict(zip(els, c))

        if self.my_char in els:
            count = counts[self.my_char]
            flag = 1
        elif self.opponent_char in els:
            count = counts[self.opponent_char]
            flag = -1
        else:
            return 0

        if count == 3 and line[0] == self.empty_char and line[-1] == self.empty_char:
            count = 4

        return flag * (self.power_foundation ** count)


HEURISTICS = {
    'l1': SimpleSumHeuristic,
    'l2': LittleMoreSophisticatedHeuristic
}
