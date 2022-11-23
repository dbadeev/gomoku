from typing import Tuple

import numpy as np

from ai_engine import ai_engine
from board.board import Field
from .base_player import Player


class GomokuAIPlayer(Player):
    timed = False

    def pregame_init(self, board: Field):
        self.my_move_idx = 0

        self.engine_idx = ai_engine.start(board, self.color, self.opponent_color, timed=False)

    def get_move(self, current_position: Field) -> Tuple[int, int]:
        self.my_move_idx += 1

        current_position_copy = current_position.copy()
        current_position_copy.make_board_readonly()

        ai_engine.set_current_position(self.engine_idx, current_position_copy, self.my_move_idx, self.color)

        graph = ai_engine.get_graph(self.engine_idx)
        neigs = graph.successors(current_position_copy)
        next_positions, scores, steps_to_end = list(zip(*([(
            x,
            graph.nodes[x]['score'],
            graph.nodes[x]['steps_to_end'],
        ) for x in neigs])))

        scores = np.array(scores)
        steps_to_end = np.array(steps_to_end)

        best_score_next_positions_indices = np.argwhere(scores == np.max(scores)).reshape((-1,))
        min_path_idx = np.argmin(steps_to_end[best_score_next_positions_indices])
        best_next_position_idx = best_score_next_positions_indices[min_path_idx]
        return graph.edges[current_position_copy, next_positions[best_next_position_idx]]['move']
