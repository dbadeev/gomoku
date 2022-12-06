from typing import Tuple

import joblib
import numpy as np

from ai_engine import ai_engine
from board.board import Field
from .base_player import Player


class AIPlayer(Player):
    timed = False

    def pregame_init(self, board: Field, heuristic_name='l1'):
        self.engine_idx = ai_engine.start(board, self.color, self.opponent_color, time_limit_for_move=0.5,
                                          heuristic_name=heuristic_name, user_opponent_time=True)

    def get_move(self, current_position: Field) -> Tuple[int, int]:

        current_position_copy = current_position.copy()
        current_position_copy.make_board_readonly()

        graph = ai_engine.get_portal(self.engine_idx).get_graph(current_position_copy, True)
        # print('got graph', str(graph), flush=True)

        successors = list(graph.successors(current_position_copy))
        # print('n successors =', len(successors), flush=True)
        try:
            next_positions, scores, steps_to_end = list(zip(*([(
                x,
                graph.nodes[x]['score'],
                graph.nodes[x]['steps_to_end'],
            ) for x in successors if (graph.nodes[x].get('score', None) is not None)])))

            scores = np.array(scores)
            steps_to_end = np.array(steps_to_end)

            best_score_next_positions_indices = np.argwhere(scores == np.max(scores)).reshape((-1,))
            min_path_idx = np.argmin(steps_to_end[best_score_next_positions_indices])
            best_next_position_idx = best_score_next_positions_indices[min_path_idx]
            best_next_position = next_positions[best_next_position_idx]
        except ValueError:
            joblib.dump(current_position_copy, 'cp_board.joblib')
            best_next_position = successors[np.argmax([graph.nodes[x]['h'] for x in successors])]
        move = graph.edges[current_position_copy, best_next_position]['move']

        ai_engine.get_portal(self.engine_idx).set_my_move(best_next_position)
        return move
