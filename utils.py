from typing import Tuple

n_directions = 8


def move_point(p: Tuple[int, int], direction: int, steps=1) -> Tuple[int, int]:
    if direction == 0:
        return p[0] - steps, p[1]
    elif direction == 1:
        return p[0] - steps, p[1] + steps
    elif direction == 2:
        return p[0], p[1] + steps
    elif direction == 3:
        return p[0] + steps, p[1] + steps
    elif direction == 4:
        return p[0] + steps, p[1]
    elif direction == 5:
        return p[0] + steps, p[1] - steps
    elif direction == 6:
        return p[0], p[1] - steps
    elif direction == 7:
        return p[0] - steps, p[1] - steps


class EmptyException(Exception):
    pass
