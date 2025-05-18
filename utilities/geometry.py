import numpy as np


def dist_between_points(a, b):
    """
    Return the Euclidean distance between two points
    :param a: first point
    :param b: second point
    :return: Euclidean distance between a and b
    """
    distance = np.linalg.norm(np.array(b) - np.array(a))
    return distance


def steer(start: np.ndarray, goal: np.ndarray, step_size: float) -> np.ndarray:
    total_dist = dist_between_points(start, goal)
    if total_dist <= step_size:
        return goal
    # compute vector from start toward goal
    direction = (goal - start) / total_dist
    return start + step_size * direction
