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


def steer(
    parent: np.ndarray,
    sample: np.ndarray,
    step_size: float,
    goal: np.ndarray,
    theta_deg: float,
) -> np.ndarray:
    # convert to radians
    max_theta = np.deg2rad(theta_deg / 2)

    # direction to sample
    v = sample - parent
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return parent

    dir_v = v / norm_v

    # direction to goal
    g = goal - parent
    dir_g = g / np.linalg.norm(g)

    # signed angle between dir_v and dir_g
    dot = np.clip(np.dot(dir_v, dir_g), -1.0, 1.0)
    angle = np.arccos(dot)
    cross = np.cross(np.append(dir_g, 0), np.append(dir_v, 0))[2]
    sign = np.sign(cross)

    # if outside cone, rotate into boundary
    if angle > max_theta:
        angle = max_theta * sign
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        dir_v = R.dot(dir_g)

    # step along dir_v
    dist = np.minimum(step_size, norm_v)
    return parent + dist * dir_v


def original_steer(start: np.ndarray, goal: np.ndarray, step_size: float) -> np.ndarray:
    total_dist = dist_between_points(start, goal)
    if total_dist <= step_size:
        return goal
    # compute vector from start toward goal
    direction = (goal - start) / total_dist
    return start + step_size * direction
