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


def steer(start: np.ndarray, goal: np.ndarray, step_size: float):
    """
    Return a point in the direction of the goal, that is distance away from start
    :param start: start location
    :param goal: goal location
    :param step_size: step size away from start
    :return: point in the direction of the goal
    """
    
    total_dist = dist_between_points(start, goal)
    
    if total_dist <= step_size:
        return goal
    
    x_dist = np.abs(start[0] - goal[0])
    y_dist = np.abs(start[1] - goal[1])
    
    step_x = step_size * x_dist / total_dist
    step_y = step_size * y_dist / total_dist
       
    return np.array([start[0] + step_x, start[1] + step_y])