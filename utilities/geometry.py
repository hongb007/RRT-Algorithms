import numpy as np


def dist_between_points(a, b):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
        a (array-like): Coordinates of the first point.
        b (array-like): Coordinates of the second point.

    Returns:
        float: Euclidean distance between points a and b.
    """
    distance = np.linalg.norm(np.subtract(b, a))
    return distance

def steer(
    parent: np.ndarray,
    sample: np.ndarray,
    step_size: float,
    goal: np.ndarray,
    theta_deg: float,
    turn_chance: float,
) -> np.ndarray:
    """
    Steer from `parent` toward `sample` by at most `step_size`, 
    but with probability `turn_chance` rotate the direction
    by up to ±(theta_deg/2) toward the goal.
    """

    # half-angle in radians
    max_theta = np.deg2rad(theta_deg / 2)

    # vector toward sample
    v = sample - parent
    dist_to_sample = np.linalg.norm(v)
    if dist_to_sample == 0:
        return parent.copy()

    dir_v = v / dist_to_sample

    # with some chance, bias the direction toward the goal
    if np.random.rand() <= turn_chance:
        g = goal - parent
        dist_to_goal = np.linalg.norm(g)
        if dist_to_goal > 0:
            dir_g = g / dist_to_goal

            # angle between dir_v and dir_g
            dot = np.clip(np.dot(dir_v, dir_g), -1.0, 1.0)
            angle_between = np.arccos(dot)

            # if that angle is too large, rotate dir_v by ±max_theta toward dir_g
            if angle_between > max_theta:
                # determine whether to rotate CW or CCW
                cross_z = np.cross(np.append(dir_v, 0), np.append(dir_g, 0))[2]
                sign = np.sign(cross_z)

                # build rotation matrix for ±max_theta
                theta = sign * max_theta
                R = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]
                ])

                # rotate the original sample‐direction
                dir_v = R.dot(dir_v)

    # move by up to step_size along dir_v
    step = min(step_size, dist_to_sample) # type: ignore
    return parent + step * dir_v


# Not used
def original_steer(start: np.ndarray, goal: np.ndarray, step_size: float) -> np.ndarray:
    """
    Steer from a start point towards a goal point, constrained by a maximum step size.

    This function moves from the start point towards the goal point by a distance not exceeding the specified
    step size. If the goal is within the step size, it returns the goal point directly.

    Parameters:
        start (np.ndarray): Coordinates of the start point.
        goal (np.ndarray): Coordinates of the goal point.
        step_size (float): Maximum distance to move from the start point.

    Returns:
        np.ndarray: New point steered from the start towards the goal, adhering to the step size constraint.
    """
    total_dist = dist_between_points(start, goal)
    if total_dist <= step_size:
        return goal

    # compute vector from start toward goal
    direction = (goal - start) / total_dist
    return start + step_size * direction
