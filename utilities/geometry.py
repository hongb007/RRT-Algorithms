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
    distance = np.linalg.norm(np.array(b) - np.array(a))
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
    Computes a new node position by steering a sample node to be within the step size of the parent node, 
    constrained by a maximum step size. Furthermore, this method includes a probabilistic adjustment 
    toward the goal direction within a specified angular constraint.

    Parameters:
    -----------
    parent : np.ndarray
        Coordinates of the current node (2D vector).
    sample : np.ndarray
        Coordinates of the randomly sampled node in the space (2D vector).
    step_size : float
        Maximum distance to move from the parent node towards the sample.
    goal : np.ndarray
        Coordinates of the goal node (2D vector).
    theta_deg : float
        Maximum allowable steering angle in degrees; defines the angular constraint.
    turn_chance : float
        Probability (between 0 and 1) of adjusting the steering direction towards the goal.

    Returns:
    --------
    np.ndarray
        Coordinates of the new node after steering (2D vector).
    """

    # Convert half of the maximum steering angle from degrees to radians
    max_theta = np.deg2rad(theta_deg / 2)

    # Compute the vector from the parent node to the sample node
    v = sample - parent
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        # If the sample coincides with the parent, no movement is needed
        return parent

    # Normalize the vector to obtain the direction
    dir_v = v / norm_v

    # Determine whether to adjust the steering direction towards the goal
    chance = np.random.uniform(0, 1)
    if chance <= turn_chance:
        # Compute the vector from the parent node to the goal
        g = goal - parent
        dir_g = g / np.linalg.norm(g)

        # Calculate the angle between dir_v and dir_g
        dot = np.clip(np.dot(dir_v, dir_g), -1.0, 1.0)
        angle = np.arccos(dot)

        # Determine the sign of the angle using the cross product
        cross = np.cross(np.append(dir_g, 0), np.append(dir_v, 0))[2]
        sign = np.sign(cross)

        # If the angle exceeds the maximum allowed, rotate dir_v towards dir_g
        if angle > max_theta:
            angle = max_theta * sign
            # Construct the 2D rotation matrix
            R = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Rotate dir_g by the constrained angle to obtain the new direction
            dir_v = R.dot(dir_g)

    # Determine the distance to move: the lesser of step_size and the distance to the sample
    dist = min(step_size, norm_v)  # type: ignore

    # Compute and return the new node position
    return parent + dist * dir_v


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
