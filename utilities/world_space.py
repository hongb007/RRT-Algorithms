import numpy as np


class space(object):
    def __init__(
        self,
        dimensions: np.ndarray | None = None,
        start: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        goal_radius: float = 1.0,
        step_size: float = 5.0,
    ):
        # 1. Default to 2×2 zero arrays if None
        self.dimensions = dimensions if dimensions is not None else np.zeros((2, 2))
        self.start = start if start is not None else np.zeros((2, 2))
        self.goal = goal if goal is not None else np.zeros((2, 2))
        self.goal_radius = goal_radius
        self.step_size = step_size
        self.obstacles = self.generate_obstacles()

    def generate_obstacles(self):
        pass

    def collision_free_path(self, pose1: np.ndarray, pose2: np.ndarray):
        return True
