import numpy as np
from tree import rrt_tree
from utilities.world_space import space
from utilities.geometry import dist_between_points


class RRT(object):
    def __init__(self, space: space):
        self.space = space
        self.tree = rrt_tree(space)
        
    def execute(self):
        pass

    def close_to_goal(self, pose: np.ndarray):
        return dist_between_points(pose, self.space) <= self.space.step_size
    