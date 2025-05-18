import numpy as np
from algorithm.tree import rrt_tree
from utilities.world_space import space
from utilities.geometry import dist_between_points


class RRT(object):
    def __init__(self, space: space):
        self.space = space
        self.tree = rrt_tree(space)

    def execute(self):
        np.random.seed(1)
        for i in range(0, self.space.n_samples):
            pose = np.array(
                [
                    np.random.uniform(0, self.space.dimensions[0]),
                    np.random.uniform(0, self.space.dimensions[1]),
                ]
            )

            steered_pose, nid = self.tree.add_node(pose)

            if self.space.close_to_goal(steered_pose):
                return self.tree.path_to_node(nid)
