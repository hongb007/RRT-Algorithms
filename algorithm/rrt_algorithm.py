import numpy as np
from algorithm.tree import rrt_tree
from utilities.search_space import space


class RRT(object):
    def __init__(self, space: space):
        self.space = space
        self.rrt_tree = rrt_tree(space)

    def execute(self):
        for i in range(0, self.space.n_samples):
            pose = np.array(
                [
                    np.random.uniform(0, self.space.dimensions[0]),
                    np.random.uniform(0, self.space.dimensions[1]),
                ]
            )

            steered_pose, nid = self.rrt_tree.add_node(pose)

            if (
                steered_pose is not None
                and nid is not None
                and self.space.close_to_goal(steered_pose)
            ):
                return self.rrt_tree.tree, self.rrt_tree.path_to_node(nid)
        return self.rrt_tree.tree, None
