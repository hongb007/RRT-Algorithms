import numpy as np
from algorithm.tree import rrt_tree
from utilities.plot import LiveRRTPlot

class RRT:
    def __init__(self, space):
        self.space = space
        self.rrt_tree = rrt_tree(space)

    def execute(self):
        # Initialize live plot
        plotter = LiveRRTPlot(self.space)

        for i in range(self.space.n_samples):
            pose = np.array([
                np.random.uniform(0, self.space.dimensions[0]),
                np.random.uniform(0, self.space.dimensions[1]),
            ])

            steered_pose, nid = self.rrt_tree.add_node(pose)

            if steered_pose is not None and nid is not None:
                # Fetch parent coords
                parent_id = self.rrt_tree.tree.get_node(nid).bpointer # type: ignore
                parent_node = self.rrt_tree.tree.get_node(parent_id)
                parent_coords = parent_node.data.array # type: ignore

                # Live update plot
                plotter.add_node(steered_pose, parent_coords)

                # Check goal
                if self.space.close_to_goal(steered_pose):
                    path_node_ids = self.rrt_tree.path_to_node(nid)
                    path_coords = [self.rrt_tree.tree.get_node(pid).data.array for pid in path_node_ids] # type: ignore
                    # Plot final path and keep window open
                    plotter.plot_path(path_coords)
                    return self.rrt_tree.tree, path_node_ids

        # Finished without goal
        return self.rrt_tree.tree, None
