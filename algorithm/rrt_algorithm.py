import numpy as np
from utilities.plot import LiveRRTPlot
from algorithm.tree import rrt_tree
import matplotlib.pyplot as plt

class RRT:
    def __init__(self, space, live: bool = True):
        self.space = space
        self.rrt_tree = rrt_tree(space)
        self.live = live

    def execute(self):
        plotter = LiveRRTPlot(self.space, live=self.live)
        tree, path = None, None

        # build until goal
        for _ in range(self.space.n_samples):
            chance = np.random.uniform(0,1)
            if (chance < self.space.bias):
                pose = self.space.goal
            else:
                pose = np.array([
                    np.random.uniform(0, self.space.dimensions[0]),
                    np.random.uniform(0, self.space.dimensions[1]),
                ])

            steered, nid = self.rrt_tree.add_node(pose)
            if steered is not None:
                parent_id = self.rrt_tree.tree.get_node(nid).bpointer # type: ignore
                parent_pt = self.rrt_tree.tree.get_node(parent_id).data.array # type: ignore
                plotter.add_node(steered, parent_pt)
                if self.space.close_to_goal(steered):
                    tree = self.rrt_tree.tree
                    path = self.rrt_tree.path_to_node(nid) # type: ignore
                    break

        # if non-live, draw full tree
        if not self.live:
            tree = tree or self.rrt_tree.tree
            plotter.plot_tree(tree)

        # draw path if found
        if path:
            coords = [self.rrt_tree.tree.get_node(pid).data.array for pid in path] # type: ignore
            plotter.plot_path(coords)

        # finally—block on the window in non-live mode
        if not self.live:
            plt.show(block=True)

        return tree, path
