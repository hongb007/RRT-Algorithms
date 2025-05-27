import numpy as np
from utilities.plot import LiveRRTPlot
from algorithm.tree import rrt_tree
import matplotlib.pyplot as plt

from algorithm.search_space import space


class RRT:
    def __init__(self, space: space, live: bool = True, plot_result: bool = True):
        """
        Initialize the Rapidly-Exploring Random Tree (RRT) planner.

        Parameters:
            space (space): The search space containing environment dimensions, obstacles, start and goal positions.
            live (bool): Flag indicating whether to enable live visualization during the planning process.
        """
        self.space = space
        self.rrt_tree = rrt_tree(space)
        self.live = live
        self.plot_result = plot_result

    def execute(self):
        """
        Execute the RRT algorithm to find a path from the start to the goal within the defined search space.

        The method performs the following steps:
            - Initializes the visualization tool if live plotting is enabled.
            - Iteratively samples random points in the search space.
            - Applies a bias towards the goal based on the specified bias probability.
            - Attempts to add new nodes to the tree by steering towards sampled points.
            - Checks for proximity to the goal to determine if a valid path has been found.
            - Visualizes the tree expansion and final path if live plotting is enabled.

        Returns:
            tuple: A tuple containing:
                - tree: The constructed RRT tree structure.
                - path: A list of node identifiers representing the path from start to goal, if found; otherwise, None.
        """
        plotter = LiveRRTPlot(self.space, live=self.live)
        path = None
        found_path = False      
        num_samples = 0

        # build until goal or no more samples
        for _ in range(self.space.n_samples):
            
            # simulating chance for generating sample near goal
            chance = np.random.uniform(0, 1)
            if chance <= self.space.bias_chance:
                # Non-uniformly generating samples radius of largest obstacle size. More dense toward center
                # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
                r = np.maximum(
                    self.space.rect_sizes[0][1], self.space.rect_sizes[1][1]
                ) * np.random.uniform(0, 1)
                theta = np.random.uniform(0, 1) * 2 * np.pi

                pose = np.array(
                    [
                        self.space.goal[0] + r * np.cos(theta),
                        self.space.goal[1] + r * np.sin(theta)
                    ]
                )
            else:
                pose = np.array(
                    [
                        np.random.uniform(0, self.space.dimensions[0]),
                        np.random.uniform(0, self.space.dimensions[1])
                    ]
                )

            steered, nid = self.rrt_tree.add_node(pose)
            if steered is not None:
                parent_id = self.rrt_tree.tree.get_node(nid).bpointer  # type: ignore
                parent_pt = self.rrt_tree.tree.get_node(parent_id).data.array  # type: ignore
                if self.live:
                    plotter.add_node(steered, parent_pt)
                if self.space.close_to_goal(steered):
                    path = self.rrt_tree.path_to_node(nid)  # type: ignore
                    found_path = True
                    break
        
        num_samples = self.rrt_tree.tree.size() - 1

        # If plot, draw full tree at the end
        if self.plot_result:
            plotter.plot_tree(self.rrt_tree.tree, num_samples)

        # draw path if found and if plot
        if path and self.plot_result:
            coords = [self.rrt_tree.tree.get_node(pid).data.array for pid in path]  # type: ignore
            plotter.plot_path(coords)

        # finally—block on the window in plot-result mode
        if self.plot_result:
            plt.show(block=True)
            
        plt.close()

        return found_path, num_samples
