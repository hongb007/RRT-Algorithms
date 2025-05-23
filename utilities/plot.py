import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from typing import List, Any

from algorithm.search_space import space


class LiveRRTPlot:
    """
    Provides real-time visualization for the Rapidly-exploring Random Tree (RRT) algorithm.
    This class handles the plotting of the search space, obstacles, start and goal points,
    and dynamically updates the tree expansion and final path.
    """

    def __init__(self, space: space, live: bool = True):
        """
        Initializes the plotting environment and renders static elements of the search space.

        Parameters:
        - space (space): The search space containing dimensions, obstacles, start, and goal.
        - live (bool): Flag to enable or disable live plotting. Defaults to True.
        """
        self.live = live
        self.space = space

        # Set interactive mode based on flag
        plt.ion() if self.live else plt.ioff()

        # Initialize the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        # Plot obstacles once
        for poly in space.rectangles:
            xs = [float(v.x) for v in poly.vertices]
            ys = [float(v.y) for v in poly.vertices]
            rect = Rectangle(
                (min(xs), min(ys)),
                max(xs) - min(xs),
                max(ys) - min(ys),
                facecolor="lightgrey",
                edgecolor="black",
                alpha=0.5,
                label="_nolegend_",
            )
            self.ax.add_patch(rect)

        # Plot start with marker only
        sx, sy = space.start
        self.ax.plot(
            sx,
            sy,
            marker="D",
            color="green",
            markersize=8,
            linestyle="None",
            label="Start",
        )

        # Plot goal point
        gx, gy = space.goal
        self.ax.plot(
            gx,
            gy,
            marker="*",
            color="orange",
            markersize=12,
            linestyle="None",
            label="Goal",
        )

        # Plot the goal region as a circle
        goal_circle = Circle(
            (gx, gy),
            space.goal_radius,
            edgecolor="orange",
            facecolor="none",
            linewidth=2,
            label="Goal Region",
        )
        self.ax.add_patch(goal_circle)

        # Add proxy legend entries for dynamic elements
        self.ax.plot([], [], "bo", markersize=4, linestyle="None", label="Nodes")
        self.ax.plot([], [], color="lightgrey", linewidth=1, label="Connections")

        # Set up axes
        self.ax.set_xlim(0, space.dimensions[0])
        self.ax.set_ylim(0, space.dimensions[1])
        self.ax.set_aspect("equal", "box")
        self.ax.set_title("RRT Expansion")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

        # Initialize storage for dynamic plot elements
        self.edge_lines = []
        self.node_scatter = []
        self.path_line = None

        # Add legend outside the plot area
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        self.fig.subplots_adjust(right=0.75)

        # Initial draw if live plotting is enabled
        if self.live:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def add_node(self, new_point, parent_point):
        """
        Adds a new node and its connection to the parent node in the plot.

        Parameters:
        - new_point (array-like): Coordinates of the new node.
        - parent_point (array-like): Coordinates of the parent node.
        """
        if not self.live:
            return

        # Draw the connection (edge) between the parent and new node
        (line,) = self.ax.plot(
            [parent_point[0], new_point[0]],
            [parent_point[1], new_point[1]],
            color="lightgrey",
            linewidth=1,
        )
        self.edge_lines.append(line)

        # Draw the new node
        scatter = self.ax.plot(
            new_point[0], new_point[1], "bo", markersize=4, linestyle="None"
        )
        self.node_scatter.append(scatter)

        # Refresh the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_tree(self, tree: Any):
        """
        Plots all existing nodes and connections from a treelib Tree.

        Parameters:
        - tree (Any): The tree structure containing nodes with 'data' attributes.
        """

        for node in tree.all_nodes_itr():
            parent = node.bpointer
            if parent is None:
                continue

            child_data = getattr(node, "data", None)
            parent_data = getattr(tree.get_node(parent), "data", None)
            if child_data is None or parent_data is None:
                continue

            child_pt = child_data.array
            parent_pt = parent_data.array

            # Draw the connection (edge)
            self.ax.plot(
                [parent_pt[0], child_pt[0]],
                [parent_pt[1], child_pt[1]],
                color="lightgrey",
                linewidth=1,
            )

            # Draw the node
            self.ax.plot(child_pt[0], child_pt[1], "bo", markersize=4, linestyle="None")

    def plot_path(self, path_coords: List[np.ndarray]):
        """
        Plots the final path from the start to the goal.

        Parameters:
        - path_coords (List[np.ndarray]): List of coordinates representing the path.
        """
        arr = np.array(path_coords)

        # Draw final path line
        (self.path_line,) = self.ax.plot(
            arr[:, 0],
            arr[:, 1],
            "-r",
            linewidth=2,
            marker="o",
            markersize=6,
            label="Path to Goal",
        )

        # Draw arrows for direction
        for i in range(len(arr) - 1):
            start, end = arr[i], arr[i + 1]
            self.ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            )

        # Update legend to include path
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

        # Replot start with marker only for clarity
        sx, sy = self.space.start
        self.ax.plot(
            sx,
            sy,
            marker="D",
            color="green",
            markersize=8,
            linestyle="None",
            label="Start",
        )

        # Redraw canvas
        self.fig.canvas.draw()
        # Always block and show final plot to keep window open
        plt.show(block=True)
