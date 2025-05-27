import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from typing import List, Any

from algorithm.search_space import space


class LiveRRTPlot:
    """
    Provides real-time visualization for the Rapidly-exploring Random Tree (RRT) algorithm.
    This class handles the plotting of the search space, obstacles, start and goal points,
    and number of samples. This class also dynamically updates the tree expansion and number of samples. 
    Finally, a valid path from the start to the goal is highlighted when found.
    """

    def __init__(self, space: space, live: bool = True):
        """
        Initialize plotting environment and render static elements.

        Parameters:
        - space (space): The search space with dimensions, obstacles, start, and goal.
        - live (bool): Enable live plotting updates if True.
        """
        self.live = live
        self.space = space

        # Sample counter excludes start/goal
        self.sample_count = 0

        # Set interactive plotting mode
        if self.live:
            plt.ion()
        else:
            plt.ioff()

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        # Plot static obstacles
        # space.rectangles has shape (4, N): [x_min; y_min; x_max; y_max]
        # we zip across columns to get each rect’s coords
        for x_min, y_min, x_max, y_max in zip(*space.rectangles):
            rect = Rectangle(
                (x_min, y_min),           # lower‐left corner
                (x_max - x_min),          # width
                (y_max - y_min),          # height
                facecolor="lightgrey",
                edgecolor="black",
                alpha=0.5,
                label="_nolegend_",
            )
            self.ax.add_patch(rect)

        # Plot start and goal
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
        goal_circle = Circle(
            (gx, gy),
            space.goal_radius,
            edgecolor="orange",
            facecolor="none",
            linewidth=2,
            label="Goal Region",
        )
        self.ax.add_patch(goal_circle)

        # Add legend proxies for dynamic elements
        self.ax.plot([], [], "bo", markersize=4, linestyle="None", label="Nodes")
        self.ax.plot([], [], color="lightgrey", linewidth=1, label="Connections")

        # Initialize dynamic storage
        self.edge_lines = []
        self.node_scatter = []
        self.path_line = None

        # Sample counter text: positioned below legend, with boxed background matching legend style
        self.sample_text = self.ax.text(
            1.06,
            0.7,
            f"Num Samples: {self.sample_count}",
            transform=self.ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(
                facecolor="white",
                edgecolor="lightgray",
                boxstyle="round,pad=0.3",
                alpha=0.8,
            ),
        )

        # Configure axes
        self.ax.set_xlim(0, space.dimensions[0])
        self.ax.set_ylim(0, space.dimensions[1])
        self.ax.set_aspect("equal", "box")
        self.ax.set_title("RRT Expansion")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

        # Add legend outside plot
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        self.fig.subplots_adjust(right=0.75)

        # Initial draw
        if self.live:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def add_node(self, new_point, parent_point):
        """
        Add a new node and its connection to the plot, updating the sample counter.

        Parameters:
        - new_point (array-like): Coordinates of the new node.
        - parent_point (array-like): Coordinates of the parent node.
        """
        if not self.live:
            return

        # Increment sample counter and update text
        self.sample_count += 1
        self.sample_text.set_text(f"Num Samples: {self.sample_count}")

        # Draw connection line
        (line,) = self.ax.plot(
            [parent_point[0], new_point[0]],
            [parent_point[1], new_point[1]],
            color="lightgrey",
            linewidth=1,
        )
        self.edge_lines.append(line)

        # Draw node point
        scatter = self.ax.plot(
            new_point[0], new_point[1], "bo", markersize=4, linestyle="None"
        )
        self.node_scatter.append(scatter)

        # Refresh canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_tree(self, tree: Any):
        """
        Plot all existing nodes and edges from a treelib Tree (non-live mode).

        Parameters:
        - tree (Any): Tree containing nodes with 'data' array attributes.
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

            # Draw edge and node
            self.ax.plot(
                [parent_pt[0], child_pt[0]],
                [parent_pt[1], child_pt[1]],
                color="lightgrey",
                linewidth=1,
            )
            self.ax.plot(child_pt[0], child_pt[1], "bo", markersize=4, linestyle="None")

    def plot_path(self, path_coords: List[np.ndarray]):
        """
        Plot the final path, update legend, and preserve sample count display.

        Parameters:
        - path_coords (List[np.ndarray]): Sequence of coordinates for the path.
        """
        arr = np.array(path_coords)

        # Draw path line with markers
        (self.path_line,) = self.ax.plot(
            arr[:, 0],
            arr[:, 1],
            "-r",
            linewidth=2,
            marker="o",
            markersize=6,
            label="Path to Goal",
        )

        # Draw directional arrows
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

        # Replot start marker for clarity
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

        # After final path, update text to show final sample count
        self.sample_text.set_text(f"Num Samples: {self.sample_count}")
        self.sample_text.set_position(xy=(1.06, 0.65))

        # Draw and block to keep window open
        self.fig.canvas.draw()
        plt.show(block=True)
