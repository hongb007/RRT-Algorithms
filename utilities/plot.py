import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from typing import List, Any

from utilities.search_space import space


class LiveRRTPlot:
    def __init__(self, space: space, live: bool = True):
        self.live = live
        self.space = space
        # Set interactive mode based on flag
        plt.ion() if self.live else plt.ioff()
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

        # Plot goal point and region with marker only
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

        # Initialize storage
        self.edge_lines = []
        self.node_scatter = []
        self.path_line = None

        # Add legend outside
        self.ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        self.fig.subplots_adjust(right=0.75)

        # Initial draw if live
        if self.live:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def add_node(self, new_point, parent_point):
        if not self.live:
            return
        # Draw connection
        (line,) = self.ax.plot(
            [parent_point[0], new_point[0]],
            [parent_point[1], new_point[1]],
            color="lightgrey",
            linewidth=1,
        )
        self.edge_lines.append(line)
        # Draw node
        scatter = self.ax.plot(
            new_point[0], new_point[1], "bo", markersize=4, linestyle="None"
        )
        self.node_scatter.append(scatter)
        # Refresh
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_tree(self, tree: Any):
        """
        Plot all existing nodes and connections from a treelib Tree.
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
            self.ax.plot(
                [parent_pt[0], child_pt[0]],
                [parent_pt[1], child_pt[1]],
                color="lightgrey",
                linewidth=1,
            )
            self.ax.plot(child_pt[0], child_pt[1], "bo", markersize=4, linestyle="None")

    def plot_path(self, path_coords: List[np.ndarray]):
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
