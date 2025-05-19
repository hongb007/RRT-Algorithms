import plotly.graph_objects as go
import numpy as np
from typing import List, cast
from treelib.tree import Tree
from algorithm.tree import ArrayHolder
from sympy import Polygon as SympyPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def plot_rrt(tree: Tree, path: List[str], rectangles: List[SympyPolygon]) -> None:
    fig = go.Figure()

    # draw obstacles
    if rectangles:
        for poly in rectangles:
            verts = poly.vertices
            xs = [float(v.x) for v in verts]  # type: ignore
            ys = [float(v.y) for v in verts]  # type: ignore
            fig.add_shape(
                type="rect",
                x0=min(xs),
                y0=min(ys),
                x1=max(xs),
                y1=max(ys),
                line=dict(color="black"),
                fillcolor="lightgrey",
                opacity=0.5,
                layer="below",
                name="Obstacle",
            )

    # tree edges
    edge_x, edge_y = [], []
    for node in tree.all_nodes_itr():
        parent_id = node.bpointer
        if not parent_id:
            continue
        parent = tree.get_node(parent_id)
        if (
            parent is None
            or not isinstance(node.data, ArrayHolder)
            or not isinstance(parent.data, ArrayHolder)
        ):
            continue
        x0, y0 = node.data.array
        x1, y1 = parent.data.array
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="lightgrey"),
            name="Tree Edges",
        )
    )

    # all nodes
    node_coords = [
        cast(ArrayHolder, n.data).array
        for n in tree.all_nodes_itr()
        if isinstance(n.data, ArrayHolder)
    ]
    if node_coords:
        arr = np.array(node_coords)
        fig.add_trace(
            go.Scatter(
                x=arr[:, 0],
                y=arr[:, 1],
                mode="markers",
                marker=dict(size=4, color="blue"),
                name="Nodes",
            )
        )

    # start
    root = tree.get_node(tree.root)
    if root and isinstance(root.data, ArrayHolder):
        sx, sy = root.data.array
        fig.add_trace(
            go.Scatter(
                x=[sx],
                y=[sy],
                mode="markers",
                marker=dict(size=10, color="green", symbol="diamond"),
                name="Start",
            )
        )

    # path and goal
    if path:
        path_coords = [
            cast(ArrayHolder, tree.get_node(nid).data).array  # type: ignore
            for nid in path
            if tree.get_node(nid) and isinstance(tree.get_node(nid).data, ArrayHolder)  # type: ignore
        ]  # type: ignore
        if path_coords:
            pc = np.array(path_coords)
            fig.add_trace(
                go.Scatter(
                    x=pc[:, 0],
                    y=pc[:, 1],
                    mode="lines+markers",
                    line=dict(width=3, color="red"),
                    marker=dict(size=6, color="red"),
                    name="Path to Goal",
                )
            )
            gx, gy = pc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[gx],
                    y=[gy],
                    mode="markers",
                    marker=dict(size=10, color="orange", symbol="star"),
                    name="Goal",
                )
            )

    # layout
    fig.update_layout(
        title="RRT with Obstacles and Path",
        xaxis_title="X",
        yaxis_title="Y",
        width=700,
        height=700,
        showlegend=True,
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgrey", range=[0, 100])
    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgrey",
        scaleanchor="x",
        scaleratio=1,
        range=[0, 100],
    )
    fig.show()


class LiveRRTPlot:
    def __init__(self, space):
        # Interactive mode on
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7,7))

        # Plot obstacles once
        for poly in space.rectangles:
            xs = [float(v.x) for v in poly.vertices]
            ys = [float(v.y) for v in poly.vertices]
            rect = Rectangle((min(xs), min(ys)), max(xs)-min(xs), max(ys)-min(ys),
                             facecolor='lightgrey', edgecolor='black', alpha=0.5, label='_nolegend_')
            self.ax.add_patch(rect)

        # Plot start
        sx, sy = space.start
        self.ax.plot(sx, sy, marker='D', color='green', markersize=8, label='Start')

        # Plot goal point and goal region
        gx, gy = space.goal
        self.ax.plot(gx, gy, marker='*', color='orange', markersize=12, label='Goal')
        goal_circle = Circle((gx, gy), space.goal_radius, edgecolor='orange',
                              facecolor='none', linewidth=2, label='Goal Region')
        self.ax.add_patch(goal_circle)

        # Set up axes
        self.ax.set_xlim(0, space.dimensions[0])
        self.ax.set_ylim(0, space.dimensions[1])
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title('RRT Live Expansion')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True)

        # Initialize lists to hold plotted lines and points
        self.edge_lines = []
        self.node_scatter = []
        self.path_line = None

        # Add legend
        self.ax.legend(loc='upper right')

        # Show initial plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_node(self, new_point, parent_point):
        # Draw edge
        line, = self.ax.plot([parent_point[0], new_point[0]],
                             [parent_point[1], new_point[1]],
                             color='lightgrey', linewidth=1)
        self.edge_lines.append(line)
        # Draw node
        scatter = self.ax.plot(new_point[0], new_point[1], 'bo', markersize=4, label='_nolegend_')
        self.node_scatter.append(scatter)

        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def plot_path(self, path_coords):
        # Turn off interactive mode to block on show
        plt.ioff()
        arr = np.array(path_coords)
        # Plot path
        self.path_line, = self.ax.plot(arr[:,0], arr[:,1], '-r', linewidth=2, marker='o', markersize=6, label='Path to Goal')
        # Update legend to include path
        self.ax.legend(loc='upper right')
        # Final draw
        self.fig.canvas.draw()
        # Block until window closed by user
        plt.show()