import plotly.graph_objects as go
import numpy as np
from typing import List
from treelib.tree import Tree
from algorithm.tree import ArrayHolder

def plot_rrt(tree: Tree, path: List[str]) -> None:
    edge_x, edge_y = [], []
    for node in tree.all_nodes_itr():
        parent_id = node.predecessor
        if parent_id is None:
            continue
        parent = tree.get_node(parent_id)
        data = node.data
        p_data = parent.data if parent else None
        if not isinstance(data, ArrayHolder) or not isinstance(p_data, ArrayHolder):
            continue
        x0, y0 = data.array
        x1, y1 = p_data.array
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1, color='lightgrey'),
            name='Tree Edges'
        )
    )

    node_coords = []
    for node in tree.all_nodes_itr():
        data = node.data
        if isinstance(data, ArrayHolder):
            node_coords.append(data.array)
    if node_coords:
        arr = np.array(node_coords)
        fig.add_trace(
            go.Scatter(
                x=arr[:, 0],
                y=arr[:, 1],
                mode='markers',
                marker=dict(size=4, color='blue'),
                name='Nodes'
            )
        )

    root_id = tree.root
    if root_id:
        root = tree.get_node(root_id)
        data = root.data if root else None
        if isinstance(data, ArrayHolder):
            sx, sy = data.array
            fig.add_trace(
                go.Scatter(
                    x=[sx],
                    y=[sy],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='diamond'),
                    name='Start'
                )
            )

    if path:
        path_coords = []
        for nid in path:
            node = tree.get_node(nid)
            data = node.data if node else None
            if isinstance(data, ArrayHolder):
                path_coords.append(data.array)
        if path_coords:
            pc = np.array(path_coords)
            fig.add_trace(
                go.Scatter(
                    x=pc[:, 0],
                    y=pc[:, 1],
                    mode='lines+markers',
                    line=dict(width=3, color='red'),
                    marker=dict(size=6, color='red'),
                    name='Path to Goal'
                )
            )
            gx, gy = pc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[gx],
                    y=[gy],
                    mode='markers',
                    marker=dict(size=10, color='orange', symbol='star'),
                    name='Goal'
                )
            )

    fig.update_layout(
        title='RRT Exploration and Path',
        xaxis_title='X',
        yaxis_title='Y',
        width=700,
        height=700,
        showlegend=True,
        plot_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridcolor='lightgrey', scaleanchor='x', scaleratio=1)
    fig.show()
