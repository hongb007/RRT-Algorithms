from algorithm.tree import rrt_tree
from utilities.map import map
import numpy as np

dimensions = np.array([100, 100])
start = np.array([50, 50])
goal = np.array([90, 90])
goal_radius = 1

rrt_map = map(dimensions=dimensions, start=start, goal=goal, goal_radius=goal_radius)

tree = rrt_tree(rrt_map)
tree.add_node(1, 2)
tree.add_node(3, 4)
tree.add_node(51, 51)
tree.show(data_property="array")
