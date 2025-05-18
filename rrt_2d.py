from algorithm.tree import rrt_tree
from utilities.world_space import space
import numpy as np

dimensions = np.array([100, 100])
start = np.array([0, 0])
goal = np.array([90, 90])
goal_radius = 1
step_size = 5

rrt_map = space(dimensions=dimensions, start=start, goal=goal, goal_radius=goal_radius)

tree = rrt_tree(rrt_map)
tree.add_node(100, 0)
tree.add_node(70, 50)
tree.show(data_property="array")
