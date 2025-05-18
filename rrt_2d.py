from algorithm.rrt_algorithm import RRT
from utilities.plot import plot_rrt
from utilities.world_space import space
import numpy as np

dimensions = np.array([100, 100])
start = np.array([50, 50])
goal = np.array([70, 70])
goal_radius = 5
step_size = 5
n_samples = 5000

rrt_space = space(
    dimensions=dimensions,
    start=start,
    goal=goal,
    goal_radius=goal_radius,
    n_samples=n_samples,
)

rrt_algorithm = RRT(rrt_space)

tree, path_to_goal = rrt_algorithm.execute() or []

if (path_to_goal is None):
    print("No solution found. Try again next time. ")
else:
    plot_rrt(tree, path_to_goal)

# tree = rrt_tree(rrt_space)
# tree.add_node(100, 0)
# tree.add_node(70, 50)
# tree.show(data_property="array")
