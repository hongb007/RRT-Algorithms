from algorithm.rrt_algorithm import RRT
from utilities.world_space import space
import numpy as np

dimensions = np.array([100, 100])
start = np.array([0, 0])
goal = np.array([100, 100])
goal_radius = 5
step_size = 2
n_samples = 1000

rrt_space = space(
    dimensions=dimensions,
    start=start,
    goal=goal,
    goal_radius=goal_radius,
    n_samples=n_samples,
)

rrt_algorithm = RRT(rrt_space)

poses = rrt_algorithm.execute() or []

for p in poses:
    print(p)

# tree = rrt_tree(rrt_space)
# tree.add_node(100, 0)
# tree.add_node(70, 50)
# tree.show(data_property="array")
