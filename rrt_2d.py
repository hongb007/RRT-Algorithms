from algorithm.rrt_algorithm import RRT
from utilities.plot import plot_rrt
from utilities.search_space import space
import numpy as np

np.random.seed(1)

dimensions = np.array([100, 100])
start = np.array([50, 50])
goal = np.array([60, 60])
goal_radius = 5
step_size = 3
n_samples = 3000
n_rectangles = 10
# size_range: (min_w, max_w), (min_h, max_h)
rect_sizes = np.array([[5, 15], [5, 15]])

rrt_space = space(
    dimensions=dimensions,
    start=start,
    goal=goal,
    goal_radius=goal_radius,
    n_samples=n_samples,
    n_rectangles=n_rectangles,
    rect_sizes=rect_sizes
)
rrt_algorithm = RRT(rrt_space)

tree, path_to_goal = rrt_algorithm.execute() or []

# if path_to_goal is None:
#     print("No solution found. Try again next time. ")
# else:
#     plot_rrt(tree, path_to_goal, rrt_space.rectangles)

# tree = rrt_tree(rrt_space)
# tree.add_node(100, 0)
# tree.add_node(70, 50)
# tree.show(data_property="array")
