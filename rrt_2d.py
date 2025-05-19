from algorithm.rrt_algorithm import RRT
from utilities.search_space import space
import numpy as np

np.random.seed(2)

dimensions = np.array([100, 100])
start = np.array([99, 99])
goal = np.array([50, 5])
goal_radius = 3
step_size = 5
# In degrees
theta = 120
bias_percent = 10.0
n_samples = 3000
n_rectangles = 30
# size_range: (min_w, max_w), (min_h, max_h)
rect_sizes = np.array([[5, 15], [5, 15]])

rrt_space = space(
    dimensions=dimensions,
    start=start,
    goal=goal,
    goal_radius=goal_radius,
    step_size=step_size,
    theta=theta,
    bias = bias_percent/100.0,
    n_samples=n_samples,
    n_rectangles=n_rectangles,
    rect_sizes=rect_sizes
)
rrt_algorithm = RRT(rrt_space, live=True)

tree, path_to_goal = rrt_algorithm.execute() or []

# if path_to_goal is None:
#     print("No solution found. Try again next time. ")
# else:
#     plot_rrt(tree, path_to_goal, rrt_space.rectangles)

# tree = rrt_tree(rrt_space)
# tree.add_node(100, 0)
# tree.add_node(70, 50)
# tree.show(data_property="array")
