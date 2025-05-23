import argparse
from algorithm.rrt_algorithm import RRT
from algorithm.search_space import space
import numpy as np

# Set the random seed for reproducibility of results
np.random.seed(8)

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run the RRT algorithm with optional live visualization."
)
parser.add_argument(
    "--live",
    type=lambda x: x.lower() == "true",
    default=True,
    help="Enable live visualization (True or False). Default is True.",
)
args = parser.parse_args()

# Define the dimensions of the 2D workspace (width, height)
dimensions = np.array([100, 100])

# Specify the starting coordinates within the workspace
start = np.array([1, 1])

# Specify the goal coordinates within the workspace
goal = np.array([99, 99])

# Define the acceptable radius around the goal to consider as reached
goal_radius = 3

# Set the maximum distance the tree can extend in one iteration
step_size = 3

# Define the maximum steering angle in degrees
theta = 180

# Set the percentage bias towards sampling the goal directly
bias_percent = 10.0

# Specify the total number of samples to be generated during RRT execution
n_samples = 3000

# Define the number of rectangular obstacles to be placed in the workspace
n_rectangles = 80

# Specify the range of sizes for the rectangular obstacles:
# First row: (min_width, max_width), Second row: (min_height, max_height)
rect_sizes = np.array([[1, 15], [1, 15]])

# Initialize the search space with the defined parameters
rrt_space = space(
    dimensions=dimensions,
    start=start,
    goal=goal,
    goal_radius=goal_radius,
    step_size=step_size,
    theta=theta,
    bias=bias_percent / 100.0,  # Convert percentage to a decimal
    n_samples=n_samples,
    n_rectangles=n_rectangles,
    rect_sizes=rect_sizes,
)

# Instantiate the RRT algorithm with the configured search space
# The 'live' parameter enables real-time visualization during execution
rrt_algorithm = RRT(rrt_space, live=args.live)

# Execute the RRT algorithm
# Returns:
# - tree: the constructed RRT tree
# - path_to_goal: the path from start to goal if found; otherwise, an empty list
tree, path_to_goal = rrt_algorithm.execute() or []

# if path_to_goal is None:
#     print("No solution found. Try again next time. ")
# else:
#     plot_rrt(tree, path_to_goal, rrt_space.rectangles)

# tree.show(data_property="array")
