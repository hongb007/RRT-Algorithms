from algorithm.rrt_algorithm import RRT
from algorithm.search_space import space
import numpy as np
from bayes_opt import BayesianOptimization

np.random.seed(1)


def black_box_function(step_size, theta, turn_percent, bias_percent):
    """
    Function with unknown internals we wish to maximize.
    """

    # Initialize the search space with the defined parameters
    rrt_space = space(
        dimensions=dimensions,
        start=start,
        goal=goal,
        goal_radius=goal_radius,
        step_size=step_size,
        theta=theta,
        turn_chance=turn_percent / 100.0,  # Convert percentage to a decimal
        bias_chance=bias_percent / 100.0,  # Convert percentage to a decimal
        n_samples=n_samples,
        n_rectangles=n_rectangles,
        rect_sizes=rect_sizes,
    )

    # Instantiate the RRT algorithm with the configured search space
    # The 'live' parameter enables real-time visualization during execution
    # The 'plot_result' parameter enables plotting after execution of the algorithm
    rrt_algorithm = RRT(rrt_space, live=False, plot_result=False)

    # Execute the RRT algorithm
    # Returns:
    # - found_path: if the algorithm found a path in the space constraints
    # - num_samples: number of samples it took to find a path to the goal
    found_path, num_samples = rrt_algorithm.execute()

    if found_path:
        return -1 * num_samples
    else:
        return -n_samples


if __name__ == "__main__":
    # Set the random seed for reproducibility of results
    np.random.seed(1)

    # Define the dimensions of the 2D workspace (width, height)
    dimensions = np.array([100, 100])

    # Specify the starting coordinates within the workspace
    start = np.array([1, 1])

    # Specify the goal coordinates within the workspace
    goal = np.array([99, 99])

    # Define the acceptable radius around the goal to consider as reached
    goal_radius = 3

    # Specify the total number of samples to be generated during RRT execution
    n_samples = 2000

    # Define the number of rectangular obstacles to be placed in the workspace
    n_rectangles = 80

    # Specify the range of sizes for the rectangular obstacles:
    # First row: (min_width, max_width), Second row: (min_height, max_height)
    rect_sizes = np.array([[5, 15], [5, 15]])

    # Bounded region of parameter space
    pbounds = {
        "step_size": (0.1, 10.0),
        "theta": (0.0, 360.0),
        "turn_percent": (0.0, 100.0),
        "bias_percent": (0.0, 100.0),
    }

    optimizer = BayesianOptimization(
        f=black_box_function,  # type: ignore
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.probe(
        params={
            "step_size": 8.306516820125049,
            "theta": 117.98523444669254,
            "turn_percent": 37.96730648800614,
            "bias_percent": 67.91604150776595,
        },
        lazy=True,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=100,
    )
    
    print(optimizer.max)
