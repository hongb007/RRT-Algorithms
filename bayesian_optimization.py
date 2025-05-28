import numpy as np
import os
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import time
from algorithm.rrt_algorithm import RRT
from algorithm.search_space import space

# Global variable to hold the current rrt_space configuration for black_box_function
# This will be updated in each Bayesian Optimization session.
current_rrt_space_config = None

def black_box_function(step_size, theta, turn_percent, bias_percent):
    """
    Evaluates the performance of the RRT algorithm for given parameters.

    This function serves as the objective for the Bayesian Optimization. It
    instantiates and executes the RRT algorithm with the provided parameters
    and a pre-configured search space. The goal is to minimize the number of
    samples (nodes) required to find a path. Since Bayesian Optimization is
    a maximization algorithm, the negative of the number of samples is returned.
    A significant penalty is applied if no path is found.

    The `current_rrt_space_config` global variable must be initialized
    before calling this function.

    Parameters
    ----------
    step_size : float
        The maximum distance for a new node to extend from its nearest neighbor.
    theta : float
        The angular range (in degrees) for steering the RRT expansion.
    turn_percent : float
        The probability (0-100) of performing a 'turn' action during RRT expansion.
    bias_percent : float
        The probability (0-100) of biasing the RRT expansion towards the goal.

    Returns
    -------
    float
        The negative of the number of samples taken by the RRT algorithm if a
        path is found. Returns a large negative penalty if no path is found,
        making it undesirable for the optimizer.

    Raises
    ------
    ValueError
        If `current_rrt_space_config` is not initialized (i.e., is None)
        before the function is called.
    """
    global current_rrt_space_config  # Explicitly state it's using the global
    if current_rrt_space_config is None:
        raise ValueError(
            "current_rrt_space_config has not been initialized for this black_box_function call."
        )

    # Instantiate the RRT algorithm with the configured search space
    rrt_algorithm = RRT(
        space=current_rrt_space_config,  # Use the globally set space for this session
        # The following parameters are optimized by Bayesian Optimization
        step_size=step_size,
        theta=theta,
        turn_chance=turn_percent / 100.0,
        bias_chance=bias_percent / 100.0,
        # Fixed parameters for RRT execution
        live=False,
        plot_result=False,
    )

    # Execute the RRT algorithm
    found_path, num_samples, n_tries_to_place = rrt_algorithm.execute()

    # Objective: Minimize num_samples if path is found.
    # BayesianOptimization maximizes, so we return negative of what we want to minimize.
    if found_path:
        # Smaller num_samples is better, so -num_samples is larger (better for BO)
        return -1.0 * float(num_samples)
    else:
        # No path found. We want to penalize this.
        # A large penalty, worse than any successful path.
        # For example, -(max_rrt_samples + current_nodes_added)
        # This ensures that not finding a path is always worse than finding one,
        # even if the found path took many samples.
        penalty = float(current_rrt_space_config.n_samples) + float(num_samples)
        return -1.0 * penalty


if __name__ == "__main__":
    # Set the random seed for reproducibility of NumPy operations (e.g., obstacle generation)
    np.random.seed(1)

    # --- RRT Configuration Parameters (used to create `current_rrt_space_config`) ---
    dimensions = np.array([100, 100])
    start_pos = np.array(
        [1, 1]
    )  
    goal_pos = np.array([99, 99]) 
    goal_radius = 3
    n_rrt_samples = 1000  # Max samples/iterations for the RRT algorithm itself
    n_rectangles = 65
    rect_sizes = np.array([[5, 15], [5, 15]])

    # --- Bayesian Optimization Configuration ---
    cumulative_log_path = "./logs.json"  # Path for the cumulative log file

    # Bounded region of parameter space for Bayesian Optimization
    pbounds = {
        "step_size": (0.1, 8.0),  # Min step_size > 0
        "theta": (0.0, 360.0),
        "turn_percent": (0.0, 100.0),
        "bias_percent": (0.0, 50.0),  # Max bias can be adjusted
    }

    n_bo_sessions = 5  # Number of Bayesian Optimization meta-iterations (sessions)
    # Initial random points for the optimizer in the first session (if no log exists)
    initial_bo_points_on_first_run = 5
    # Number of optimization iterations per BO session (after initial points)
    n_bo_iterations_per_session = 50  # Adjust as needed (e.g., to 50-100 for real runs)

    # Initialize the optimizer outside the loop to maintain its state across sessions
    # We will load logs into this single optimizer instance.
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,  # Use a consistent random state for the optimizer
        verbose=2,
    )

    # Setup the JSONLogger to append to the cumulative log file.
    # Saves all points evaluated by any optimizer it's subscribed to.
    logger = JSONLogger(path=cumulative_log_path)
    # Subscribe the logger to the optimizer.
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # Load all previously logged points into the optimizer before the first session
    # or before starting any new optimization run.
    if os.path.exists(cumulative_log_path):
        print(f"Loading existing logs from: {cumulative_log_path}")
        load_logs(optimizer, logs=[cumulative_log_path])
        print(
            f"Optimizer space now has {len(optimizer.space)} points after loading logs."
        )
    else:
        print(
            f"No existing log file found at {cumulative_log_path}. A new log will be created."
        )

    start = time.perf_counter()
    print("\nStarting timer.")
    
    for i in range(n_bo_sessions):
        print(f"\n--- Bayesian Optimization Session {i + 1}/{n_bo_sessions} ---")

        # Update the global rrt_space configuration for this session
        # This object will be used by black_box_function
        current_rrt_space_config = space(
            dimensions=dimensions,
            start=start_pos,
            goal=goal_pos,
            goal_radius=goal_radius,
            n_samples=n_rrt_samples,
            n_rectangles=n_rectangles,
            rect_sizes=rect_sizes,
        )

        # Determine initial points for this specific session's `maximize` call
        # If the optimizer already has points (from loaded logs or previous sessions),
        # we don't need new random `init_points`.
        current_maximize_init_points = 0
        if (
            len(optimizer.space) == 0
        ):  # Only use initial_bo_points_on_first_run if space is truly empty
            current_maximize_init_points = initial_bo_points_on_first_run
            print(
                f"Optimizer space is empty. Using {current_maximize_init_points} initial random points for this session."
            )
        else:
            print(
                f"Optimizer space has {len(optimizer.space)} points. Using 0 new initial random points, relying on history."
            )

        print(
            f"Starting optimizer.maximize() with init_points={current_maximize_init_points} and n_iter={n_bo_iterations_per_session}."
        )

        optimizer.maximize(
            init_points=current_maximize_init_points,
            n_iter=n_bo_iterations_per_session,
        )

        print(f"--- End of Session {i + 1} ---")
        if optimizer.max:
            print(
                "Best parameters found by this optimizer instance (current overall best):"
            )
            print(optimizer.max)
        else:
            print(
                "No maximum found by this optimizer instance (perhaps no points were evaluated)."
            )
        print(
            f"Total unique points known to this optimizer instance: {len(optimizer.space)}"
        )
        
    end = time.perf_counter()
    print(f"\nTotal time taken to run: {(end - start) / 60} minutes")

    print("\n--- Examining Best Point and its Neighbors ---")
    if optimizer.max:
        best_target = optimizer.max["target"]
        best_params = optimizer.max["params"]
        print(f"Overall Best Target: {best_target} with params: {best_params}")

        # Find points close to the best one in terms of target value
        tolerance = 30 # Within 30 samples of best target
        nearby_good_points = []
        for res in optimizer.res:
            if (
                abs(res["target"] - best_target) < tolerance
                and not (res["target"] == best_target)
                and not (res["params"] == best_params)
            ):
                nearby_good_points.append(res)

        print(
            f"\nFound {len(nearby_good_points)} points with target values close to the best (within {tolerance:.2f}):"
        )
        # Sort by target value (descending)
        nearby_good_points.sort(key=lambda x: x["target"], reverse=True)

        for i, point in enumerate(
            nearby_good_points[:10]
        ):  # Print top 10 similar points
            print(
                f"  {i + 1}. Target: {point['target']:.2f}, Params: {point['params']}"
            )
    else:
        print("No overall best result to examine.")
