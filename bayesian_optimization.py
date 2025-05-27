import numpy as np
import os
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# Assuming your RRT and space classes are in these locations
from algorithm.rrt_algorithm import RRT
from algorithm.search_space import space

# Global variable to hold the current rrt_space configuration for black_box_function
# This will be updated in each Bayesian Optimization session.
current_rrt_space_config = None


def black_box_function(step_size, theta, turn_percent, bias_percent):
    """
    Function with unknown internals we wish to maximize (actually minimize num_samples).
    It uses the global 'current_rrt_space_config'.
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
    )  # Renamed from 'start' to avoid conflict if 'start' is a keyword
    goal_pos = np.array([99, 99])  # Renamed from 'goal'
    goal_radius = 3
    n_rrt_samples = 1000  # Max samples/iterations for the RRT algorithm itself
    n_rectangles = 70
    rect_sizes = np.array([[5, 15], [5, 15]])

    # --- Bayesian Optimization Configuration ---
    cumulative_log_path = "./logs.json"  # Path for the cumulative log file

    # Bounded region of parameter space for Bayesian Optimization
    pbounds = {
        "step_size": (0.1, 10.0),  # Min step_size > 0
        "theta": (0.0, 360.0),
        "turn_percent": (0.0, 100.0),
        "bias_percent": (0.0, 50.0),  # Max bias can be adjusted
    }

    n_bo_sessions = 10  # Number of Bayesian Optimization meta-iterations (sessions)
    # Initial random points for the optimizer in the *first ever* session (if no log exists)
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
    # This logger will save all points evaluated by any optimizer it's subscribed to.
    logger = JSONLogger(path=cumulative_log_path)
    # Subscribe the logger to the optimizer. This should be done once.
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # Load all previously logged points into the optimizer before the first session
    # or before starting any new optimization run.
    if os.path.exists(cumulative_log_path):
        print(f"Loading existing logs from: {cumulative_log_path}")
        load_logs(optimizer, logs=[cumulative_log_path])
        print(f"Optimizer space now has {len(optimizer.space)} points after loading logs.")
    else:
        print(f"No existing log file found at {cumulative_log_path}. A new log will be created.")

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
            print("Best parameters found by this optimizer instance (current overall best):")
            print(optimizer.max)
        else:
            print(
                "No maximum found by this optimizer instance (perhaps no points were evaluated)."
            )
        print(
            f"Total unique points known to this optimizer instance: {len(optimizer.space)}"
        )

    print("\n--- Overall Best Result (from the final optimizer instance) ---")
    if optimizer.max:
        print("Overall best result from the optimizer's collected data:")
        print(optimizer.max)
    else:
        print("No overall best result found by the optimizer.")

    # You can also explicitly load and print the overall best from the log file
    # for verification, though the final 'optimizer.max' should reflect this.
    print("\n--- Verification: Overall Best Result (loaded directly from cumulative log) ---")
    if os.path.exists(cumulative_log_path):
        overall_optimizer_verifier = BayesianOptimization(
            f=None, pbounds=pbounds, random_state=999
        )
        load_logs(overall_optimizer_verifier, logs=[cumulative_log_path])
        if overall_optimizer_verifier.max:
            print("Overall best result from cumulative log file:")
            print(overall_optimizer_verifier.max)
        else:
            print(
                "Could not determine overall max from the log file (log might be empty or contain no valid iterations)."
            )
    else:
        print(
            f"Cumulative log file {cumulative_log_path} not found. Cannot perform direct log verification."
        )