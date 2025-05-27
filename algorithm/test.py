import numpy as np

np.random.seed(1)

N_obstacles = 3

# (min_width, max_width), (min_height, max_height)
rect_sizes = np.array([[5, 15], [5, 15]])
bounds = np.array([100, 100])  # [bounds_x, bounds_y]
start = np.array([20, 20])
goal = np.array([80, 80])


def generate_rect_sizes(n):
    """
    Sample n widths/heights from rect_sizes.
    Returns two arrays of shape (n,): widths, heights.
    """
    (w_min, w_max), (h_min, h_max) = rect_sizes
    widths = np.random.rand(n) * (w_max - w_min) + w_min
    heights = np.random.rand(n) * (h_max - h_min) + h_min
    return widths, heights


def generate_rect_starting_pose(widths, heights):
    """
    Given arrays widths and heights of length n,
    sample bottom-left corners so each rect fits in [0..bounds].
    Returns two arrays (x_min, y_min) of shape (n,).
    """
    x_min = np.random.rand(widths.size) * (bounds[0] - widths)
    y_min = np.random.rand(heights.size) * (bounds[1] - heights)
    return x_min, y_min


# Initial generation
widths, heights = generate_rect_sizes(N_obstacles)
x_min, y_min = generate_rect_starting_pose(widths, heights)
x_max = x_min + widths
y_max = y_min + heights

# Keep regenerating bad rectangles until not covering start or goal
while True:
    # does each rect cover start or goal?
    covers_start = (
        (start[0] >= x_min)
        & (start[0] <= x_max)
        & (start[1] >= y_min)
        & (start[1] <= y_max)
    )
    covers_goal = (
        (goal[0] >= x_min)
        & (goal[0] <= x_max)
        & (goal[1] >= y_min)
        & (goal[1] <= y_max)
    )
    invalid = covers_start | covers_goal

    if not invalid.any():
        break

    # resample only the bad ones
    bad = np.nonzero(invalid)[0]
    nb = bad.size

    # widths/heights -> new x_min,y_min -> new x_max,y_max
    w_new, h_new = generate_rect_sizes(nb)
    x_min_new, y_min_new = generate_rect_starting_pose(w_new, h_new)

    widths[bad] = w_new
    heights[bad] = h_new
    x_min[bad] = x_min_new
    y_min[bad] = y_min_new
    x_max[bad] = x_min_new + w_new
    y_max[bad] = y_min_new + h_new

print("x_min:", x_min)
print("y_min:", y_min)
print("x_max:", x_max)
print("y_max:", y_max)



arr = np.array([[1,1,1,1], [2,2,2,2]])


ones, twos = arr
x, y, threes, fours = arr.T

print(arr)
print(ones)
print(twos)
print(ones.T.shape)
print(twos.T.shape)

print(1 > ones)

one = np.array([2])

array = np.array([1,2,3])
array2 = np.array([4,5,6])

print(one*array)

print(np.array([0]) == 0)

print(ones / twos)

print(array < array2)

print(arr[0])

print(np.array([array, array2]))