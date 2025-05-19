from typing import Optional
import numpy as np
from utilities.geometry import dist_between_points
from sympy import Point, Polygon, Segment


class space(object):
    def __init__(
        self,
        dimensions: np.ndarray,
        start: np.ndarray,
        goal: np.ndarray,
        goal_radius: float = 1.0,
        step_size: float = 5.0,
        theta: float = 90,
        bias: float = 0,
        n_samples: int = 1000,
        n_rectangles: int = 10,
        rect_sizes: np.ndarray = np.zeros((2, 2)),
    ):
        self.dimensions = dimensions
        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius
        self.step_size = step_size
        self.theta = theta
        self.bias = bias
        self.n_samples = n_samples
        self.rect_sizes = rect_sizes
        self.generate_obstacles(n_rectangles, rect_sizes)

    def generate_obstacles(self, n_rectangles: int, rect_sizes):
        self.rectangles = self.generate_rectangles(n_rectangles, rect_sizes)
        self.rectangles.extend(self.generate_border())

    def generate_border(self):
        border = []

        x = self.dimensions[0]
        y = self.dimensions[1]

        p1, p2, p3, p4 = (Point(0, 0), Point(x, 0), Point(x, y), Point(0, y))

        thickness = 0.1

        rect1 = Polygon(p1, p2, Point(x, thickness), Point(0, thickness))
        rect2 = Polygon(Point(x - thickness, 0), p2, p3, Point(x - thickness, y))
        rect3 = Polygon(Point(0, y - thickness), Point(x, y - thickness), p3, p4)
        rect4 = Polygon(p1, Point(thickness, 0), Point(thickness, y), p4)

        border.append(rect1)
        border.append(rect2)
        border.append(rect3)
        border.append(rect4)

        return border

    def generate_rectangles(self, n: int, size_range):
        rects = []

        for _ in range(n):
            encloses_start_or_goal = True
            rectangle = Polygon(Point(0, 0))

            while encloses_start_or_goal:
                rectangle = self.generate_random_rectangle(size_range)
                if rectangle is not None and not (
                    self.encloses_point(rectangle, Point(self.start[0], self.start[1]))  # type: ignore
                    or self.encloses_point(rectangle, Point(self.goal[0], self.goal[1]))  # type: ignore
                ):
                    encloses_start_or_goal = False

            rects.append(rectangle)
        return rects

    def encloses_point(self, poly: Polygon, point: Point):
        return poly.encloses_point(point) or poly.intersection(point)

    def generate_random_rectangle(self, size_range):
        # size_range: (min_w, max_w), (min_h, max_h)
        w = np.random.uniform(size_range[0][0], size_range[0][1])
        h = np.random.uniform(size_range[1][0], size_range[1][1])
        x = np.random.uniform(0, self.dimensions[0] - w)

        y = np.random.uniform(0, self.dimensions[1] - h)

        p1, p2, p3, p4 = (
            Point(x, y),
            Point(x + w, y),
            Point(x + w, y + h),
            Point(x, y + h),
        )

        return Polygon(p1, p2, p3, p4)

    def collision_free_path(self, pose1: np.ndarray, pose2: np.ndarray):
        path_segment = Segment(Point(pose1[0], pose1[1]), Point(pose2[0], pose2[1]))
        for i in range(len(self.rectangles)):
            if self.rectangles[i].intersection(path_segment):
                return False
        return True

    def close_to_goal(self, pose: np.ndarray):
        return dist_between_points(pose, self.goal) <= self.goal_radius
