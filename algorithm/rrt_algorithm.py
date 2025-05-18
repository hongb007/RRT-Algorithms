from tree import rrt_tree


class RRT(object):
    def __init__(self, map):
        self.tree = rrt_tree(map)
