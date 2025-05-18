from typing import Optional
import numpy as np
from treelib.tree import Tree
from utilities.map import map
from utilities.geometry import dist_between_points


class rrt_tree(object):
    def __init__(self, map: map):
        self.tree = Tree()
        self.node_count = 0
        self.map = map
        self.add_node(map.start[0], map.start[1])

    def add_node(self, a: float, b: float):
        self.node_count += 1
        nid = f"Node {self.node_count}"
        pose = np.array([a, b])

        if self.tree.size() == 0:
            self.tree.create_node(nid, nid, data=ArrayHolder(pose))
            return nid
        else:
            parent_node = None
            min_distance = np.finfo(np.float64).max

            for node in self.tree.all_nodes_itr():
                d = dist_between_points(node.data.array, pose)

                # print("Distance between " + str(pose) + " and " + str(node.data.array) + " is " + str(d))

                if (
                    self.map.collision_free_path(pose1=node.data.array, pose2=pose)
                    and d < min_distance
                ):
                    min_distance = d
                    parent_node = node

            self.tree.create_node(
                nid,
                nid,
                parent=parent_node,
                data=ArrayHolder(pose),
            )

            return nid

    def path_to_node(self, node_nid: str):
        return Tree.rsearch(self.tree, nid=node_nid)

    def show(self, data_property: Optional[str] = None):
        return self.tree.show(data_property=data_property)

    def size(self):
        return self.tree.size()


class ArrayHolder:
    def __init__(self, arr):
        self.array = arr

    def __repr__(self):
        # Format the array any way you like
        return np.array2string(self.array, separator=", ")
