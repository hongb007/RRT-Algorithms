from typing import Optional
import numpy as np
from treelib.tree import Tree
from utilities.world_space import space
from utilities.geometry import steer, dist_between_points


class rrt_tree(object):
    def __init__(self, space: space):
        self.tree = Tree()
        self.node_count = 0
        self.space = space
        self.add_node(space.start)

    def add_node(self, pose: np.ndarray):
        self.node_count += 1
        nid = f"Node {self.node_count}"
        steered_pose = pose

        if self.tree.size() == 0:
            self.tree.create_node(nid, nid, data=ArrayHolder(pose))
            return pose, nid
        else:
            parent_node = None
            min_distance = np.finfo(np.float64).max

            for node in self.tree.all_nodes_itr():
                d = dist_between_points(node.data.array, pose)

                # print("Distance between " + str(pose) + " and " + str(node.data.array) + " is " + str(d))

                if (
                    self.space.collision_free_path(pose1=node.data.array, pose2=pose)
                    and d < min_distance
                ):
                    min_distance = d
                    parent_node = node
                    steered_pose = steer(
                        parent_node.data.array, pose, self.space.step_size
                    )

            self.tree.create_node(
                nid,
                nid,
                parent=parent_node,
                data=ArrayHolder(steered_pose),
            )

            return steered_pose, nid

    def path_to_node(self, node_nid: str):
        poses = []
        for nid in self.tree.rsearch(nid=node_nid):
            node = self.tree[nid]  # guaranteed to be a Node
            poses.append(node.data.array)
        return poses

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
